from io import BytesIO
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import Response
import torch
from ray import serve
import ray

import asyncio
from typing import List, Dict
import uuid
import json
import base64
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

class BatchRequest(BaseModel):
    prompts: List[str]
    img_size: int = 512  # Default value

app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle) -> None:
        self.handle = diffusion_model_handle
        self.batch_progress: Dict[str, asyncio.Queue] = {}

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        # Get Ray worker metadata
        runtime_context = ray.get_runtime_context()
        worker_metadata = {
            "node_id": str(runtime_context.node_id)
        }

        # Generate image
        image = await self.handle.generate.remote(prompt, img_size=img_size)
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        file_stream.seek(0)

        # Prepare custom headers with metadata
        headers = {
            "X-Ray-Node-Id": worker_metadata["node_id"]
        }

        return Response(
            content=file_stream.getvalue(),
            media_type="image/png",
            headers=headers,
        )

    @app.post("/imagine/batch")
    async def start_batch_generation(self, request: BatchRequest):
        batch_id = str(uuid.uuid4())
        # Create queue for this batch
        self.batch_progress[batch_id] = asyncio.Queue()
        
        # Start background generation task
        asyncio.create_task(
            self._generate_batch(batch_id, request.prompts, request.img_size)
        )
        
        return {"batch_id": batch_id, "total_images": len(request.prompts)}

    async def _generate_batch(self, batch_id: str, prompts: List[str], img_size: int):
        try:
            # Generate images concurrently but process results as they complete
            tasks = [
                self._generate_and_notify(batch_id, i, prompt, img_size)
                for i, prompt in enumerate(prompts)
            ]
            
            # Wait for all generations to complete
            await asyncio.gather(*tasks)
            
            # Publish final completion event to PubSub
        finally:
            # Cleanup
            await self.batch_progress[batch_id].put(None)  # Signal completion
            del self.batch_progress[batch_id]

    async def _generate_and_notify(self, batch_id: str, index: int, prompt: str, img_size: int):
        try:
            # Generate single image
            image = await self.handle.generate.remote(prompt, img_size=img_size)
            
            # Convert to bytes
            image_bytes = self._image_to_bytes(image)
            
            # Put result in queue for SSE
            await self.batch_progress[batch_id].put({
                "index": index,
                "prompt": prompt,
                "image": image_bytes
            })
            
        except Exception as e:
            # Handle errors
            await self.batch_progress[batch_id].put({
                "index": index,
                "prompt": prompt,
                "error": str(e)
            })

    @app.get("/imagine/batch/{batch_id}")
    async def batch_progress_endpoint(self, batch_id: str, request: Request):
        if batch_id not in self.batch_progress:
            raise HTTPException(status_code=404, detail=f"Batch \"{batch_id}\" not found")

        async def event_generator():
            queue = self.batch_progress[batch_id]
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                # Get next result
                result = await queue.get()
                if result is None:  # End signal
                    break

                # Send event
                if "error" in result:
                    yield {
                        "event": "error",
                        "data": json.dumps({
                            "index": result["index"],
                            "prompt": result["prompt"],
                            "error": result["error"]
                        })
                    }
                else:
                    yield {
                        "event": "image",
                        "data": json.dumps({
                            "index": result["index"],
                            "prompt": result["prompt"],
                            "image": base64.b64encode(result["image"]).decode()
                        })
                    }

        return EventSourceResponse(event_generator())

    def _image_to_bytes(self, image) -> bytes:
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        file_stream.seek(0)
        return file_stream.getvalue()

    @app.get(
        "/health",
        response_class=Response,
    )
    async def health(self):
        return Response(
            content="OK",
            media_type="text/plain",
        )


@serve.deployment(ray_actor_options={"num_gpus": 1},)
class StableDiffusionV2:
    def __init__(self):
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

        model_id = "stabilityai/stable-diffusion-2"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()

    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        with torch.autocast("cuda"):
            image = self.pipe(prompt, height=img_size, width=img_size).images[0]
            return image


entrypoint = APIIngress.bind(StableDiffusionV2.bind())
