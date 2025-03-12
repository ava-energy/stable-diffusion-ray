from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import Response
import torch
from ray import serve
import ray

app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle) -> None:
        self.handle = diffusion_model_handle

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
