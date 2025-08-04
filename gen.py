from diffusers import StableDiffusion3Pipeline
import torch

pipe = StableDiffusion3Pipeline.from_pretrained(
    "./stable-diffusion-3.5-medium",  # ローカルにクローンしたフォルダ
    torch_dtype=torch.bfloat16
)
pipe.to("mps")


image = pipe(
    "A hawk and a rhinoceros engaged in an intense battle",
    num_inference_steps=40,
    guidance_scale=4.5,
    generator=torch.manual_seed(42)
).images[0]

image.save("capybara_hello.jpg")



