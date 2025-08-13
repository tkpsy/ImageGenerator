import cv2
from insightface.app import FaceAnalysis
import torch

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

image = cv2.imread("reference.png")
faces = app.get(image)

faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)



import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from PIL import Image

from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

base_model_path = "SG161222/RealVisXL_V3.0"
ip_ckpt = "models/ip-adapter-faceid_sdxl.bin"
device = "mps"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    add_watermarker=False,
)

# load ip-adapter
ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

# generate image
prompt = "crying face, expressive watercolor painting, soft bleeding edges, dynamic splashes of color, delicate brushwork, fine art illustration, high detail, award-winning painting style"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

images = ip_model.generate(
    prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=2,
    width=1024, height=1024,
    num_inference_steps=30, guidance_scale=7.5, seed=2023
)


images[0].save("result.png")