
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image

# デバイス設定（Mac: mps）
device = "cuda" if torch.cuda.is_available() else "mps"

# パイプライン読み込み（SDXL）
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to(device)

# IP-Adapter (SDXL対応モデル)
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)

# 画像の前処理（IP-Adapter用）
ip_image = load_image("reference.png").resize((512, 512))

# プロンプトによる画像生成
prompt = "anime illustration of a man playing baseball"
result = pipe(
    prompt=prompt,
    num_inference_steps=40,
    guidance_scale=4.5,
    ip_adapter_image=ip_image,
    generator=torch.manual_seed(42)
)

# 保存
result.images[0].save("output.png")
