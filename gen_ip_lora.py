import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL  
from diffusers.utils import load_image

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "mps"

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "Linaqruf/animagine-xl-2.0", 
    vae=vae,
    torch_dtype=torch.float16, 
    use_safetensors=True, 
    variant="fp16"
)

# IP-Adapter 読み込み（SDXL）
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)

# ✅ LoRA読み込み（例: pastel-mix）
lora_model_id = "Linaqruf/style-enhancer-xl-lora"
lora_filename = "style-enhancer-xl.safetensors"
lora_scale = 0.6

# Load and fuse LoRA weights
pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
pipe.fuse_lora(lora_scale=lora_scale)

# 画像の前処理
ip_image = load_image("rikka.png").resize((512, 512))

# プロンプトによる画像生成
prompt = "face focus, cute, masterpiece, best quality, 1girl, looking at viewer, upper body"
result = pipe(
    prompt=prompt,
    num_inference_steps=40,
    guidance_scale=4.5,
    ip_adapter_image=ip_image,
    generator=torch.manual_seed(42)
)

# 保存
result.images[0].save("output_lora.png")
