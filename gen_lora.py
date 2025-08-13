import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from diffusers import StableDiffusionPipeline
from PIL import Image

# 1. ベースモデルとLoRAのIDまたはパスを設定
# 使用するベースモデルのID
# SDXLを使用する場合
# base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# SD1.5を使用する場合
base_model_id = "runwayml/stable-diffusion-v1-5"

# 使用するLoRAのID (例として、Hugging Face HubのピクセルアートLoRAを使用)
# お手元のLoRAを使用する場合は、ローカルファイルのパスを指定してください
# lora_path = "/path/to/your/lora_model.safetensors"
lora_path = "rusty2930/hidream-pixel-art-lora"



# 2. パイプラインをロード
# SDXLを使用する場合
# pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16")
# SD1.5を使用する場合
pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)

# GPUが利用可能であればGPUに移動
pipe = pipe.to("mps")

# 3. LoRAをロード
# LoRAのパスを指定してweightsをロード
pipe.load_lora_weights(lora_path)

# 4. プロンプトと生成パラメータを設定
prompt = "pixel art style, a cute cat wearing a wizard hat, magical sparks, detailed"
negative_prompt = "blurry, low quality, noise, deformed, bad anatomy, ugly"

# LoRAの強さ (0.0から1.0の間で調整)
lora_scale = 0.8

# 5. 画像を生成
# SDXLを使用する場合
# generated_image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     cross_attention_kwargs={"scale": lora_scale}
# ).images[0]

# SD1.5を使用する場合
generated_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cross_attention_kwargs={"scale": lora_scale}
).images[0]

# 6. 生成された画像を保存
generated_image.save("generated_image_with_lora.png")
print("画像が正常に生成されました: generated_image_with_lora.png")