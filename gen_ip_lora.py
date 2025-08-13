import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image

# 1. ベースモデルとLoRAのIDまたはパスを設定
base_model_id = "./models/nova_xl"
lora_path = "./models/bocchi_lora"

# 2. パイプラインをロード
pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("mps")

# 3. LoRAとSD1.5対応のIP-Adapterをロード
pipe.load_lora_weights(lora_path)

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models", # SD1.5モデルが格納されているサブフォルダー
    weight_name="ip-adapter_sd15.bin" # SD1.5向けのweightファイル
)

ip_image = load_image("rikka.png").resize((512, 512))

# 4. プロンプトと生成パラメータを設定
prompt = "pixel art style, a cute girl wearing a wizard stick, magical sparks, detailed"
negative_prompt = "blurry, low quality, noise, deformed, bad anatomy, ugly"

lora_scale = 0.8

# 5. 画像を生成
generated_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cross_attention_kwargs={"scale": lora_scale},
    ip_adapter_image=ip_image,
).images[0]

# 6. 生成された画像を保存
generated_image.save("generated_image_with_lora_ip.png")
print("画像が正常に生成されました: generated_image_with_lora_ip.png")