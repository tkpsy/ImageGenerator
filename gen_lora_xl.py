import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image

# 1. ローカルのベースモデルとLoRAのパスを設定
# CivitaiからダウンロードしたSDXLベースモデルのパスを指定してください。
# 例: "./models/Illustrious_V1.safetensors"
base_model_path = "./models/nova_xl.safetensors"

# CivitaiからダウンロードしたLoRAファイルのパスを指定してください。
# 例: "./lora/your_lora.safetensors"
lora_path = "./models/bocchi_lora.safetensors"


# 2. パイプラインをロード
# ベースモデルは、シングルファイル形式(safetensors)で提供されることが多いため、from_single_fileを使用します。
pipe = StableDiffusionXLPipeline.from_single_file(
    base_model_path,
    torch_dtype=torch.float16
)

# GPUが利用可能であればGPUに移動
pipe.to("mps") # または Apple Silicon Mac の場合は "mps"


# 3. LoRAをロード
# LoRAのパスを指定してweightsをロードします。
pipe.load_lora_weights(lora_path)


# 4. プロンプトと生成パラメータを設定
# ここに生成したい内容を日本語でも、英語でも記述します。
# LoRAによっては、Civitaiのページに記載されている「トリガーワード」が必要です。
# 例: 'your_trigger_word, a beautiful anime girl, detailed background'
prompt = "ijichi_nijika, flowing hair, vibrant colors, sitting on a park bench"
negative_prompt = "blurry, low quality, noise, deformed, bad anatomy, ugly"

# LoRAの強さを調整します。(0.0から1.0の間で)
# 1.0に近いほどLoRAの効果が強くなります。
lora_scale = 0.8

# 5. 画像を生成
# SDXLは推奨画像サイズが1024x1024です。
generated_image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    cross_attention_kwargs={"scale": lora_scale},
    width=1024,
    height=1024,
).images[0]


# 6. 生成された画像を保存
generated_image.save("generated_image_sdxl_civitai.png")
print("画像が正常に生成されました: generated_image_sdxl_civitai.png")