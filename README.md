
# 🧩 Hugging Face の Stable Diffusion 3.5 モデルをローカルにダウンロードする手順（Git + Git LFS）

## 🔧 前提条件

* Hugging Face アカウントがある（[https://huggingface.co](https://huggingface.co)）
* モデルにアクセス可能（例: `stabilityai/stable-diffusion-3.5-medium` に access request 済み）
* macOS (Apple Silicon 含む)、Python 環境が整っている
* `git`, `git-lfs`, `huggingface_hub`, `transformers`, `diffusers`, `torch` などがインストール済み

---

## ✅ ステップ 1: Hugging Face のアクセストークンを取得

1. Hugging Face にログイン: [https://huggingface.co/login](https://huggingface.co/login)
2. アクセストークンを生成: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. 「read」権限を持つトークンをコピーしておく

---

## ✅ ステップ 2: Git LFS をインストール（初回のみ）

```bash
brew install git-lfs  # Homebrew 経由（macOSの場合）
git lfs install       # 初期化
```

---

## ✅ ステップ 3: Hugging Face にログイン（トークン使用）

```bash
huggingface-cli login
# → ここで取得したトークンを入力
```

---

## ✅ ステップ 4: モデルを Git LFS 経由でクローン

```bash
git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
cd stable-diffusion-3.5-medium
```

> ⚠️ クローン直後は LFS 管理ファイルがダミーのままなので、次のステップで本体を取得します。

---

## ✅ ステップ 5: LFSオブジェクトをすべてダウンロード

```bash
git lfs pull
```

※ ここで `.safetensors`, `.bin`, `.pt` などの大容量ファイルがすべて取得されます。

---

## ✅ 確認: ファイルが正しくダウンロードされたかチェック

```bash
ls -lh
# モデルの重みファイルなどが表示されていればOK
```

---

## ✅ Python スクリプトからローカルモデルを使う例

```python
from diffusers import StableDiffusion3Pipeline
import torch

pipe = StableDiffusion3Pipeline.from_pretrained(
    "./stable-diffusion-3.5-medium",  # クローンしたフォルダを指定
    torch_dtype=torch.float32         # Apple Siliconの場合（bfloat16やcudaは不可）
)
pipe.to("mps")  # Mシリーズ Mac の GPU (Metal) を使う場合

image = pipe("A hawk and a rhinoceros engaged in an intense battle").images[0]
image.save("hawk_vs_rhino.jpg")
```

---

## ✅ 補足

* `.gitattributes` に `*.safetensors`, `*.bin` などが `filter=lfs` として定義されていることを確認しておく
* クローン中や `lfs pull` 中にエラーが出たら、`huggingface-cli login` の再確認を

---

## 🎉 完了！

これで、Stable Diffusion 3.5 モデルを Hugging Face からローカルに完全ダウンロードし、Python スクリプトで利用できるようになります。

