# Stable Diffusion 3.5 高度な画像生成ツール

このプロジェクトは、Stable Diffusion 3.5を使用して、ControlNet、IP-Adapter、LoRA、T2I-Adapterなどの高度な制御技術を組み合わせた画像生成を行うためのPythonツールです。

## 特徴

- **Stable Diffusion 3.5** - 最新の高品質画像生成モデル
- **ControlNet** - 構造制御による精密な画像生成
- **IP-Adapter** - 参照画像からのスタイル転写
- **LoRA** - 軽量なファインチューニング
- **T2I-Adapter** - 条件付き画像生成
- **複数技術の組み合わせ** - 複数の制御技術を同時に使用可能

## セットアップ

### 1. 依存関係のインストール

```bash
pip install diffusers transformers torch torchvision accelerate safetensors Pillow numpy tiktoken protobuf
```

### 2. モデルの準備

- **ベースモデル**: Stable Diffusion 3.5 Medium
- **ControlNet**: 各種ControlNetモデル（canny、depth、pose等）
- **IP-Adapter**: IP-Adapterモデル
- **T2I-Adapter**: T2I-Adapterモデル
- **LoRA**: 各種LoRAモデル

## 使用方法

### 基本的な画像生成

```bash
python gen.py "a beautiful sunset over mountains"
```

### ControlNetを使用した画像生成

```bash
python gen.py "a beautiful landscape" \
    --controlnet "./models/controlnet-canny" \
    --control-image "./input/sketch.png" \
    --control-scale 1.0
```

### IP-Adapterを使用したスタイル転写

```bash
python gen.py "a portrait of a person" \
    --ip-adapter "./models/ip-adapter" \
    --ip-image "./reference/style_image.jpg" \
    --ip-scale 0.8
```

### LoRAを使用したファインチューニング

```bash
python gen.py "anime girl with blue hair" \
    --lora "./models/anime_lora.safetensors" \
    --lora-weight 0.7
```

### 複数技術の組み合わせ

```bash
python gen.py "a realistic portrait in anime style" \
    --controlnet "./models/controlnet-canny" \
    --control-image "./input/face_sketch.png" \
    --control-scale 0.8 \
    --ip-adapter "./models/ip-adapter" \
    --ip-image "./reference/anime_style.jpg" \
    --ip-scale 0.6 \
    --lora "./models/portrait_lora.safetensors" \
    --lora-weight 0.5
```

## 利用可能なオプション

### 基本オプション

- `prompt` - 生成したい画像の説明（必須）
- `--negative-prompt` - 避けたい要素の説明
- `--steps` - 推論ステップ数（デフォルト: 40）
- `--guidance-scale` - ガイダンススケール（デフォルト: 4.5）
- `--seed` - 乱数シード（再現性のため）
- `--output` - 出力ファイルパス（デフォルト: generated_image.png）
- `--model-path` - ベースモデルのパス
- `--device` - 使用するデバイス（auto/cpu/cuda/mps）

### ControlNet関連

- `--controlnet` - ControlNetモデルのパス
- `--controlnet-type` - ControlNetの種類（canny, depth, pose等）
- `--control-image` - ControlNet用の制御画像パス
- `--control-scale` - ControlNetの制御強度（0.0-1.0）

### IP-Adapter関連

- `--ip-adapter` - IP-Adapterモデルのパス
- `--ip-image` - IP-Adapter用の参照画像パス
- `--ip-scale` - IP-Adapterの制御強度（0.0-1.0）

### T2I-Adapter関連

- `--t2i-adapter` - T2I-Adapterモデルのパス
- `--t2i-adapter-type` - T2I-Adapterの種類（sketch, depth等）
- `--t2i-image` - T2I-Adapter用の制御画像パス
- `--t2i-scale` - T2I-Adapterの制御強度（0.0-1.0）

### LoRA関連

- `--lora` - LoRAモデルのパス（複数指定可能）
- `--lora-weight` - LoRAの重み（0.0-1.0）

## 技術の説明

### ControlNet
- **用途**: 構造制御、ポーズ制御、深度制御
- **入力**: 制御画像（スケッチ、深度マップ、ポーズ等）
- **効果**: 画像の構造や構図を精密に制御

### IP-Adapter
- **用途**: スタイル転写、参照画像からの特徴抽出
- **入力**: 参照画像
- **効果**: 参照画像のスタイルや特徴を生成画像に反映

### LoRA
- **用途**: 軽量ファインチューニング、特定スタイルの学習
- **入力**: ファインチューニング済みLoRAモデル
- **効果**: 特定のスタイルやキャラクターを生成画像に適用

### T2I-Adapter
- **用途**: 条件付き画像生成、構造制御
- **入力**: 制御画像
- **効果**: 画像の構造や条件を制御しながら生成

## 使用例

### 1. アニメ風ポートレート生成

```bash
python gen.py "anime girl portrait" \
    --controlnet "./models/controlnet-canny" \
    --control-image "./input/face_outline.png" \
    --control-scale 0.7 \
    --lora "./models/anime_style_lora.safetensors" \
    --lora-weight 0.8 \
    --output "anime_portrait.png"
```

### 2. スタイル転写

```bash
python gen.py "a landscape painting" \
    --ip-adapter "./models/ip-adapter" \
    --ip-image "./reference/van_gogh_style.jpg" \
    --ip-scale 0.9 \
    --steps 50 \
    --guidance-scale 5.0 \
    --output "van_gogh_landscape.png"
```

### 3. 複合制御

```bash
python gen.py "a realistic portrait in cyberpunk style" \
    --controlnet "./models/controlnet-depth" \
    --control-image "./input/face_depth.png" \
    --control-scale 0.6 \
    --ip-adapter "./models/ip-adapter" \
    --ip-image "./reference/cyberpunk_style.jpg" \
    --ip-scale 0.7 \
    --lora "./models/realistic_portrait_lora.safetensors" \
    --lora-weight 0.5 \
    --output "cyberpunk_portrait.png"
```

## パフォーマンスのヒント

1. **メモリ使用量**: 複数の制御技術を使用する場合、メモリ使用量が増加します
2. **生成時間**: 制御技術が多いほど生成時間が長くなります
3. **制御強度**: 各技術の制御強度を調整して最適な結果を得てください
4. **組み合わせ**: 技術の組み合わせによっては競合する場合があります

## トラブルシューティング

### メモリ不足エラー

```bash
# より小さなバッチサイズで試してください
# または制御技術の数を減らしてください
```

### モデル読み込みエラー

```bash
# モデルパスが正しいか確認してください
# 必要な依存関係がインストールされているか確認してください
```

### 制御強度の調整

```bash
# 各技術の制御強度を0.0-1.0の範囲で調整してください
# 強すぎる制御は不自然な結果になる場合があります
```

## 注意事項

- 生成された画像の著作権は利用者に帰属します
- 不適切なコンテンツの生成は避けてください
- 商用利用の際は各モデルのライセンスを確認してください
- 制御技術の組み合わせは実験的な機能です 