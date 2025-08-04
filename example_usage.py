#!/usr/bin/env python3
"""
Stable Diffusion 3.5 高度な画像生成の使用例
"""

from gen import AdvancedImageGenerator


def example_basic_generation():
    """基本的な画像生成の例"""
    print("=== 基本的な画像生成 ===")
    
    generator = AdvancedImageGenerator()
    
    image = generator.generate_image(
        prompt="A beautiful sunset over mountains, digital art style",
        num_inference_steps=30,
        guidance_scale=4.5,
        seed=42,
        output_path="basic_sunset.png"
    )
    print("基本的な画像生成が完了しました")


def example_controlnet_generation():
    """ControlNetを使用した画像生成の例"""
    print("\n=== ControlNetを使用した画像生成 ===")
    
    generator = AdvancedImageGenerator()
    
    # ControlNetを読み込み（実際のパスに変更してください）
    # generator.load_controlnet("./models/controlnet-canny", "canny")
    
    image = generator.generate_image(
        prompt="A beautiful landscape with mountains and trees",
        num_inference_steps=40,
        guidance_scale=4.5,
        seed=42,
        output_path="controlnet_landscape.png",
        # control_image_path="./input/sketch.png",  # 実際のパスに変更してください
        controlnet_conditioning_scale=0.8
    )
    print("ControlNet画像生成が完了しました")


def example_ip_adapter_generation():
    """IP-Adapterを使用した画像生成の例"""
    print("\n=== IP-Adapterを使用した画像生成 ===")
    
    generator = AdvancedImageGenerator()
    
    # IP-Adapterを読み込み（実際のパスに変更してください）
    # generator.load_ip_adapter("./models/ip-adapter")
    
    image = generator.generate_image(
        prompt="A portrait of a person",
        num_inference_steps=40,
        guidance_scale=4.5,
        seed=42,
        output_path="ip_adapter_portrait.png",
        # ip_adapter_image_path="./reference/style_image.jpg",  # 実際のパスに変更してください
        ip_adapter_scale=0.7
    )
    print("IP-Adapter画像生成が完了しました")


def example_lora_generation():
    """LoRAを使用した画像生成の例"""
    print("\n=== LoRAを使用した画像生成 ===")
    
    generator = AdvancedImageGenerator()
    
    # LoRAを読み込み（実際のパスに変更してください）
    # generator.load_lora("./models/anime_style_lora.safetensors", 0.8)
    
    image = generator.generate_image(
        prompt="An anime girl with blue hair and green eyes",
        num_inference_steps=40,
        guidance_scale=4.5,
        seed=42,
        output_path="lora_anime_girl.png"
    )
    print("LoRA画像生成が完了しました")


def example_combined_generation():
    """複数技術を組み合わせた画像生成の例"""
    print("\n=== 複数技術を組み合わせた画像生成 ===")
    
    generator = AdvancedImageGenerator()
    
    # 複数の技術を読み込み（実際のパスに変更してください）
    # generator.load_controlnet("./models/controlnet-canny", "canny")
    # generator.load_ip_adapter("./models/ip-adapter")
    # generator.load_lora("./models/portrait_lora.safetensors", 0.6)
    
    image = generator.generate_image(
        prompt="A realistic portrait in anime style",
        num_inference_steps=50,
        guidance_scale=5.0,
        seed=42,
        output_path="combined_portrait.png",
        # control_image_path="./input/face_sketch.png",  # 実際のパスに変更してください
        # ip_adapter_image_path="./reference/anime_style.jpg",  # 実際のパスに変更してください
        controlnet_conditioning_scale=0.7,
        ip_adapter_scale=0.6
    )
    print("複合技術画像生成が完了しました")


def main():
    """メイン関数"""
    print("Stable Diffusion 3.5 高度な画像生成の使用例")
    print("=" * 50)
    
    try:
        # 基本的な画像生成
        example_basic_generation()
        
        # ControlNetの例（コメントアウトされているため実行されません）
        # example_controlnet_generation()
        
        # IP-Adapterの例（コメントアウトされているため実行されません）
        # example_ip_adapter_generation()
        
        # LoRAの例（コメントアウトされているため実行されません）
        # example_lora_generation()
        
        # 複合技術の例（コメントアウトされているため実行されません）
        # example_combined_generation()
        
        print("\n" + "=" * 50)
        print("すべての例が完了しました！")
        print("\n注意: ControlNet、IP-Adapter、LoRAの例はコメントアウトされています。")
        print("実際に使用する場合は、適切なモデルパスを設定してコメントアウトを解除してください。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main() 