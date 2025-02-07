import os
import shutil
from pathlib import Path
import logging
import torch
from PIL import Image, ImageEnhance
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import gc
from diffusers import StableDiffusionUpscalePipeline, StableDiffusionImg2ImgPipeline
from torch.cuda.amp import autocast
from contextlib import nullcontext
import random
import warnings
import time

warnings.filterwarnings("ignore")

# 參數設定：降低推論步數以加速處理（依需求調整）
UPSCALE_INFERENCE_STEPS = 10   # 原本20步，降低至10步
SD_INFERENCE_STEPS = 20        # 原本30步，降低至20步

def is_valid_image(image: Image.Image, threshold: float = 30.0) -> bool:
    """檢查圖片是否有效（不過暗且尺寸夠大）"""
    if image is None:
        return False
    # 轉換為灰階並計算平均亮度
    gray = image.convert('L')
    brightness = np.mean(np.array(gray))
    if brightness < threshold or image.size[0] < 32 or image.size[1] < 32:
        return False
    return True

class ImageAugmentor:
    @staticmethod
    def apply_color_transforms(image: Image.Image) -> Image.Image:
        """執行色彩調整"""
        try:
            # 增強對比度
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(1.2)
            # 增強亮度
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(1.1)
            return image
        except Exception as e:
            logging.error(f"Color transform error: {str(e)}")
            return image

    @staticmethod
    def apply_geometric_transforms(image: Image.Image) -> Image.Image:
        """執行幾何變換"""
        try:
            transform = T.Compose([
                T.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=255
                ),
                T.RandomPerspective(
                    distortion_scale=0.2,
                    p=0.5,
                    fill=255
                )
            ])
            img_tensor = T.ToTensor()(image)
            augmented = transform(img_tensor)
            return T.ToPILImage()(augmented)
        except Exception as e:
            logging.error(f"Geometric transform error: {str(e)}")
            return image

class ImageProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sd_model = None
        self.upscaler_model = None
        self.load_models()

    def load_models(self):
        """在初始化時只載入一次模型"""
        try:
            self.sd_model = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16",
                safety_checker=None
            )
            self.sd_model.to(self.device)
            if self.device == "cuda":
                self.sd_model.enable_attention_slicing()
                self.sd_model.enable_vae_slicing()
            logging.info("Loaded SD model successfully.")
        except Exception as e:
            logging.error(f"Failed to load SD model: {str(e)}")

        try:
            self.upscaler_model = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16",
                safety_checker=None
            )
            self.upscaler_model.to(self.device)
            if self.device == "cuda":
                self.upscaler_model.enable_attention_slicing()
                self.upscaler_model.enable_vae_slicing()
            logging.info("Loaded Upscaler model successfully.")
        except Exception as e:
            logging.error(f"Failed to load Upscaler model: {str(e)}")

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """預處理圖片（例如調整尺寸）"""
        if max(image.size) > 512:
            image = image.resize((512, 512), Image.LANCZOS)
        return image

    @torch.no_grad()
    def upscale_image(self, image: Image.Image) -> Image.Image:
        """使用上采樣模型進行圖片放大"""
        try:
            with autocast() if self.device == "cuda" else nullcontext():
                result = self.upscaler_model(
                    prompt="high quality photo, sharp details",
                    image=image,
                    noise_level=20,
                    num_inference_steps=UPSCALE_INFERENCE_STEPS
                ).images[0]
            if is_valid_image(result):
                return result
            return image
        except Exception as e:
            logging.error(f"Upscaling error: {str(e)}")
            return image

    @torch.no_grad()
    def apply_stable_diffusion(self, image: Image.Image) -> Image.Image:
        """使用 Stable Diffusion 進行圖片增強"""
        try:
            with autocast() if self.device == "cuda" else nullcontext():
                result = self.sd_model(
                    prompt="high quality photo, same as input, sharp, clear details",
                    image=image,
                    strength=0.3,
                    guidance_scale=7.5,
                    num_inference_steps=SD_INFERENCE_STEPS,
                    negative_prompt="blur, dark, black, deformed, bad quality"
                ).images[0]
            if is_valid_image(result):
                return result
            return image
        except Exception as e:
            logging.error(f"SD augmentation error: {str(e)}")
            return image

    def process_single_image(self, input_path: Path, output_dir: Path) -> bool:
        """對單張圖片進行完整的處理流程"""
        try:
            # 建立以檔名命名的輸出子目錄
            image_output_dir = output_dir / input_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 讀取並預處理圖片
            img = Image.open(input_path).convert('RGB')
            if not is_valid_image(img):
                logging.warning(f"Skipping invalid image: {input_path}")
                return False
            img = self.preprocess_image(img)
            
            # 儲存原始圖片
            img.save(image_output_dir / "1_original.png")
            
            # 原始圖片上采樣
            upscaled = self.upscale_image(img)
            upscaled.save(image_output_dir / "2_original_upscaled.png")
            
            # 色彩增強 -> 上采樣
            color_aug = ImageAugmentor.apply_color_transforms(img)
            color_upscaled = self.upscale_image(color_aug)
            color_upscaled.save(image_output_dir / "3_color_upscaled.png")
            
            # 幾何增強 -> 上采樣
            geometric_aug = ImageAugmentor.apply_geometric_transforms(img)
            geometric_upscaled = self.upscale_image(geometric_aug)
            geometric_upscaled.save(image_output_dir / "4_geometric_upscaled.png")
            
            # SD 增強 -> 上采樣
            sd_aug = self.apply_stable_diffusion(img)
            sd_upscaled = self.upscale_image(sd_aug)
            sd_upscaled.save(image_output_dir / "5_sd_upscaled.png")
            
            logging.info(f"Successfully processed: {input_path}")
            return True
        except Exception as e:
            logging.error(f"Error processing {input_path}: {str(e)}")
            return False

def reorganize_output(output_path: Path, final_output_path: Path):
    """重組並重新命名所有處理後的圖片"""
    final_output_path.mkdir(parents=True, exist_ok=True)
    all_images = []
    for subdir in output_path.glob("*"):
        if subdir.is_dir():
            all_images.extend(sorted(subdir.glob("*.png")))
    all_images.sort()
    for idx, src_path in enumerate(all_images):
        dst_path = final_output_path / f"{idx:03d}.png"
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            logging.error(f"Failed to copy file {src_path}: {str(e)}")
    logging.info(f"Reorganized {len(all_images)} images to {final_output_path}")

def process_directory(input_dir: str, output_dir: str, final_dir: str):
    """處理資料夾中的所有圖片"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    final_output_path = Path(final_dir)
    
    # 建立輸出目錄
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 取得所有 PNG 檔案
    image_files = list(input_path.glob("*.png"))
    total_files = len(image_files)
    
    if total_files == 0:
        logging.warning(f"No PNG files found in {input_dir}")
        return
    
    logging.info(f"Found {total_files} PNG files to process")
    
    processor = ImageProcessor()
    success_count = 0
    start_time = time.time()
    
    for idx, image_path in enumerate(image_files, 1):
        logging.info(f"Processing image {idx}/{total_files}: {image_path}")
        if processor.process_single_image(image_path, output_path):
            success_count += 1
        # 每處理 10 張圖片做一次垃圾回收與 GPU 清理
        if idx % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    end_time = time.time()
    success_rate = (success_count / total_files) * 100
    logging.info(f"\nProcessing complete!")
    logging.info(f"Success rate: {success_rate:.1f}% ({success_count}/{total_files})")
    logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    reorganize_output(output_path, final_output_path)

def main():
    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('augmentation.log')
        ]
    )
    
    # 設定隨機種子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 設定輸入與輸出目錄（注意：輸出目錄下會先建立一個 temp 子目錄）
    input_dir = "/home/anywhere3090l/Desktop/compalmtk/Dynamic-noise-AD-master/dataset/btad/two/train/good"  # 輸入目錄
    output_dir = "/home/anywhere3090l/Desktop/compalmtk/Dynamic-noise-AD-master/dataset/btad/newtwo/train/good/temp"  # 臨時輸出目錄
    final_dir = "/home/anywhere3090l/Desktop/compalmtk/Dynamic-noise-AD-master/dataset/btad/newtwo/train/good"  # 最終輸出目錄
    
    process_directory(input_dir, output_dir, final_dir)
    
    # 處理完畢後清理臨時目錄
    try:
        shutil.rmtree(Path(output_dir))
        logging.info(f"Cleaned up temporary directory: {output_dir}")
    except Exception as e:
        logging.error(f"Failed to clean up temporary directory: {str(e)}")

if __name__ == "__main__":
    main()
