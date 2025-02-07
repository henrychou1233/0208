import os
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
import shutil
from tqdm import tqdm

warnings.filterwarnings("ignore")

def is_valid_image(image: Image.Image, threshold: float = 30.0) -> bool:
    """Check if image is valid (not too dark or too small)"""
    if image is None:
        return False
        
    # Convert to grayscale and calculate mean brightness
    gray = image.convert('L')
    brightness = np.mean(np.array(gray))
    
    # Check brightness and size
    if brightness < threshold or image.size[0] < 32 or image.size[1] < 32:
        return False
        
    return True

class ImageAugmentor:
    @staticmethod
    def apply_color_transforms(image: Image.Image) -> Image.Image:
        """Apply color adjustments"""
        try:
            # Enhance contrast
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(1.2)
            
            # Enhance brightness
            brightness = ImageEnhance.Brightness(image)
            image = brightness.enhance(1.1)
            
            return image
        except Exception as e:
            logging.error(f"Color transform error: {str(e)}")
            return image

    @staticmethod
    def apply_geometric_transforms(image: Image.Image) -> Image.Image:
        """Apply geometric transformations"""
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
        self.models = {
            "sd": None,
            "upscaler": None
        }

    def _load_model(self, model_type: str) -> bool:
        """Load model"""
        if self.models[model_type] is not None:
            return True

        try:
            if model_type == "sd":
                model = StableDiffusionImg2ImgPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    variant="fp16",
                    safety_checker=None
                )
            else:
                model = StableDiffusionUpscalePipeline.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    variant="fp16",
                    safety_checker=None
                )

            model.to(self.device)
            if self.device == "cuda":
                model.enable_attention_slicing()
                model.enable_vae_slicing()

            self.models[model_type] = model
            return True
        except Exception as e:
            logging.error(f"Failed to load {model_type} model: {str(e)}")
            return False

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for model input"""
        if max(image.size) > 512:
            image = image.resize((512, 512), Image.LANCZOS)
        return image

    @torch.no_grad()
    def upscale_image(self, image: Image.Image) -> Image.Image:
        """Upscale image using SD upscaler"""
        try:
            with autocast() if self.device == "cuda" else nullcontext():
                result = self.models["upscaler"](
                    prompt="high quality photo, sharp details",
                    image=image,
                    noise_level=20,
                    num_inference_steps=20
                ).images[0]
                
            if is_valid_image(result):
                return result
            return image
        except Exception as e:
            logging.error(f"Upscaling error: {str(e)}")
            return image

    @torch.no_grad()
    def apply_stable_diffusion(self, image: Image.Image) -> Image.Image:
        """Apply Stable Diffusion augmentation"""
        try:
            with autocast() if self.device == "cuda" else nullcontext():
                result = self.models["sd"](
                    prompt="high quality photo, same as input, sharp, clear details",
                    image=image,
                    strength=0.3,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    negative_prompt="blur, dark, black, deformed, bad quality"
                ).images[0]
                
            if is_valid_image(result):
                return result
            return image
        except Exception as e:
            logging.error(f"SD augmentation error: {str(e)}")
            return image

    def process_single_image(self, input_path: Path, output_dir: Path, pbar: tqdm) -> bool:
        """Process a single image according to the workflow"""
        try:
            # Create output subdirectory using input filename
            image_output_dir = output_dir / input_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load and preprocess image
            img = Image.open(input_path).convert('RGB')
            if not is_valid_image(img):
                logging.warning(f"Skipping invalid image: {input_path}")
                pbar.update(1)
                return False
                
            img = self.preprocess_image(img)
            
            # Save original
            img.save(image_output_dir / "1_original.png")
            
            # Original -> Upscale
            if self._load_model("upscaler"):
                upscaled = self.upscale_image(img)
                upscaled.save(image_output_dir / "2_original_upscaled.png")
            
            # Color aug -> Upscale
            color_aug = ImageAugmentor.apply_color_transforms(img)
            if self._load_model("upscaler"):
                color_upscaled = self.upscale_image(color_aug)
                color_upscaled.save(image_output_dir / "3_color_upscaled.png")
            
            # Geometric aug -> Upscale
            geometric_aug = ImageAugmentor.apply_geometric_transforms(img)
            if self._load_model("upscaler"):
                geometric_upscaled = self.upscale_image(geometric_aug)
                geometric_upscaled.save(image_output_dir / "4_geometric_upscaled.png")
            
            # SD aug -> Upscale
            if self._load_model("sd"):
                sd_aug = self.apply_stable_diffusion(img)
                if self._load_model("upscaler"):
                    sd_upscaled = self.upscale_image(sd_aug)
                    sd_upscaled.save(image_output_dir / "5_sd_upscaled.png")
            
            logging.info(f"Successfully processed: {input_path}")
            pbar.update(1)
            return True
            
        except Exception as e:
            logging.error(f"Error processing {input_path}: {str(e)}")
            pbar.update(1)
            return False
            
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def reorganize_output(output_path: Path, final_output_path: Path):
    """Reorganize and rename all processed images"""
    # Create final output directory
    final_output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all processed images
    all_images = []
    for subdir in output_path.glob("*"):
        if subdir.is_dir():
            all_images.extend(sorted(subdir.glob("*.png")))
    
    # Sort images to ensure consistent ordering
    all_images.sort()
    
    # Copy and rename files with progress bar
    with tqdm(total=len(all_images), desc="Reorganizing files", unit="file") as pbar:
        for idx, src_path in enumerate(all_images):
            dst_path = final_output_path / f"{idx:03d}.png"
            try:
                shutil.copy2(src_path, dst_path)
                pbar.update(1)
            except Exception as e:
                logging.error(f"Failed to copy file {src_path}: {str(e)}")
    
    logging.info(f"Reorganized {len(all_images)} images to {final_output_path}")

def process_directory(input_dir: str, output_dir: str, final_dir: str, category_idx: int):
    """Process all images in directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    final_output_path = Path(final_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files
    image_files = list(input_path.glob("*.png"))
    total_files = len(image_files)
    
    if total_files == 0:
        logging.warning(f"No PNG files found in {input_dir}")
        return
    
    logging.info(f"Found {total_files} PNG files to process")
    
    # Process images
    processor = ImageProcessor()
    success_count = 0
    
    # Create progress bar for this category
    with tqdm(total=total_files, 
              desc=f"Category {category_idx}/3", 
              unit="img",
              position=0, 
              leave=True) as pbar:
        
        for idx, image_path in enumerate(image_files, 1):
            if processor.process_single_image(image_path, output_path, pbar):
                success_count += 1
    
    success_rate = (success_count / total_files) * 100
    logging.info(f"\nProcessing complete!")
    logging.info(f"Success rate: {success_rate:.1f}% ({success_count}/{total_files})")
    
    # Reorganize files
    reorganize_output(output_path, final_output_path)

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('augmentation.log')
        ]
    )
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Define the directory pairs for all three categories
    directory_pairs = [
        {
            "input": "/home/anywhere3090l/Desktop/compalmtk/Dynamic-noise-AD-master/dataset/btad/one/train/good",
            "output": "/home/anywhere3090l/Desktop/compalmtk/Dynamic-noise-AD-master/dataset/btad/newone/train/good"
        },
        {
            "input": "/home/anywhere3090l/Desktop/compalmtk/Dynamic-noise-AD-master/dataset/btad/two/train/good",
            "output": "/home/anywhere3090l/Desktop/compalmtk/Dynamic-noise-AD-master/dataset/btad/newtwo/train/good"
        },
        {
            "input": "/home/anywhere3090l/Desktop/compalmtk/Dynamic-noise-AD-master/dataset/btad/three/train/good",
            "output": "/home/anywhere3090l/Desktop/compalmtk/Dynamic-noise-AD-master/dataset/btad/newthree/train/good"
        }
    ]
    
    # Create overall progress bar for categories
    with tqdm(total=len(directory_pairs), 
              desc="Overall Progress", 
              unit="category",
              position=1, 
              leave=True) as category_pbar:
        
        # Process each category
        for idx, dirs in enumerate(directory_pairs, 1):
            logging.info(f"\nProcessing Category {idx}/3")
            logging.info(f"Input directory: {dirs['input']}")
            logging.info(f"Output directory: {dirs['output']}")
            
            # Create temporary directory for this category
            temp_dir = Path(dirs['output']) / "temp"
            final_dir = Path(dirs['output'])
            
            # Process images for this category
            process_directory(dirs['input'], temp_dir, final_dir, idx)
            
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logging.error(f"Failed to clean up temporary directory: {str(e)}")
            
            # Clear GPU memory between categories
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            category_pbar.update(1)
            logging.info(f"Completed processing category {idx}/3\n")

if __name__ == "__main__":
    main()