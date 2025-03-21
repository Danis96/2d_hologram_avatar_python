#!/usr/bin/env python3
import os
import sys
import argparse
import base64
import io
from typing import Tuple, Dict, Any, List, Optional, Union
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from models.generator import Generator
from utils.transforms import get_no_aug_transform

class CartoonAvatarGenerator:
    """A class to generate cartoon style avatars from images."""
    
    def __init__(self, 
                 checkpoint_path: str = "./checkpoints/trained_netG.pth", 
                 device: str = "cpu"):
        """
        Initialize the Cartoon Avatar Generator.
        
        Args:
            checkpoint_path (str): Path to the pretrained model weights
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.netG = self._load_model()
    
    def _load_model(self) -> torch.nn.Module:
        """Load the generator model with pretrained weights."""
        try:
            netG = Generator().to(self.device)
            netG.eval()
            
            if self.device.type == "cuda":
                netG.load_state_dict(torch.load(self.checkpoint_path))
            else:
                netG.load_state_dict(torch.load(self.checkpoint_path, map_location=torch.device('cpu')))
                
            return netG
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from a file path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Image.Image: The loaded PIL image
        """
        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image: Image.Image, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Preprocess the image for the model.
        
        Args:
            image (Image.Image): Input image
            size (Tuple[int, int]): Target size for resizing
            
        Returns:
            Image.Image: Preprocessed image
        """
        # Resize while maintaining aspect ratio
        width, height = image.size
        ratio = min(size[0] / width, size[1] / height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        
        # Create a blank background
        background = Image.new('RGB', size, (255, 255, 255))
        
        # Paste the resized image onto the background
        offset = ((size[0] - new_size[0]) // 2, (size[1] - new_size[1]) // 2)
        background.paste(image, offset)
        
        return background
    
    def encode_image_to_base64(self, image: Image.Image) -> str:
        """
        Encode an image to base64 string.
        
        Args:
            image (Image.Image): The PIL image to encode
            
        Returns:
            str: Base64 encoded string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def inv_normalize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Inverse normalize the image tensor.
        
        Args:
            img (torch.Tensor): Normalized image tensor
            
        Returns:
            torch.Tensor: Denormalized image tensor
        """
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

        img = img * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
        img = img.clamp(0, 1)
        return img
    
    def generate_cartoon_avatar(self, image: Image.Image) -> Image.Image:
        """
        Generate a cartoon avatar from the input image.
        
        Args:
            image (Image.Image): Input image
            
        Returns:
            Image.Image: Cartoon avatar image
        """
        # Preprocess and transform the image
        trf = get_no_aug_transform()
        image_tensor = torch.from_numpy(np.array([trf(image).numpy()]))
        image_tensor = image_tensor.to(self.device)
        
        # Generate cartoon image
        with torch.no_grad():
            cartoon_tensor = self.netG(image_tensor)
        
        # Post-process the cartoon image
        cartoon_tensor = self.inv_normalize(cartoon_tensor)
        cartoon_image = TF.to_pil_image(cartoon_tensor[0].cpu())
        
        return cartoon_image
    
    def generate_avatar_from_image(self, 
                                  image_path: str, 
                                  output_path: Optional[str] = None) -> str:
        """
        Generate a cartoon avatar from an image file.
        
        Args:
            image_path (str): Path to the input image
            output_path (str, optional): Path to save the output avatar
            
        Returns:
            str: Path to the saved avatar image
        """
        # Load and preprocess the image
        image = self.load_image(image_path)
        processed_image = self.preprocess_image(image)
        
        # Generate cartoon avatar
        cartoon_avatar = self.generate_cartoon_avatar(processed_image)
        
        # Save the output
        if output_path is None:
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = f"{name}_cartoon{ext}"
        
        cartoon_avatar.save(output_path)
        print(f"Cartoon avatar saved to: {output_path}")
        
        return output_path
    
    def display_result(self, original_path: str, avatar_path: str) -> None:
        """
        Display the original image and the generated avatar side by side.
        
        Args:
            original_path (str): Path to the original image
            avatar_path (str): Path to the generated avatar
        """
        original = Image.open(original_path)
        avatar = Image.open(avatar_path)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original image
        ax1.imshow(np.array(original))
        ax1.set_title("Original Image")
        ax1.axis("off")
        
        # Display generated avatar
        ax2.imshow(np.array(avatar))
        ax2.set_title("Cartoon Avatar")
        ax2.axis("off")
        
        plt.tight_layout()
        plt.show()

def main() -> None:
    """Main function to run the cartoon avatar generator from command line."""
    parser = argparse.ArgumentParser(description="Generate cartoon avatars from images")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", "-o", type=str, help="Path to save output avatar")
    parser.add_argument("--checkpoint", "-c", type=str, default="./checkpoints/trained_netG.pth", 
                        help="Path to model checkpoint")
    parser.add_argument("--device", "-d", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run the model on")
    parser.add_argument("--display", action="store_true", help="Display the results")
    
    args = parser.parse_args()
    
    # Check if CUDA is available if requested
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
        args.device = "cpu"
    
    # Generate avatar
    generator = CartoonAvatarGenerator(checkpoint_path=args.checkpoint, device=args.device)
    output_path = generator.generate_avatar_from_image(args.input, args.output)
    
    # Display results if requested
    if args.display:
        generator.display_result(args.input, output_path)

if __name__ == "__main__":
    main() 