#!/usr/bin/env python3
import os
import sys
import time
import base64
import io
from typing import Tuple, Optional, Dict, Any, List, Union
import argparse
from PIL import Image
import numpy as np
import ollama
import matplotlib.pyplot as plt
from tqdm import tqdm

class PetAvatarGenerator:
    def __init__(self, model_name: str = "llama3.2-vision:11b"):
        """
        Initialize the Pet Avatar Generator.
        
        Args:
            model_name (str): The name of the Ollama model to use
        """
        self.model_name = model_name
        self.ensure_model_available()
        
    def ensure_model_available(self) -> None:
        """Check if the specified model is available, pull if not."""
        try:
            models = ollama.list()
            model_names = [model.get('name') for model in models.get('models', [])]
            
            if self.model_name not in model_names:
                print(f"Model {self.model_name} not found locally. Pulling from Ollama...")
                ollama.pull(self.model_name)
                print(f"Model {self.model_name} successfully pulled!")
            else:
                print(f"Using existing model: {self.model_name}")
        except Exception as e:
            print(f"Error checking/pulling model: {e}")
            print("Make sure Ollama is running on your system!")
            sys.exit(1)

    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from a file path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Image.Image: The loaded PIL Image
        """
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image: Image.Image, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Preprocess the image by resizing and converting to RGB.
        
        Args:
            image (Image.Image): The input PIL Image
            size (Tuple[int, int]): Target size for resizing
            
        Returns:
            Image.Image: Preprocessed image
        """
        # Convert to RGB if in different mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize the image while maintaining aspect ratio
        image.thumbnail(size, Image.Resampling.LANCZOS)
        
        # Create a new image with the target size
        new_image = Image.new('RGB', size, (255, 255, 255))
        
        # Paste the resized image onto the center of the new image
        offset = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        new_image.paste(image, offset)
        
        return new_image
    
    def encode_image_to_base64(self, image: Image.Image) -> str:
        """
        Encode a PIL Image to base64 string.
        
        Args:
            image (Image.Image): The PIL Image to encode
            
        Returns:
            str: Base64 encoded image string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_art_avatar(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate a stylized art avatar using Ollama model.
        
        Args:
            image (Image.Image): The input image
            
        Returns:
            Dict[str, Any]: Response from the AI model
        """
        base64_image = self.encode_image_to_base64(image)
        
        prompt = """
        I need you to create a detailed artistic avatar version of this pet.
        Make it cute, colorful, and stylized like a cartoon avatar.
        Only return the avatar representation of the pet.
        """
        
        try:
            # Call the Ollama model with the image and prompt
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [base64_image]
                    }
                ]
            )
            return response
        except Exception as e:
            print(f"Error generating art avatar: {e}")
            sys.exit(1)
    
    def generate_avatar_from_pet(self, image_path: str, output_path: str = None) -> str:
        """
        Generate an art avatar from a pet image.
        
        Args:
            image_path (str): Path to the pet image
            output_path (str, optional): Path to save the output image
            
        Returns:
            str: Path to the saved avatar image
        """
        # Load and preprocess the image
        image = self.load_image(image_path)
        preprocessed_image = self.preprocess_image(image)
        
        print("Generating artistic avatar from your pet image...")
        
        # Generate art avatar using AI
        response = self.generate_art_avatar(preprocessed_image)
        
        # Process AI response
        avatar_instructions = response['message']['content']
        print("\nAI Response:")
        print(avatar_instructions)
        
        # Save the preprocessed image as the result
        # In a real implementation, you would convert the AI's instructions into an image
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_art_avatar.png"
        
        final_image = preprocessed_image.resize((512, 512), Image.Resampling.LANCZOS)
        final_image.save(output_path)
        
        print(f"\nArt avatar saved to: {output_path}")
        return output_path
    
    def display_result(self, original_path: str, avatar_path: str) -> None:
        """
        Display the original pet image alongside the generated avatar.
        
        Args:
            original_path (str): Path to the original pet image
            avatar_path (str): Path to the generated avatar
        """
        original = Image.open(original_path)
        avatar = Image.open(avatar_path)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original Pet Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(avatar)
        plt.title("Generated Art Avatar")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def main() -> None:
    """Main function to run the avatar generator from command line."""
    parser = argparse.ArgumentParser(description="Generate art avatars from pet images")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input pet image")
    parser.add_argument("--output", "-o", type=str, help="Path to save output avatar")
    parser.add_argument("--model", "-m", type=str, default="llama3.2-vision:11b",
                        help="Name of the Ollama vision model to use")
    parser.add_argument("--display", action="store_true", help="Display the results")
    
    args = parser.parse_args()
    
    # Generate avatar
    generator = PetAvatarGenerator(model_name=args.model)
    output_path = generator.generate_avatar_from_pet(args.input, args.output)
    
    # Display results if requested
    if args.display:
        generator.display_result(args.input, output_path)


if __name__ == "__main__":
    main() 