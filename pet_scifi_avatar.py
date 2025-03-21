#!/usr/bin/env python3
import os
import sys
import base64
import io
import argparse
from typing import Tuple, Dict, Any, List, Optional, Union
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import ollama
import matplotlib.pyplot as plt
import cv2

class PetSciFiAvatarGenerator:
    """A class to generate sci-fi avatars from pet face images."""
    
    def __init__(self, model_name: str = "llama3.2-vision:11b"):
        """
        Initialize the Pet Sci-Fi Avatar Generator.
        
        Args:
            model_name (str): The name of the Ollama model to use
        """
        self.model_name: str = model_name
        self.color_palette: List[Tuple[int, int, int]] = []
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
    
    def crop_pet_face(self, image: Image.Image) -> Image.Image:
        """
        Detect and crop the pet face from the image using OpenCV.
        
        Args:
            image (Image.Image): The input PIL Image
            
        Returns:
            Image.Image: Cropped pet face image
        """
        # Convert PIL image to OpenCV format
        opencv_image: np.ndarray = np.array(image.convert('RGB'))
        opencv_image = opencv_image[:, :, ::-1].copy()  # Convert RGB to BGR
        
        # Load pre-trained models for pet face detection
        face_cascade: cv2.CascadeClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface_extended.xml')
        
        # Convert to grayscale for detection
        gray: np.ndarray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect pet faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If no face detected, try with a more relaxed detection
        if len(faces) == 0:
            # Try with more relaxed parameters
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
            
            if len(faces) == 0:
                print("No pet face detected. Using the entire image with center crop.")
                # Use a centered square crop instead
                width, height = image.size
                size = min(width, height)
                left = (width - size) // 2
                top = (height - size) // 2
                right = left + size
                bottom = top + size
                return image.crop((left, top, right, bottom))
        
        # Get the largest face
        largest_face: Tuple[int, int, int, int] = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = largest_face
        
        # Add some margin (30% for better framing)
        margin_x: int = int(w * 0.3)
        margin_y: int = int(h * 0.3)
        
        # Adjust coordinates with margin
        x_with_margin: int = max(0, x - margin_x)
        y_with_margin: int = max(0, y - margin_y)
        w_with_margin: int = min(opencv_image.shape[1] - x_with_margin, w + 2 * margin_x)
        h_with_margin: int = min(opencv_image.shape[0] - y_with_margin, h + 2 * margin_y)
        
        # Crop the image
        cropped_opencv: np.ndarray = opencv_image[y_with_margin:y_with_margin + h_with_margin, 
                                                  x_with_margin:x_with_margin + w_with_margin]
        
        # Convert back to PIL
        cropped_image: Image.Image = Image.fromarray(cv2.cvtColor(cropped_opencv, cv2.COLOR_BGR2RGB))
        
        print(f"Pet face detected and cropped successfully!")
        return cropped_image
    
    def preprocess_image(self, image: Image.Image, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Preprocess the image by resizing, enhancing, and converting to RGB.
        
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
        
        # Create a new image with the target size and black background (for sci-fi feel)
        new_image: Image.Image = Image.new('RGB', size, (0, 0, 0))
        
        # Paste the resized image onto the center of the new image
        offset: Tuple[int, int] = ((size[0] - image.width) // 2, (size[1] - image.height) // 2)
        new_image.paste(image, offset)
        
        # Enhance contrast
        enhancer: ImageEnhance.Contrast = ImageEnhance.Contrast(new_image)
        enhanced_image: Image.Image = enhancer.enhance(1.4)
        
        # Enhance color with sci-fi blue/purple tint
        enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = enhancer.enhance(1.5)
        
        # Apply slight sharpening
        enhanced_image = enhanced_image.filter(ImageFilter.SHARPEN)
        
        return enhanced_image
    
    def encode_image_to_base64(self, image: Image.Image) -> str:
        """
        Encode a PIL Image to base64 string.
        
        Args:
            image (Image.Image): The PIL Image to encode
            
        Returns:
            str: Base64 encoded image string
        """
        buffered: io.BytesIO = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def add_scifi_elements(self, image: Image.Image) -> Image.Image:
        """
        Add sci-fi elements to the image.
        
        Args:
            image (Image.Image): The input image
            
        Returns:
            Image.Image: Image with sci-fi elements
        """
        # Create a copy of the image
        scifi_image: Image.Image = image.copy()
        width, height = scifi_image.size
        
        # Convert original image to RGBA
        if scifi_image.mode != 'RGBA':
            scifi_image = scifi_image.convert('RGBA')
        
        # Create sci-fi overlay layers
        glow_layer: Image.Image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        tech_layer: Image.Image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        circuit_layer: Image.Image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # Apply a futuristic glow effect around the edges
        for x in range(width):
            for y in range(height):
                # Calculate distance from edge
                edge_dist: int = min(x, y, width-x-1, height-y-1)
                if edge_dist < 30:
                    intensity: int = int(255 * (1 - edge_dist/30))
                    # Cyan/blue glow with randomized color
                    r = np.random.randint(0, 40)
                    g = np.random.randint(100, 200)
                    b = np.random.randint(180, 255)
                    glow_layer.putpixel((x, y), (r, g, b, intensity))
        
        # Generate tech pattern in the background
        for _ in range(50):  # More tech lines
            start_x: int = np.random.randint(0, width)
            start_y: int = np.random.randint(0, height)
            length: int = np.random.randint(30, 150)  # Longer lines
            direction: float = np.random.random() * 2 * np.pi
            thickness: int = np.random.randint(1, 3)  # Line thickness
            
            # Vary tech line color
            r = np.random.randint(0, 50)
            g = np.random.randint(150, 255)
            b = np.random.randint(150, 255)
            alpha = np.random.randint(120, 200)
            
            # Draw the tech line with thickness
            for i in range(length):
                x: int = int(start_x + i * np.cos(direction))
                y: int = int(start_y + i * np.sin(direction))
                
                if 0 <= x < width and 0 <= y < height:
                    # Draw line with thickness
                    for dx in range(-thickness, thickness+1):
                        for dy in range(-thickness, thickness+1):
                            if 0 <= x+dx < width and 0 <= y+dy < height:
                                # Fade out toward the end
                                fade = 1 - (i / length)
                                current_alpha = int(alpha * fade)
                                tech_layer.putpixel((x+dx, y+dy), (r, g, b, current_alpha))
        
        # Add circuit patterns
        for _ in range(10):
            # Create circuit nodes
            node_x: int = np.random.randint(0, width)
            node_y: int = np.random.randint(0, height)
            node_radius: int = np.random.randint(3, 8)
            
            # Draw circuit node
            for x in range(node_x - node_radius, node_x + node_radius + 1):
                for y in range(node_y - node_radius, node_y + node_radius + 1):
                    if 0 <= x < width and 0 <= y < height:
                        dist = np.sqrt((x - node_x)**2 + (y - node_y)**2)
                        if dist <= node_radius:
                            circuit_layer.putpixel((x, y), (0, 255, 255, 180))
            
            # Create circuit branches
            for _ in range(np.random.randint(2, 5)):
                branch_length: int = np.random.randint(20, 80)
                branch_angle: float = np.random.random() * 2 * np.pi
                
                for i in range(branch_length):
                    x: int = int(node_x + i * np.cos(branch_angle))
                    y: int = int(node_y + i * np.sin(branch_angle))
                    
                    if 0 <= x < width and 0 <= y < height:
                        # Vary color along branch
                        g = int(255 * (1 - i/branch_length)) + 100
                        b = int(255 * (1 - i/branch_length)) + 100
                        circuit_layer.putpixel((x, y), (0, g, b, 150))
        
        # Composite the images
        scifi_image = Image.alpha_composite(scifi_image, glow_layer)
        scifi_image = Image.alpha_composite(scifi_image, tech_layer)
        scifi_image = Image.alpha_composite(scifi_image, circuit_layer)
        
        # Apply a slight color enhancement for more sci-fi feel
        scifi_image = scifi_image.convert('RGB')
        enhancer: ImageEnhance.Color = ImageEnhance.Color(scifi_image)
        scifi_image = enhancer.enhance(1.4)
        
        # Apply slight vignette effect
        vignette = Image.new('RGB', scifi_image.size, (0, 0, 0))
        vignette_radius = min(width, height) // 2
        vignette_center = (width // 2, height // 2)
        
        for x in range(width):
            for y in range(height):
                dist = np.sqrt((x - vignette_center[0])**2 + (y - vignette_center[1])**2)
                factor = max(0, min(1, (vignette_radius - dist) / vignette_radius))
                
                # Get the pixel from the sci-fi image
                r, g, b = scifi_image.getpixel((x, y))
                
                # Apply vignette factor
                r = int(r * factor)
                g = int(g * factor)
                b = int(b * factor)
                
                vignette.putpixel((x, y), (r, g, b))
        
        return vignette
    
    def generate_scifi_avatar_prompt(self, image: Image.Image) -> Dict[str, Any]:
        """
        Generate a prompt for the AI model to create a sci-fi avatar.
        
        Args:
            image (Image.Image): The input image
            
        Returns:
            Dict[str, Any]: Response from the AI model
        """
        base64_image: str = self.encode_image_to_base64(image)
        
        prompt: str = """
        I need you to transform this pet face into a sci-fi character avatar. 
        Create a detailed, futuristic, cyberpunk-style portrait with the following elements:
        
        1. Maintain the distinctive features and expression of the pet
        2. Add futuristic tech elements like holographic displays, cybernetic enhancements, or neon glows
        3. Use a color palette with bright blues, purples, and teals on a dark background
        4. Make it look like a character from a sci-fi movie or game
        
        Please describe in detail how this pet would look as a sci-fi character avatar.
        """
        
        try:
            # Call the Ollama model with the image and prompt
            response: Dict[str, Any] = ollama.chat(
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
            print(f"Error generating AI response: {e}")
            sys.exit(1)
    
    def generate_avatar_from_pet(self, image_path: str, output_path: Optional[str] = None) -> str:
        """
        Generate a sci-fi avatar from a pet image.
        
        Args:
            image_path (str): Path to the pet image
            output_path (str, optional): Path to save the output image
            
        Returns:
            str: Path to the saved avatar image
        """
        # Load the image
        image: Image.Image = self.load_image(image_path)
        
        # Crop the pet face
        print("Detecting and cropping pet face...")
        pet_face: Image.Image = self.crop_pet_face(image)
        
        # Preprocess the image
        preprocessed_image: Image.Image = self.preprocess_image(pet_face)
        
        print("Analyzing your pet and generating sci-fi concept...")
        
        # Get AI suggestions for the sci-fi avatar
        ai_response: Dict[str, Any] = self.generate_scifi_avatar_prompt(preprocessed_image)
        
        print("Generating sci-fi avatar...")
        
        # Apply sci-fi elements
        scifi_avatar: Image.Image = self.add_scifi_elements(preprocessed_image)
        
        # Save the result
        if output_path is None:
            base_name: str = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_scifi_avatar.png"
        
        scifi_avatar.save(output_path)
        
        # Print AI's description
        print("\nAI's Sci-Fi Avatar Concept:")
        print(ai_response['message']['content'])
        
        print(f"\nSci-fi avatar saved to: {output_path}")
        return output_path
    
    def display_result(self, original_path: str, avatar_path: str) -> None:
        """
        Display the original pet image alongside the generated avatar.
        
        Args:
            original_path (str): Path to the original pet image
            avatar_path (str): Path to the generated avatar
        """
        original: Image.Image = Image.open(original_path)
        avatar: Image.Image = Image.open(avatar_path)
        
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original)
        plt.title("Original Pet Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(avatar)
        plt.title("Generated Sci-Fi Avatar")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def main() -> None:
    """Main function to run the pet sci-fi avatar generator from command line."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate sci-fi avatars from pet face images")
    parser.add_argument("image_path", type=str, help="Path to the pet image")
    parser.add_argument("--output", "-o", type=str, help="Output path for the avatar image")
    parser.add_argument("--model", "-m", type=str, default="llama3.2-vision:11b", 
                        help="Ollama model to use")
    parser.add_argument("--display", "-d", action="store_true", 
                        help="Display result after generation")
    
    args: argparse.Namespace = parser.parse_args()
    
    generator: PetSciFiAvatarGenerator = PetSciFiAvatarGenerator(model_name=args.model)
    avatar_path: str = generator.generate_avatar_from_pet(
        args.image_path, 
        args.output
    )
    
    if args.display:
        generator.display_result(args.image_path, avatar_path)


if __name__ == "__main__":
    main() 