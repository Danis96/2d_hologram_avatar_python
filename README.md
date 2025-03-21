# Pet Avatar Generator Collection

A collection of Python tools to transform pet images into stylized avatars using various techniques.

## Overview

This repository contains three distinct tools for generating artistic avatars from pet images:

1. **Pet Avatar Generator** - Creates artistic avatars using the Ollama/Llama3 vision model
2. **Pet Sci-Fi Avatar Generator** - Transforms pets into cyberpunk-style sci-fi characters
3. **Cartoon Avatar Generator** - Converts images to cartoon style using a trained neural network

## Requirements

- Python 3.8+
- Ollama with Llama 3.2 Vision models (for Pet Avatar and Sci-Fi generators)
- PyTorch (for Cartoon Avatar generator)
- Other dependencies listed in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pet-avatar-generator.git
cd pet-avatar-generator

# Install dependencies
pip install -r requirements.txt

# Install Ollama (if not already installed)
# Follow instructions at https://ollama.ai/

# Pull required models
ollama pull llama3.2-vision:11b
```

## Usage

### Pet Avatar Generator

Creates artistic avatars using Ollama's vision model:

```bash
python pet_avatar_generator.py --input path/to/pet_image.jpg --output avatar.png --display
```

Options:
- `--input`, `-i`: Path to the input pet image (required)
- `--output`, `-o`: Path to save the output avatar (optional)
- `--model`, `-m`: Name of the Ollama vision model to use (default: llama3.2-vision:11b)
- `--display`: Display the results with matplotlib (optional)

### Pet Sci-Fi Avatar Generator

Transforms pets into cyberpunk-style sci-fi characters:

```bash
python pet_scifi_avatar.py path/to/pet_image.jpg --output scifi_avatar.png --display
```

Options:
- First argument: Path to the pet image (required)
- `--output`, `-o`: Output path for the avatar image (optional)
- `--model`, `-m`: Ollama model to use (default: llama3.2-vision:11b)
- `--display`, `-d`: Display result after generation (optional)

### Cartoon Avatar Generator

Converts images to cartoon style using a trained neural network:

```bash
python cartoon_avatar_generator.py --input path/to/image.jpg --output cartoon_avatar.png --display
```

Options:
- `--input`, `-i`: Path to input image (required)
- `--output`, `-o`: Path to save output avatar (optional)
- `--checkpoint`, `-c`: Path to model checkpoint (default: ./checkpoints/trained_netG.pth)
- `--device`, `-d`: Device to run the model on ("cpu" or "cuda", default: "cpu")
- `--display`: Display the results (optional)

## Example

```bash
# Generate a standard artistic avatar
python pet_avatar_generator.py --input my_cat.jpg --display

# Generate a sci-fi themed avatar
python pet_scifi_avatar.py my_cat.jpg --display

# Generate a cartoon style avatar
python cartoon_avatar_generator.py --input my_cat.jpg --display
```

## Model Information

- **Pet Avatar Generator**: Uses Ollama's vision model to generate artistic descriptions of pets
- **Pet Sci-Fi Avatar Generator**: Uses vision models and custom image processing to create sci-fi themed pet avatars
- **Cartoon Avatar Generator**: Uses a trained Generator model to transform images into cartoon style

## Note

For the cartoon avatar generator, you'll need to download the pre-trained model weights and place them in the `checkpoints` directory.

## License

[MIT License](LICENSE)