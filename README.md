# 2D Hologram Avatar Generator

A collection of tools for generating different styles of avatars from images:

1. **Cartoon Avatar Generator** - Transform photos into cartoon style images using CartoonGAN
2. **Pet Avatar Generator** - Generate art-style avatars from pet images using AI
3. **Enhanced Pixel Avatar** - Create pixel art style avatars with AI enhancement
4. **Sci-Fi Avatar** - Transform photos into sci-fi style images

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model Weights

For the CartoonGAN model:

```bash
python download_weights.py
```

## Usage

### Cartoon Avatar Generator

```bash
python cartoon_avatar_generator.py --input your_image.jpg --output cartoon_avatar.jpg --display
```

Arguments:
- `--input` or `-i`: Path to input image (required)
- `--output` or `-o`: Path to save output avatar (optional)
- `--checkpoint` or `-c`: Path to model checkpoint (default: ./checkpoints/trained_netG.pth)
- `--device` or `-d`: Device to run the model on ('cpu' or 'cuda', default: 'cpu')
- `--display`: Display the results (optional)

### Pet Avatar Generator

```bash
python pet_avatar_generator.py --input your_pet.jpg --output pet_avatar.jpg --display
```

### Enhanced Pixel Avatar

```bash
python enhanced_pixel_avatar.py --input your_image.jpg --output pixel_avatar.jpg --resolution 32 --display
```

### Sci-Fi Avatar

```bash
python pet_scifi_avatar.py --input your_image.jpg --output scifi_avatar.jpg --display
```

## Acknowledgements

- CartoonGAN implementation based on [FilipAndersson245/cartoon-gan](https://github.com/FilipAndersson245/cartoon-gan) 