import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(filename, 'wb') as f, tqdm(
        desc=filename, total=total,
        unit='B', unit_scale=True
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def setup_dataset():
    """
    Downloads a small sample beard/clean face dataset from public sources.
    For full training, you should use CelebA dataset.
    """

    os.makedirs("data/train/beard", exist_ok=True)
    os.makedirs("data/train/clean", exist_ok=True)
    os.makedirs("data/val/beard",   exist_ok=True)
    os.makedirs("data/val/clean",   exist_ok=True)

    # Download sample paired face images
    print("Downloading sample dataset...")

    # Using publicly available face images for testing
    sample_urls = {
        "beard": [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg",
        ]
    }

    print("\nNOTE: For real training, follow these steps to get CelebA dataset:")
    print("=" * 60)
    print("1. Go to: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("2. Download 'img_align_celeba.zip'")
    print("3. Also download 'list_attr_celeba.txt'")
    print("4. Run the script below to auto-sort beard vs clean images")
    print("=" * 60)
    print("\nFor now, creating a DEMO dataset with synthetic images...")
    create_demo_dataset()

def create_demo_dataset():
    """Creates a small demo dataset using random noise images for testing."""
    import torch
    import torchvision.transforms as T
    from PIL import Image, ImageDraw
    import random

    print("Creating demo paired images...")

    def create_face_image(has_beard=True, size=256):
        """Creates a simple synthetic face image."""
        img = Image.new('RGB', (size, size), color=(255, 220, 185))
        draw = ImageDraw.Draw(img)

        cx, cy = size // 2, size // 2

        # Face oval
        draw.ellipse([cx-80, cy-100, cx+80, cy+100],
                     fill=(255, 210, 170), outline=(200, 160, 120), width=2)

        # Eyes
        draw.ellipse([cx-40, cy-30, cx-20, cy-15],
                     fill=(50, 30, 20))
        draw.ellipse([cx+20, cy-30, cx+40, cy-15],
                     fill=(50, 30, 20))

        # Nose
        draw.ellipse([cx-8, cy, cx+8, cy+15],
                     fill=(220, 170, 140))

        # Mouth
        draw.arc([cx-25, cy+25, cx+25, cy+50],
                 start=0, end=180,
                 fill=(180, 80, 80), width=3)

        # Beard (only if has_beard=True)
        if has_beard:
            draw.ellipse([cx-60, cy+40, cx+60, cy+110],
                         fill=(80, 50, 30))
            # Mustache
            draw.ellipse([cx-30, cy+20, cx+30, cy+45],
                         fill=(70, 40, 20))

        # Add slight noise for realism
        import numpy as np
        img_array = np.array(img).astype(float)
        noise = np.random.normal(0, 8, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    # Generate 50 training pairs and 10 validation pairs
    print("Generating 50 training pairs...")
    for i in tqdm(range(50)):
        seed = i * 42
        random.seed(seed)

        beard_img = create_face_image(has_beard=True)
        clean_img = create_face_image(has_beard=False)

        beard_img.save(f"data/train/beard/face_{i:04d}.png")
        clean_img.save(f"data/train/clean/face_{i:04d}.png")

    print("Generating 10 validation pairs...")
    for i in tqdm(range(10)):
        seed = (i + 1000) * 42
        random.seed(seed)

        beard_img = create_face_image(has_beard=True)
        clean_img = create_face_image(has_beard=False)

        beard_img.save(f"data/val/beard/face_{i:04d}.png")
        clean_img.save(f"data/val/clean/face_{i:04d}.png")

    print("\nDemo dataset created!")
    print("data/train/beard → 50 images")
    print("data/train/clean → 50 images")
    print("data/val/beard   → 10 images")
    print("data/val/clean   → 10 images")
    print("\nFor your final project use real CelebA images!")

if __name__ == "__main__":
    setup_dataset()
