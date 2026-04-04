import os
import subprocess
import sys

def install_gdown():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    install_gdown()
    import gdown

import zipfile
import shutil
from tqdm import tqdm
from PIL import Image

# ── Download CelebA ───────────────────────────────────────────────────────────
def download_celeba():
    os.makedirs("celeba_raw", exist_ok=True)

    print("Downloading CelebA images (this may take a while ~1.3GB)...")
    # CelebA aligned images
    url = "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    output = "celeba_raw/img_align_celeba.zip"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    else:
        print("Zip already downloaded, skipping...")

    print("Extracting images...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("celeba_raw/")
    print("Extraction done!")

# ── Download attribute file ───────────────────────────────────────────────────
def download_attributes():
    print("Downloading attribute labels...")
    url = "https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U"
    output = "celeba_raw/list_attr_celeba.txt"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    else:
        print("Attributes already downloaded, skipping...")

# ── Sort into beard / clean folders ──────────────────────────────────────────
def sort_images():
    print("\nReading attribute file...")
    attr_file = "celeba_raw/list_attr_celeba.txt"
    img_dir   = "celeba_raw/img_align_celeba"

    with open(attr_file, 'r') as f:
        lines = f.readlines()

    # Line 0 = count, Line 1 = attribute names, Line 2+ = image data
    attr_names = lines[1].split()
    beard_idx  = attr_names.index("No_Beard")   # 0=has beard, -1=no beard... inverted!
    mustache_idx = attr_names.index("Mustache")

    beard_images = []
    clean_images = []

    print("Sorting images by beard/clean...")
    for line in tqdm(lines[2:]):
        parts    = line.split()
        filename = parts[0]
        attrs    = list(map(int, parts[1:]))

        no_beard  = attrs[beard_idx]      # -1 means HAS beard
        mustache  = attrs[mustache_idx]   #  1 means has mustache

        if no_beard == -1 or mustache == 1:
            beard_images.append(filename)
        else:
            clean_images.append(filename)

    print(f"Found {len(beard_images)} bearded images")
    print(f"Found {len(clean_images)} clean images")

    # Use equal numbers for balanced dataset
    # For quick training use 1000, for better results use 5000+
    NUM_TRAIN = 1000
    NUM_VAL   = 100

    # Clear old data
    for folder in ["data/train/beard", "data/train/clean",
                   "data/val/beard",   "data/val/clean"]:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
        print(f"Cleared {folder}")

    # Copy beard images
    print(f"\nCopying {NUM_TRAIN} training beard images...")
    for fname in tqdm(beard_images[:NUM_TRAIN]):
        src = os.path.join(img_dir, fname)
        dst = os.path.join("data/train/beard", fname)
        if os.path.exists(src):
            img = Image.open(src).convert("RGB")
            # Crop to face area (CelebA faces are centered)
            w, h = img.size
            img = img.crop([w//8, h//8, 7*w//8, 7*h//8])
            img.save(dst)

    print(f"Copying {NUM_VAL} validation beard images...")
    for fname in tqdm(beard_images[NUM_TRAIN:NUM_TRAIN+NUM_VAL]):
        src = os.path.join(img_dir, fname)
        dst = os.path.join("data/val/beard", fname)
        if os.path.exists(src):
            img = Image.open(src).convert("RGB")
            w, h = img.size
            img = img.crop([w//8, h//8, 7*w//8, 7*h//8])
            img.save(dst)

    # Copy clean images
    print(f"\nCopying {NUM_TRAIN} training clean images...")
    for fname in tqdm(clean_images[:NUM_TRAIN]):
        src = os.path.join(img_dir, fname)
        dst = os.path.join("data/train/clean", fname)
        if os.path.exists(src):
            img = Image.open(src).convert("RGB")
            w, h = img.size
            img = img.crop([w//8, h//8, 7*w//8, 7*h//8])
            img.save(dst)

    print(f"Copying {NUM_VAL} validation clean images...")
    for fname in tqdm(clean_images[NUM_TRAIN:NUM_TRAIN+NUM_VAL]):
        src = os.path.join(img_dir, fname)
        dst = os.path.join("data/val/clean", fname)
        if os.path.exists(src):
            img = Image.open(src).convert("RGB")
            w, h = img.size
            img = img.crop([w//8, h//8, 7*w//8, 7*h//8])
            img.save(dst)

    print("\nDataset ready!")
    print(f"data/train/beard → {len(os.listdir('data/train/beard'))} images")
    print(f"data/train/clean → {len(os.listdir('data/train/clean'))} images")
    print(f"data/val/beard   → {len(os.listdir('data/val/beard'))} images")
    print(f"data/val/clean   → {len(os.listdir('data/val/clean'))} images")

if __name__ == "__main__":
    download_celeba()
    download_attributes()
    sort_images()
    print("\nAll done! Now run: python train.py")