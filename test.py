import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from utils.model import UNetGenerator
import os

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE  = 256
MODEL_PATH  = "models/generator_epoch10.pth"   # change epoch number as needed
OUTPUT_DIR  = "outputs/test_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
generator = UNetGenerator().to(DEVICE)

if not os.path.exists(MODEL_PATH):
    print(f"No checkpoint found at {MODEL_PATH}")
    print("Available checkpoints:")
    for f in os.listdir("models"):
        print(f"  models/{f}")
    exit()

generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.eval()
print(f"Model loaded from {MODEL_PATH}")

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# ── Run on single image ───────────────────────────────────────────────────────
def remove_hair(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = generator(input_tensor)

    # save side by side: input | output
    comparison = torch.cat([input_tensor, output], dim=3)
    filename   = os.path.basename(image_path)
    save_path  = os.path.join(OUTPUT_DIR, f"result_{filename}")
    save_image(comparison * 0.5 + 0.5, save_path)
    print(f"Result saved → {save_path}")

# ── Run on all images in a folder ─────────────────────────────────────────────
def remove_hair_folder(folder_path):
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Processing {len(images)} images...")
    for img_file in images:
        remove_hair(os.path.join(folder_path, img_file))
    print("Done!")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test on validation beard images
    remove_hair_folder("data/val/beard")