import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.model import UNetGenerator
import os

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
OUTPUT_DIR = "outputs/gradcam"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
def load_model(model_path):
    model = UNetGenerator().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ── GradCAM wrapper ───────────────────────────────────────────────────────────
class GeneratorOutputTarget:
    """Target for GradCAM — we want to visualize where
       the generator is focusing to make changes."""
    def __call__(self, model_output):
        # Focus on pixels that changed the most
        return model_output.abs().mean()

def generate_gradcam(image_path, model_path):
    model     = load_model(model_path)
    pil_img   = Image.open(image_path).convert("RGB")
    input_t   = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Target the last encoder layer — shows what regions model focuses on
    target_layer = [model.enc4[-1]]

    cam = GradCAM(model=model, target_layers=target_layer)

    targets   = [GeneratorOutputTarget()]
    grayscale = cam(input_tensor=input_t, targets=targets)

    # Prepare original image for overlay
    img_np = np.array(pil_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    img_np = img_np.astype(np.float32) / 255.0

    # Create heatmap overlay
    visualization = show_cam_on_image(img_np, grayscale[0], use_rgb=True)

    # Save result
    filename  = os.path.basename(image_path)
    save_path = os.path.join(OUTPUT_DIR, f"gradcam_{filename}")
    cv2.imwrite(save_path,
                cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"GradCAM saved → {save_path}")
    return save_path

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    model_path = "models/generator_epoch10.pth"

    # Check for available checkpoints
    if not os.path.exists(model_path):
        checkpoints = [f for f in os.listdir("models") if f.endswith(".pth")
                       and "generator" in f]
        if not checkpoints:
            print("No checkpoints yet! Let training reach epoch 10 first.")
            exit()
        model_path = f"models/{sorted(checkpoints)[-1]}"
        print(f"Using latest checkpoint: {model_path}")

    # Run on all val beard images
    images = [f for f in os.listdir("data/val/beard")
              if f.endswith(('.png', '.jpg', '.jpeg'))]
    for img in images[:5]:   # first 5 only
        generate_gradcam(f"data/val/beard/{img}", model_path)

    print("\nGradCAM visualizations saved in outputs/gradcam/")
    print("These show WHERE the model focuses to remove beard!")