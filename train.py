import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from facenet_pytorch import InceptionResnetV1
from utils.dataset import FacialHairDataset
from utils.model import UNetGenerator, PatchGANDiscriminator
import os
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS        = 50
BATCH_SIZE    = 8
LR            = 2e-4
L1_LAMBDA     = 100      # weight for pixel-level loss
IDENTITY_LAMBDA = 10     # weight for identity preservation loss
IMAGE_SIZE    = 256

TRAIN_BEARD   = "data/train/beard"
TRAIN_CLEAN   = "data/train/clean"
VAL_BEARD     = "data/val/beard"
VAL_CLEAN     = "data/val/clean"
CHECKPOINT_DIR = "models"
OUTPUT_DIR    = "outputs"

print(f"Using device: {DEVICE}")

# ── Datasets & Loaders ────────────────────────────────────────────────────────
train_dataset = FacialHairDataset(TRAIN_BEARD, TRAIN_CLEAN, IMAGE_SIZE)
val_dataset   = FacialHairDataset(VAL_BEARD,   VAL_CLEAN,   IMAGE_SIZE)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                           shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=1,
                           shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

# ── Models ────────────────────────────────────────────────────────────────────
generator     = UNetGenerator().to(DEVICE)
discriminator = PatchGANDiscriminator().to(DEVICE)

# FaceNet — frozen, used only for identity loss (Feature #2)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
for param in facenet.parameters():
    param.requires_grad = False

# ── Optimizers ────────────────────────────────────────────────────────────────
opt_gen  = torch.optim.Adam(generator.parameters(),     lr=LR, betas=(0.5, 0.999))
opt_disc = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# ── Loss Functions ────────────────────────────────────────────────────────────
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1  = nn.L1Loss()
criterion_id  = nn.L1Loss()   # identity loss

# ── Helper: resize for FaceNet (needs 160x160) ────────────────────────────────
resize_for_facenet = torch.nn.functional.interpolate

def get_identity_loss(real_clean, fake_clean):
    """Compare FaceNet embeddings of real vs generated image."""
    real_resized = resize_for_facenet(real_clean, size=(160, 160),
                                      mode='bilinear', align_corners=False)
    fake_resized = resize_for_facenet(fake_clean, size=(160, 160),
                                      mode='bilinear', align_corners=False)
    with torch.no_grad():
        real_embed = facenet(real_resized)
    fake_embed = facenet(fake_resized)
    return criterion_id(fake_embed, real_embed)

# ── Training Loop ─────────────────────────────────────────────────────────────
def train_one_epoch(epoch):
    generator.train()
    discriminator.train()

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for beard_img, clean_img in loop:
        beard_img = beard_img.to(DEVICE)
        clean_img = clean_img.to(DEVICE)

        # ── Train Discriminator ──────────────────────────────────────────────
        fake_clean = generator(beard_img)

        real_pred  = discriminator(beard_img, clean_img)
        fake_pred  = discriminator(beard_img, fake_clean.detach())

        loss_disc_real = criterion_GAN(real_pred,
                                       torch.ones_like(real_pred))
        loss_disc_fake = criterion_GAN(fake_pred,
                                       torch.zeros_like(fake_pred))
        loss_disc = (loss_disc_real + loss_disc_fake) * 0.5

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # ── Train Generator ──────────────────────────────────────────────────
        fake_pred_for_gen = discriminator(beard_img, fake_clean)

        loss_gan      = criterion_GAN(fake_pred_for_gen,
                                      torch.ones_like(fake_pred_for_gen))
        loss_l1       = criterion_L1(fake_clean, clean_img) * L1_LAMBDA
        loss_identity = get_identity_loss(clean_img, fake_clean) * IDENTITY_LAMBDA

        loss_gen = loss_gan + loss_l1 + loss_identity

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        loop.set_postfix(
            D=f"{loss_disc.item():.3f}",
            G=f"{loss_gen.item():.3f}",
            ID=f"{loss_identity.item():.3f}"
        )

# ── Validation: save sample outputs ───────────────────────────────────────────
def validate(epoch):
    generator.eval()
    with torch.no_grad():
        for i, (beard_img, clean_img) in enumerate(val_loader):
            if i >= 5:   # save only 5 samples per epoch
                break
            beard_img = beard_img.to(DEVICE)
            fake_clean = generator(beard_img)

            # save side-by-side: input | generated | ground truth
            comparison = torch.cat([beard_img, fake_clean,
                                    clean_img.to(DEVICE)], dim=3)
            save_image(comparison * 0.5 + 0.5,
                       f"{OUTPUT_DIR}/epoch{epoch+1}_sample{i+1}.png")

# ── Save checkpoint ───────────────────────────────────────────────────────────
def save_checkpoint(epoch):
    torch.save(generator.state_dict(),
               f"{CHECKPOINT_DIR}/generator_epoch{epoch+1}.pth")
    torch.save(discriminator.state_dict(),
               f"{CHECKPOINT_DIR}/discriminator_epoch{epoch+1}.pth")
    print(f"Checkpoint saved at epoch {epoch+1}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR,    exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        train_one_epoch(epoch)
        validate(epoch)

        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(epoch)

    print("Training complete!")