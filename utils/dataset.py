import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FacialHairDataset(Dataset):
    def __init__(self, beard_dir, clean_dir, image_size=256):
        self.beard_dir = beard_dir
        self.clean_dir = clean_dir
        self.image_size = image_size

        self.beard_images = sorted(os.listdir(beard_dir))
        self.clean_images = sorted(os.listdir(clean_dir))

        assert len(self.beard_images) == len(self.clean_images), \
            "Beard and clean folders must have the same number of images!"

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.beard_images)

    def __getitem__(self, idx):
        beard_path = os.path.join(self.beard_dir, self.beard_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])

        beard_img = Image.open(beard_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")

        return self.transform(beard_img), self.transform(clean_img)