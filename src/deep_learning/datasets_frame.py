import csv
import random
from pathlib import Path
from PIL import Image, ImageFilter
import io

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class RandomJPEGCompression:
    def __init__(self, quality_min=35, quality_max=95, p=0.3):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        q = random.randint(self.quality_min, self.quality_max)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlur:
    def __init__(self, radius_min=0.1, radius_max=1.2, p=0.2):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        r = random.uniform(self.radius_min, self.radius_max)
        return img.filter(ImageFilter.GaussianBlur(radius=r))


class AddGaussianNoiseTensor:
    def __init__(self, sigma_min=0.0, sigma_max=0.03, p=0.2):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def __call__(self, x: torch.Tensor):
        if random.random() > self.p:
            return x
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        return torch.clamp(x + torch.randn_like(x) * sigma, 0.0, 1.0)


def build_transforms(img_size: int, aug_mode: str):
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if aug_mode == "none":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    if aug_mode == "soft":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.01),
            RandomGaussianBlur(p=0.10, radius_min=0.1, radius_max=0.8),
            RandomJPEGCompression(p=0.20, quality_min=50, quality_max=95),
            transforms.ToTensor(),
            AddGaussianNoiseTensor(p=0.10, sigma_max=0.02),
            transforms.Normalize(mean=mean, std=std),
        ])

    if aug_mode == "strong":
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.25, hue=0.02),
            RandomGaussianBlur(p=0.25, radius_min=0.1, radius_max=1.2),
            RandomJPEGCompression(p=0.40, quality_min=35, quality_max=95),
            transforms.ToTensor(),
            AddGaussianNoiseTensor(p=0.25, sigma_max=0.03),
            transforms.Normalize(mean=mean, std=std),
        ])

    raise ValueError(f"aug_mode invalide: {aug_mode}")


class CASIAFrameDataset(Dataset):
    def __init__(self, csv_path: str, img_size: int = 224, aug_mode: str = "none"):
        self.rows = []
        with Path(csv_path).open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                row["label"] = int(row["label"])
                self.rows.append(row)

        self.tf = build_transforms(img_size, aug_mode)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(row["label"], dtype=torch.long)
        video_id = row.get("video_id", "")
        return x, y, video_id
