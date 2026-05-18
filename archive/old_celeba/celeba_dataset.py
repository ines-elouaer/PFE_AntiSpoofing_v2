import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])

class CelebASpoofDataset(Dataset):
    """
    Dataset CelebA-Spoof adapté à ton pipeline CNN+LSTM+Behavior.

    Retourne:
        seq   : [T, 3, 224, 224]
        behav : [9]
        label : int
    """

    def __init__(self, csv_path, base_dir, seq_len=16, augment=False):
        self.df = pd.read_csv(csv_path)
        self.base_dir = base_dir
        self.seq_len = seq_len
        self.transform = transform_train if augment else transform_eval

        print(f"[CelebA] CSV chargé: {csv_path}")
        print(f"[CelebA] Nb images: {len(self.df)}")
        print(f"[CelebA] Real: {(self.df['label'] == 0).sum()} | Spoof: {(self.df['label'] == 1).sum()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rel_path = row["image_path"]
        img_path = os.path.join(self.base_dir, rel_path)

        img = Image.open(img_path).convert("RGB")
        frame = self.transform(img)  # [3, 224, 224]

        # image statique répétée T fois
        seq = frame.unsqueeze(0).repeat(self.seq_len, 1, 1, 1)  # [T, 3, 224, 224]

        # pas de signaux comportementaux disponibles sur CelebA image-based
        behav = torch.zeros(9, dtype=torch.float32)

        label = int(row["label"])

        return seq, behav, label
