import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from torch.utils.data import DataLoader
from celeba_dataset import CelebASpoofDataset

BASE_DIR = r"E:\CelebA-Spoof\CelebA_Spoof\Data"
CSV_TRAIN = r"E:\PFE_AntiSpoofing_v2\data\celeba\celeba_8000_ft_train.csv"
ds = CelebASpoofDataset(
    csv_path=CSV_TRAIN,
    base_dir=BASE_DIR,
    seq_len=16,
    augment=False
)

dl = DataLoader(
    ds,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

seq, behav, label = next(iter(dl))

print("seq shape  :", seq.shape)
print("behav shape:", behav.shape)
print("label shape:", label.shape)
print("labels     :", label)
print("OK — DataLoader fonctionnel ✔")
