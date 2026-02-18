from torch.utils.data import DataLoader
from src.deep_learning.datasets import CASIASequenceDataset

def main():
    ds = CASIASequenceDataset(
        csv_path=r"data\processed\casia\splits\train.csv",
        T=16,
        img_size=224
    )
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    x, y, vid = next(iter(dl))
    print("x shape:", x.shape)  # [B, T, C, H, W]
    print("y:", y)
    print("vid:", vid)

if __name__ == "__main__":
    main()
