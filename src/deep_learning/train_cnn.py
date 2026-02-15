import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from tqdm import tqdm

from src.deep_learning.datasets_frame import CASIAFrameDataset


def build_model(num_classes=2):
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, total_loss = 0, 0, 0.0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():

    train_csv = r"data\processed\CASIA\splits_subject\train.csv"
    val_csv   = r"data\processed\CASIA\splits_subject\val.csv"

    aug_mode = "strong"
    out_dir = fr"experiments\exp3_casia_mnv3large_{aug_mode}"

    img_size = 224
    batch_size = 16
    epochs = 10
    lr = 2e-4
    weight_decay = 1e-4
    num_workers = 0

    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = CASIAFrameDataset(train_csv, img_size=img_size, aug_mode=aug_mode)
    val_ds = CASIAFrameDataset(val_csv, img_size=img_size, aug_mode="none")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = build_model().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ce = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = os.path.join(out_dir, "best_model.pth")

    history_csv = os.path.join(out_dir, "history.csv")
    with open(history_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    for epoch in range(1, epochs + 1):
        model.train()

        total, correct, running_loss = 0, 0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for x, y, _ in pbar:
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

            pbar.set_postfix(
                loss=running_loss/max(total,1),
                acc=correct/max(total,1),
                lr=opt.param_groups[0]["lr"]
            )

        scheduler.step()

        train_loss = running_loss/max(total,1)
        train_acc = correct/max(total,1)

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        with open(history_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{opt.param_groups[0]['lr']:.8f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict()}, best_path)
            print("Saved best model")

    print("Done. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
