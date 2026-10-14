from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


FEATURE_COLS = [
    "brightness",
    "blur",
    "motion_mean",
    "motion_max",
    "frame_valid",
]


def compute_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    real_count = tn + fp
    spoof_count = fn + tp

    apcer = fn / spoof_count if spoof_count else 0.0
    bpcer = fp / real_count if real_count else 0.0
    acer = (apcer + bpcer) / 2.0

    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")

    return {
        "total": int(len(y_true)),
        "real_count": int(real_count),
        "spoof_count": int(spoof_count),
        "tn_real": int(tn),
        "fp_real_as_spoof": int(fp),
        "fn_spoof_as_real": int(fn),
        "tp_spoof": int(tp),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_spoof": float(precision_score(y_true, pred, zero_division=0)),
        "recall_spoof": float(recall_score(y_true, pred, zero_division=0)),
        "f1_spoof": float(f1_score(y_true, pred, zero_division=0)),
        "APCER": float(apcer),
        "BPCER": float(bpcer),
        "ACER": float(acer),
        "AUC": float(auc),
        "threshold": float(threshold),
    }


def find_best_threshold(y_true, scores):
    best = None

    for thr in np.linspace(0.05, 0.95, 181):
        m = compute_metrics(y_true, scores, threshold=float(thr))

        # Objectif PAD :
        # 1. minimiser ACER
        # 2. minimiser APCER pour sécurité
        # 3. maximiser F1
        key = (m["ACER"], m["APCER"], -m["f1_spoof"])

        if best is None or key < best["key"]:
            best = {
                "threshold": float(thr),
                "metrics": m,
                "key": key,
            }

    return best


def sample_sequence(g, T):
    g = g.sort_values("frame_idx").reset_index(drop=True)
    n = len(g)

    if n >= T:
        start = max(0, (n - T) // 2)
        g = g.iloc[start:start + T].copy()
    else:
        last = g.iloc[[-1]].copy()
        pads = [last.copy() for _ in range(T - n)]
        g = pd.concat([g] + pads, ignore_index=True)

    return g


class BehaviorSeqDataset(Dataset):
    def __init__(self, csv_path, T=32, scaler=None, fit_scaler=False):
        self.df = pd.read_csv(csv_path)
        self.df["video_id"] = self.df["video_id"].astype(str)
        self.T = T
        self.video_ids = sorted(self.df["video_id"].unique().tolist())

        missing = [c for c in FEATURE_COLS if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        if "label" not in self.df.columns:
            raise ValueError("CSV must contain label column.")

        if fit_scaler:
            all_values = []

            for vid in self.video_ids:
                g = self.df[self.df["video_id"] == vid]
                seq = sample_sequence(g, T)
                all_values.append(seq[FEATURE_COLS].fillna(0.0).values)

            all_values = np.concatenate(all_values, axis=0)
            self.scaler = StandardScaler()
            self.scaler.fit(all_values)
        else:
            if scaler is None:
                raise ValueError("Scaler is required when fit_scaler=False")
            self.scaler = scaler

    def __len__(self):
        return len(self.video_ids)

    def get_label(self, idx):
        vid = self.video_ids[idx]
        g = self.df[self.df["video_id"] == vid]
        return int(g["label"].iloc[0])

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        g = self.df[self.df["video_id"] == vid]
        seq = sample_sequence(g, self.T)

        x = seq[FEATURE_COLS].fillna(0.0).values.astype(np.float32)
        x = self.scaler.transform(x).astype(np.float32)

        y = int(seq["label"].iloc[0])

        return {
            "video_id": vid,
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
        }


class BehaviorLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=1,
        dropout=0.4,
        bidirectional=False,
    ):
        super().__init__()

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else dropout,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        out, (h, c) = self.lstm(x)

        if self.bidirectional:
            z = torch.cat([h[-2], h[-1]], dim=1)
        else:
            z = h[-1]

        logits = self.head(z)
        return logits


def build_weighted_sampler(dataset):
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])

    real_count = np.sum(labels == 0)
    spoof_count = np.sum(labels == 1)

    if real_count == 0 or spoof_count == 0:
        raise ValueError("Both REAL and SPOOF classes are required for weighted sampling.")

    class_counts = np.array([real_count, spoof_count])
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler, labels


def build_class_weighted_loss(labels, device):
    labels = np.asarray(labels).astype(int)

    real_count = np.sum(labels == 0)
    spoof_count = np.sum(labels == 1)
    total = len(labels)

    if real_count == 0 or spoof_count == 0:
        raise ValueError("Both REAL and SPOOF classes are required for class weighting.")

    weight_real = total / (2.0 * real_count)
    weight_spoof = total / (2.0 * spoof_count)

    class_weights = torch.tensor(
        [weight_real, weight_spoof],
        dtype=torch.float32,
        device=device,
    )

    print("Class counts: REAL =", real_count, "| SPOOF =", spoof_count)
    print("Class weights:", class_weights.detach().cpu().numpy())

    return nn.CrossEntropyLoss(weight=class_weights)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    all_y = []
    all_scores = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()

        losses.append(float(loss.item()))
        all_y.extend(y.detach().cpu().numpy().tolist())
        all_scores.extend(probs.tolist())

    metrics = compute_metrics(all_y, all_scores, threshold=0.5)
    return float(np.mean(losses)), metrics, np.array(all_y), np.array(all_scores)


def predict_dataset(model, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    rows = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy()
            vids = batch["video_id"]

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds_05 = (probs >= 0.5).astype(int)

            for vid, yy, score, pred in zip(vids, y, probs, preds_05):
                rows.append({
                    "video_id": str(vid),
                    "label": int(yy),
                    "score_behavior_temporal": float(score),
                    "pred_behavior_temporal_05": int(pred),
                })

    return pd.DataFrame(rows)


def add_predictions_with_threshold(df, threshold):
    df = df.copy()
    df["pred_behavior_temporal"] = (
        df["score_behavior_temporal"].astype(float) >= threshold
    ).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--bidirectional", type=int, default=0)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = BehaviorSeqDataset(args.train_csv, T=args.T, fit_scaler=True)
    val_ds = BehaviorSeqDataset(args.val_csv, T=args.T, scaler=train_ds.scaler)
    test_ds = BehaviorSeqDataset(args.test_csv, T=args.T, scaler=train_ds.scaler)

    sampler, train_labels = build_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = BehaviorLSTM(
        input_dim=len(FEATURE_COLS),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=bool(args.bidirectional),
    ).to(device)

    criterion = build_class_weighted_loss(train_labels, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_acer = 999.0
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train=True,
        )

        val_loss, val_metrics_05, val_y, val_scores = run_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            train=False,
        )

        best_thr_info = find_best_threshold(val_y, val_scores)
        val_metrics_best = best_thr_info["metrics"]
        val_best_threshold = best_thr_info["threshold"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ACER_05": train_metrics["ACER"],
            "val_ACER_05": val_metrics_05["ACER"],
            "val_ACER_best_thr": val_metrics_best["ACER"],
            "val_best_threshold": val_best_threshold,
            "val_F1_best_thr": val_metrics_best["f1_spoof"],
            "val_AUC": val_metrics_05["AUC"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_ACER@0.5={val_metrics_05['ACER']:.4f} | "
            f"val_ACER@best={val_metrics_best['ACER']:.4f} | "
            f"thr={val_best_threshold:.3f} | "
            f"val_AUC={val_metrics_05['AUC']:.4f}"
        )

        # Early stopping sur ACER avec seuil optimisé
        if val_metrics_best["ACER"] < best_acer:
            best_acer = val_metrics_best["ACER"]
            best_state = {
                "model_state": model.state_dict(),
                "feature_cols": FEATURE_COLS,
                "T": args.T,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "bidirectional": bool(args.bidirectional),
                "best_threshold_from_val": val_best_threshold,
            }
            wait = 0
        else:
            wait += 1

        if wait >= args.patience:
            print("Early stopping")
            break

    if best_state is None:
        raise RuntimeError("No best model saved.")

    model.load_state_dict(best_state["model_state"])
    best_threshold = float(best_state["best_threshold_from_val"])

    train_pred = predict_dataset(model, train_ds, device, batch_size=args.batch_size)
    val_pred = predict_dataset(model, val_ds, device, batch_size=args.batch_size)
    test_pred = predict_dataset(model, test_ds, device, batch_size=args.batch_size)

    train_pred = add_predictions_with_threshold(train_pred, best_threshold)
    val_pred = add_predictions_with_threshold(val_pred, best_threshold)
    test_pred = add_predictions_with_threshold(test_pred, best_threshold)

    train_metrics = compute_metrics(
        train_pred["label"],
        train_pred["score_behavior_temporal"],
        threshold=best_threshold,
    )

    val_metrics = compute_metrics(
        val_pred["label"],
        val_pred["score_behavior_temporal"],
        threshold=best_threshold,
    )

    test_metrics = compute_metrics(
        test_pred["label"],
        test_pred["score_behavior_temporal"],
        threshold=best_threshold,
    )

    train_pred.to_csv(out_dir / "train_behavior_temporal_predictions.csv", index=False, encoding="utf-8")
    val_pred.to_csv(out_dir / "val_behavior_temporal_predictions.csv", index=False, encoding="utf-8")
    test_pred.to_csv(out_dir / "test_behavior_temporal_predictions.csv", index=False, encoding="utf-8")

    final_report = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "best_threshold_from_val": best_threshold,
        "feature_cols": FEATURE_COLS,
        "T": args.T,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": bool(args.bidirectional),
        "notes": {
            "class_weighted_loss": True,
            "weighted_random_sampler": True,
            "threshold_optimized_on_validation": True,
        },
    }

    with (out_dir / "metrics_behavior_temporal.json").open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    torch.save(best_state, out_dir / "best_behavior_lstm_v5_balanced.pth")

    with (out_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(train_ds.scaler, f)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8")

    print("\\n========== FINAL TEST METRICS ==========")
    print(json.dumps(test_metrics, indent=2))
    print("\\nBest threshold from validation:", best_threshold)
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
