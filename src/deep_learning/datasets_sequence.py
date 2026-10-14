import csv
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.deep_learning.datasets_frame import build_transforms


def read_rows(csv_path: str) -> List[Dict]:
    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["label"] = int(row["label"])
            row["frame_idx"] = int(row["frame_idx"])
            rows.append(row)
    return rows


def uniform_sample(sorted_rows: List[Dict], T: int) -> List[Dict]:
    n = len(sorted_rows)
    if n == 0:
        raise RuntimeError("Video without frames.")
    if n >= T:
        idxs = [int(round(i * (n - 1) / (T - 1))) for i in range(T)]
        return [sorted_rows[i] for i in idxs]
    out = list(sorted_rows)
    while len(out) < T:
        out.append(sorted_rows[-1])
    return out


def consecutive_sample(sorted_rows: List[Dict], T: int, rng: random.Random) -> List[Dict]:
    n = len(sorted_rows)
    if n == 0:
        raise RuntimeError("Video without frames.")

    if n >= T:
        start = rng.randint(0, n - T)
        return sorted_rows[start:start + T]

    out = list(sorted_rows)
    while len(out) < T:
        out.append(sorted_rows[-1])
    return out


def center_consecutive_sample(sorted_rows: List[Dict], T: int) -> List[Dict]:
    n = len(sorted_rows)
    if n == 0:
        raise RuntimeError("Video without frames.")

    if n >= T:
        start = max(0, (n - T) // 2)
        return sorted_rows[start:start + T]

    out = list(sorted_rows)
    while len(out) < T:
        out.append(sorted_rows[-1])
    return out


def read_behav_csv(behav_csv: str):
    feat_cols = [
        "ear_mean",
        "ear_std",
        "ear_min",
        "ear_max",
        "blink_count",
        "motion_mean",
        "motion_std",
        "motion_max",
        "skipped_rate",
        "rppg_dominant_freq",
        "rppg_hr_estimate",
        "rppg_snr",
        "rppg_signal_std",
        "rppg_skipped_rate",
        "rppg_valid",
    ]

    by_vid = {}
    with Path(behav_csv).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)

        for row in r:
            vid = row["video_id"]
            vec = [float(row.get(col, 0.0)) for col in feat_cols]
            by_vid[vid] = torch.tensor(vec, dtype=torch.float32)

    return by_vid, len(feat_cols)

class CASIASequenceDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        T: int = 16,
        img_size: int = 224,
        aug_mode: str = "none",
        sample_mode: str = "uniform",   # "uniform" | "random_clip" | "consecutive" | "center_consecutive"
        seed: int = 42,
        behav_csv: Optional[str] = None,
    ):
        super().__init__()
        self.T = T
        self.sample_mode = sample_mode
        self.rng = random.Random(seed)
        self.tf = build_transforms(img_size, aug_mode)

        rows = read_rows(csv_path)

        by_vid = defaultdict(list)
        for row in rows:
            by_vid[row["video_id"]].append(row)

        self.video_ids = sorted(by_vid.keys())
        self.rows_by_vid: Dict[str, List[Dict]] = {}
        self.labels_by_vid: Dict[str, int] = {}

        for vid in self.video_ids:
            vid_rows = sorted(by_vid[vid], key=lambda r: r["frame_idx"])
            labels = {r["label"] for r in vid_rows}
            if len(labels) != 1:
                raise RuntimeError(f"Inconsistent labels in video_id={vid}: {labels}")

            self.rows_by_vid[vid] = vid_rows
            self.labels_by_vid[vid] = vid_rows[0]["label"]

        self.behav_by_vid = None
        self.behav_dim = 0
        if behav_csv is not None:
            self.behav_by_vid, self.behav_dim = read_behav_csv(behav_csv)

    def __len__(self):
        return len(self.video_ids)

    def _sample_rows(self, rows: List[Dict]) -> List[Dict]:
        n = len(rows)
        if n == 0:
            raise RuntimeError("Video without frames.")

        if self.sample_mode == "uniform":
            return uniform_sample(rows, self.T)

        if self.sample_mode == "random_clip":
            if n >= self.T:
                start = self.rng.randint(0, n - self.T)
                return rows[start:start + self.T]
            return uniform_sample(rows, self.T)

        if self.sample_mode == "consecutive":
            return consecutive_sample(rows, self.T, self.rng)

        if self.sample_mode == "center_consecutive":
            return center_consecutive_sample(rows, self.T)

        raise ValueError(f"Invalid sample_mode: {self.sample_mode}")

    def __getitem__(self, idx: int):
        vid = self.video_ids[idx]
        rows = self.rows_by_vid[vid]
        y = self.labels_by_vid[vid]

        sampled = self._sample_rows(rows)

        frames = []
        for r in sampled:
            img = Image.open(r["path"]).convert("RGB")
            frames.append(self.tf(img))
        x = torch.stack(frames, dim=0)

        if self.behav_by_vid is None:
            return x, torch.tensor(y, dtype=torch.long), vid

        behav = self.behav_by_vid.get(vid, torch.zeros(self.behav_dim, dtype=torch.float32))
        return x, torch.tensor(y, dtype=torch.long), vid, behav