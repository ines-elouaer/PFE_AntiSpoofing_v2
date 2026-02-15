import csv
import math
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from PIL import Image

def read_rows(csv_path: str):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["label"] = int(row["label"])
            row["frame_idx"] = int(row["frame_idx"])
            rows.append(row)
    return rows

def uniform_sample(sorted_rows, T: int):
   
    n = len(sorted_rows)
    if n == 0:
        raise RuntimeError("Vidéo sans frames.")
    if n >= T:
       
        idxs = [int(round(i*(n-1)/(T-1))) for i in range(T)]
        return [sorted_rows[i] for i in idxs]
    else:
      
        out = list(sorted_rows)
        while len(out) < T:
            out.append(sorted_rows[-1])
        return out

class CASIASequenceDataset(Dataset):
    
    def __init__(self, csv_path: str, T: int = 16, img_size: int = 224, transform=None):
        self.T = T
        self.img_size = img_size
        self.transform = transform

        rows = read_rows(csv_path)

        by_vid = defaultdict(list)
        for row in rows:
            by_vid[row["video_id"]].append(row)

        self.video_ids = sorted(by_vid.keys())
        self.rows_by_vid = {}
        self.labels_by_vid = {}

        for vid in self.video_ids:
            vid_rows = sorted(by_vid[vid], key=lambda r: r["frame_idx"])
            self.rows_by_vid[vid] = vid_rows
            
            self.labels_by_vid[vid] = vid_rows[0]["label"]

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        rows = self.rows_by_vid[vid]
        label = self.labels_by_vid[vid]

        sampled = uniform_sample(rows, self.T)

        frames = []
        for r in sampled:
            img = Image.open(r["path"]).convert("RGB")
            img = img.resize((self.img_size, self.img_size))
            if self.transform is not None:
                img_t = self.transform(img)
            else:

                img_t = torch.from_numpy(
                    (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                     .view(img.size[1], img.size[0], 3)
                     .numpy()).copy()
                ).float() / 255.0
                img_t = img_t.permute(2, 0, 1) 
            frames.append(img_t)

        frames_tensor = torch.stack(frames, dim=0) 
        return frames_tensor, torch.tensor(label, dtype=torch.long), vid
