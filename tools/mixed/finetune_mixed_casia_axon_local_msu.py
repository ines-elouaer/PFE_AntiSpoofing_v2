from pathlib import Path
import importlib.util
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_SCRIPT = PROJECT_ROOT / "tools" / "axon" / "finetune_mixed_casia_axon.py"

if not BASE_SCRIPT.exists():
    raise FileNotFoundError(f"Script base introuvable: {BASE_SCRIPT}")


# ==========================================================
# Charger le script original sans le modifier
# ==========================================================

spec = importlib.util.spec_from_file_location(
    "finetune_mixed_casia_axon_base",
    str(BASE_SCRIPT),
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


# ==========================================================
# Nouvelle configuration : CASIA + AXON + LOCAL_REAL + MSU
# ==========================================================

MIXED_DIR = PROJECT_ROOT / "data" / "mixed_casia_axon_local_msu"

CHECKPOINT_BASE = (
    PROJECT_ROOT
    / "experiments"
    / "mixed_casia_axon"
    / "seed42"
    / "best_model_mixed_casia_axon.pth"
)

OUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "mixed_casia_axon_local_msu"
    / "seed42"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Overriding des chemins du script original
# ==========================================================

base.CHECKPOINT_CASIA = CHECKPOINT_BASE

base.MIXED_DIR = MIXED_DIR

base.TRAIN_FRAMES = MIXED_DIR / "mixed_train_frames.csv"
base.VAL_FRAMES = MIXED_DIR / "mixed_val_frames.csv"
base.TRAIN_BEHAV = MIXED_DIR / "mixed_train_behav.csv"
base.VAL_BEHAV = MIXED_DIR / "mixed_val_behav.csv"

base.OUT_DIR = OUT_DIR

base.BEST_MODEL_PATH = OUT_DIR / "best_model_mixed_casia_axon_local_msu.pth"
base.LAST_MODEL_PATH = OUT_DIR / "last_model_mixed_casia_axon_local_msu.pth"
base.HISTORY_PATH = OUT_DIR / "training_history.json"
base.CONFIG_PATH = OUT_DIR / "config.json"


# ==========================================================
# Fine-tuning léger
# ==========================================================

base.PHASE1_EPOCHS = 3
base.PHASE2_EPOCHS = 0

base.LR_PHASE1 = 5e-5
base.LR_PHASE2 = 1e-5

base.BATCH_SIZE = 4
base.NUM_WORKERS = 0

base.T = 16
base.IMG_SIZE = 224

base.USE_BEHAV = True
base.BEHAV_DIM = 9
base.BEHAV_HIDDEN = 16
base.TEMPORAL_POOL = "median"

base.WEIGHT_DECAY = 1e-4


# ==========================================================
# Sampler équilibré REAL/SPOOF + équilibrage par domaine
# ==========================================================

def build_label_domain_balanced_sampler(dataset, frames_csv: Path, min_count: int = 1):
    """
    Sampler adapté à notre problème webcam.

    Objectif :
    - 50 % REAL / 50 % SPOOF pendant l'entraînement
    - équilibrer les sous-domaines REAL :
      CASIA_REAL, AXON_REAL, LOCAL_REAL, MSU_MFSD
    - équilibrer les types SPOOF par dataset / attack_type

    Pourquoi :
    Le train brut reste déséquilibré :
    REAL  = 146
    SPOOF = 578

    Donc on ne veut pas entraîner naïvement sur la distribution brute.
    """

    video_df = base.load_video_meta(frames_csv)
    meta_by_vid = video_df.set_index("video_id").to_dict(orient="index")

    groups = []
    labels = []

    for vid in dataset.video_ids:
        meta = meta_by_vid.get(vid, {})

        label = int(meta.get("label", dataset.labels_by_vid[vid]))
        source_dataset = str(meta.get("source_dataset", meta.get("domain", "UNKNOWN")))
        attack_type = str(meta.get("attack_type", "unknown"))

        labels.append(label)

        if label == 0:
            group = f"{source_dataset}::REAL"
        else:
            group = f"{source_dataset}::SPOOF::{attack_type}"

        groups.append(group)

    group_counts = Counter(groups)
    label_counts = Counter(labels)

    groups_by_label = defaultdict(set)

    for label, group in zip(labels, groups):
        groups_by_label[label].add(group)

    groups_by_label = {
        label: sorted(list(group_set))
        for label, group_set in groups_by_label.items()
    }

    # masse cible : 50 % REAL, 50 % SPOOF
    target_label_mass = {
        0: 0.5,
        1: 0.5,
    }

    weights = []

    for label, group in zip(labels, groups):
        num_groups_for_label = max(1, len(groups_by_label[label]))
        target_group_mass = target_label_mass[label] / num_groups_for_label
        sample_weight = target_group_mass / max(1, group_counts[group])
        weights.append(sample_weight)

    weights = np.array(weights, dtype=np.float64)
    weights = weights / weights.sum()

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

    print("\n========== LABEL COUNTS RAW ==========")
    print(f"REAL  : {label_counts.get(0, 0)}")
    print(f"SPOOF : {label_counts.get(1, 0)}")

    print("\n========== BALANCED SAMPLER GROUPS ==========")
    for k, v in sorted(group_counts.items(), key=lambda x: x[0]):
        print(f"{k:<45}: {v}")

    print("\n========== TARGET SAMPLING ==========")
    print("REAL  target mass : 0.50")
    print("SPOOF target mass : 0.50")

    return sampler, group_counts


# Remplacer le sampler original par notre sampler équilibré
base.build_stratified_sampler = build_label_domain_balanced_sampler


# ==========================================================
# Class weights neutres
# ==========================================================

def compute_neutral_class_weights(dataset, device):
    """
    Comme le sampler force déjà un apprentissage équilibré REAL/SPOOF,
    on évite de surpondérer encore la classe REAL avec les class weights bruts.

    CrossEntropyLoss utilisera donc weight=[1.0, 1.0].
    """

    labels = [dataset.labels_by_vid[vid] for vid in dataset.video_ids]
    counts = Counter(labels)

    print("\n========== CLASS WEIGHTS ==========")
    print(f"REAL count raw  : {counts.get(0, 0)}")
    print(f"SPOOF count raw : {counts.get(1, 0)}")
    print("weights used    : [1.0, 1.0]")

    return torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)


base.compute_class_weights = compute_neutral_class_weights


# ==========================================================
# Print config claire
# ==========================================================

def print_adaptation_config():
    print("\n" + "=" * 70)
    print("FINE-TUNING LOCAL/MSU — ADAPTATION WEBCAM / APPAREILS RÉELS")
    print("=" * 70)
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Base script         : {BASE_SCRIPT}")
    print(f"Checkpoint initial  : {CHECKPOINT_BASE}")
    print(f"Mixed dir           : {MIXED_DIR}")
    print(f"Train frames        : {base.TRAIN_FRAMES}")
    print(f"Val frames          : {base.VAL_FRAMES}")
    print(f"Train behav         : {base.TRAIN_BEHAV}")
    print(f"Val behav           : {base.VAL_BEHAV}")
    print(f"Output dir          : {OUT_DIR}")
    print(f"Best model out      : {base.BEST_MODEL_PATH}")
    print(f"Phase 1 epochs      : {base.PHASE1_EPOCHS}")
    print(f"Phase 2 epochs      : {base.PHASE2_EPOCHS}")
    print(f"LR phase 1          : {base.LR_PHASE1}")
    print(f"LR phase 2          : {base.LR_PHASE2}")
    print("Backbone            : gelé pendant le fine-tuning")
    print("Sampler             : REAL/SPOOF 50/50 + équilibrage par source")
    print("=" * 70 + "\n")


def main():
    required = [
        CHECKPOINT_BASE,
        base.TRAIN_FRAMES,
        base.VAL_FRAMES,
        base.TRAIN_BEHAV,
        base.VAL_BEHAV,
    ]

    for p in required:
        if not Path(p).exists():
            raise FileNotFoundError(f"Fichier introuvable: {p}")

    print_adaptation_config()

    base.main()


if __name__ == "__main__":
    main()