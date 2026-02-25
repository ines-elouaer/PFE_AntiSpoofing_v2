
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Colonnes comportementales à normaliser
FEAT_COLS = [
    "ear_mean",
    "ear_std",
    "ear_min",
    "ear_max",
    "blink_count",
    "motion_mean",
    "motion_std",
    "motion_max",
    "skipped_rate",
]


BEHAV_DIR = Path("data/processed/casia/behav")

TRAIN_IN = BEHAV_DIR / "train_behav.csv"
VAL_IN   = BEHAV_DIR / "val_behav.csv"
TEST_IN  = BEHAV_DIR / "test_behav.csv"

TRAIN_OUT = BEHAV_DIR / "train_behav_norm.csv"
VAL_OUT   = BEHAV_DIR / "val_behav_norm.csv"
TEST_OUT  = BEHAV_DIR / "test_behav_norm.csv"

SCALER_OUT = BEHAV_DIR / "behav_scaler.pkl"


def _check_exists(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def main():
    _check_exists(TRAIN_IN)
    _check_exists(VAL_IN)
    _check_exists(TEST_IN)

    train = pd.read_csv(TRAIN_IN)
    val   = pd.read_csv(VAL_IN)
    test  = pd.read_csv(TEST_IN)

    print(f"Train videos: {len(train)} | Val: {len(val)} | Test: {len(test)}")

  
    for col in FEAT_COLS:
        for name, df in [("train", train), ("val", val), ("test", test)]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing in {name}: {df.columns.tolist()}")

   
    for df in [train, val, test]:
        df[FEAT_COLS] = df[FEAT_COLS].replace([float("inf"), float("-inf")], 0.0).fillna(0.0)

    scaler = StandardScaler()

    
    train[FEAT_COLS] = scaler.fit_transform(train[FEAT_COLS])

   
    val[FEAT_COLS]   = scaler.transform(val[FEAT_COLS])
    test[FEAT_COLS]  = scaler.transform(test[FEAT_COLS])

    BEHAV_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    val.to_csv(VAL_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)

    print("\n✅ Saved normalized CSVs:")
    print(" -", TRAIN_OUT)
    print(" -", VAL_OUT)
    print(" -", TEST_OUT)
    print("✅ Saved scaler:", SCALER_OUT)

    
    print("\nSanity (train mean≈0, std≈1):")
    print(train[FEAT_COLS].describe().loc[["mean", "std"]].round(3).to_string())


if __name__ == "__main__":
    main()