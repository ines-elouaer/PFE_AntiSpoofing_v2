from pathlib import Path
import json
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_paths(project_root: Path):
    config_path = project_root / "configs" / "paths.json"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    axon_prepared_dir = Path(cfg["axon_prepared_dir"])

    if not axon_prepared_dir.is_absolute():
        axon_prepared_dir = (project_root / axon_prepared_dir).resolve()

    return axon_prepared_dir


def main():
    project_root = get_project_root()
    axon_prepared_dir = load_paths(project_root)

    csv_path = axon_prepared_dir / "manifests" / "axon_video_frames_manifest.csv"
    report_dir = project_root / "reports" / "axon_video_frames_check"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV introuvable: {csv_path}\n"
            "Lance d'abord: python tools\\axon\\prepare_axon_video_frames.py"
        )

    df = pd.read_csv(csv_path)

    print("========== CONFIG ==========")
    print(f"Project root : {project_root}")
    print(f"CSV          : {csv_path}")
    print(f"Rows         : {len(df)}")

    print("\n========== BASIC COUNTS ==========")
    print("Unique videos:", df["video_id"].nunique())
    print("\nLabels frames:")
    print(df["label_name"].value_counts())

    print("\nLabels videos:")
    print(df.drop_duplicates("video_id")["label_name"].value_counts())

    print("\nAttack type videos:")
    print(df.drop_duplicates("video_id")["attack_type"].value_counts())

    print("\nLevel videos:")
    print(df.drop_duplicates("video_id")["level"].value_counts())

    print("\n========== FRAMES PER VIDEO ==========")
    frames_per_video = df.groupby("video_id")["frame_idx"].nunique()
    print(frames_per_video.describe())

    bad_counts = frames_per_video[frames_per_video < 16]
    print("\nVideos avec moins de 16 frames:", len(bad_counts))

    missing = []

    for _, row in df.iterrows():
        p = project_root / str(row["path"])
        if not p.exists():
            missing.append(row.to_dict())

    print("\n========== FILE CHECK ==========")
    print("Missing frame files:", len(missing))

    missing_path = report_dir / "missing_frames.csv"
    bad_counts_path = report_dir / "videos_less_than_16_frames.csv"
    samples_path = report_dir / "random_samples.csv"

    if missing:
        pd.DataFrame(missing).to_csv(missing_path, index=False, encoding="utf-8")

    if len(bad_counts) > 0:
        bad_counts.reset_index(name="num_frames").to_csv(
            bad_counts_path,
            index=False,
            encoding="utf-8",
        )

    df.sample(min(50, len(df)), random_state=42).to_csv(
        samples_path,
        index=False,
        encoding="utf-8",
    )

    print("\n========== OUTPUTS ==========")
    print(f"Random samples : {samples_path}")

    if missing:
        print(f"Missing frames : {missing_path}")

    if len(bad_counts) > 0:
        print(f"Bad counts     : {bad_counts_path}")

    print("\n[OK] Vérification terminée.")


if __name__ == "__main__":
    main()