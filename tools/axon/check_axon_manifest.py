from pathlib import Path
import json
import csv
from collections import Counter
import random


def load_config():
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "paths.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    axon_raw_dir = Path(cfg["axon_raw_dir"])
    if not axon_raw_dir.is_absolute():
        axon_raw_dir = (project_root / axon_raw_dir).resolve()
    else:
        axon_raw_dir = axon_raw_dir.resolve()

    axon_prepared_dir = Path(cfg["axon_prepared_dir"])
    if not axon_prepared_dir.is_absolute():
        axon_prepared_dir = (project_root / axon_prepared_dir).resolve()
    else:
        axon_prepared_dir = axon_prepared_dir.resolve()

    return project_root, axon_raw_dir, axon_prepared_dir


def read_csv_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV introuvable: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_project_path(project_root: Path, rel_path: str) -> Path:
    return (project_root / rel_path).resolve()


def resolve_axon_path(axon_raw_dir: Path, rel_path: str) -> Path:
    return (axon_raw_dir / rel_path).resolve()


def print_counter(title: str, counter: Counter):
    print(f"\n========== {title} ==========")

    if not counter:
        print("(vide)")
        return

    for key, value in sorted(counter.items(), key=lambda x: str(x[0])):
        print(f"{key}: {value}")


def save_random_samples(rows, output_path: Path, n: int = 30, seed: int = 42):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    sample = rows[:]

    if len(sample) > n:
        sample = rng.sample(sample, n)

    if not sample:
        return

    fieldnames = list(sample[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample)


def check_static_manifest(project_root: Path, static_csv: Path):
    rows = read_csv_rows(static_csv)

    print("\n\n##################################################")
    print("# CHECK STATIC MANIFEST")
    print("##################################################")
    print(f"CSV: {static_csv}")
    print(f"Rows: {len(rows)}")

    label_counts = Counter()
    attack_counts = Counter()
    source_counts = Counter()
    level_counts = Counter()

    missing = []
    invalid_label = []

    for row in rows:
        label = row.get("label")
        label_name = row.get("label_name")
        attack_type = row.get("attack_type")
        source = row.get("source")
        level = row.get("level")

        label_counts[label_name] += 1
        attack_counts[attack_type] += 1
        source_counts[source] += 1
        level_counts[level] += 1

        if label not in {"0", "1"}:
            invalid_label.append(row)

        rel = row.get("prepared_relative_path", "")
        full_path = resolve_project_path(project_root, rel)

        if not full_path.exists():
            missing.append(row)

    print_counter("STATIC LABEL COUNTS", label_counts)
    print_counter("STATIC ATTACK TYPE COUNTS", attack_counts)
    print_counter("STATIC SOURCE COUNTS", source_counts)
    print_counter("STATIC LEVEL COUNTS", level_counts)

    print("\n========== STATIC FILE CHECK ==========")
    print(f"Missing files: {len(missing)}")
    print(f"Invalid labels: {len(invalid_label)}")

    return {
        "rows": rows,
        "missing": missing,
        "invalid_label": invalid_label,
        "label_counts": label_counts,
        "attack_counts": attack_counts,
    }


def check_video_manifest(axon_raw_dir: Path, video_csv: Path):
    rows = read_csv_rows(video_csv)

    print("\n\n##################################################")
    print("# CHECK VIDEO MANIFEST")
    print("##################################################")
    print(f"CSV: {video_csv}")
    print(f"Rows: {len(rows)}")

    label_counts = Counter()
    attack_counts = Counter()
    media_group_counts = Counter()
    level_counts = Counter()

    missing = []
    invalid_label = []

    for row in rows:
        label = row.get("label")
        label_name = row.get("label_name")
        attack_type = row.get("attack_type")
        media_group = row.get("media_group")
        level = row.get("level")

        label_counts[label_name] += 1
        attack_counts[attack_type] += 1
        media_group_counts[media_group] += 1
        level_counts[level] += 1

        if label not in {"0", "1"}:
            invalid_label.append(row)

        rel = row.get("raw_relative_path", "")
        full_path = resolve_axon_path(axon_raw_dir, rel)

        if not full_path.exists():
            missing.append(row)

    print_counter("VIDEO LABEL COUNTS", label_counts)
    print_counter("VIDEO ATTACK TYPE COUNTS", attack_counts)
    print_counter("VIDEO MEDIA GROUP COUNTS", media_group_counts)
    print_counter("VIDEO LEVEL COUNTS", level_counts)

    print("\n========== VIDEO FILE CHECK ==========")
    print(f"Missing files: {len(missing)}")
    print(f"Invalid labels: {len(invalid_label)}")

    return {
        "rows": rows,
        "missing": missing,
        "invalid_label": invalid_label,
        "label_counts": label_counts,
        "attack_counts": attack_counts,
        "media_group_counts": media_group_counts,
    }


def save_problem_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    project_root, axon_raw_dir, axon_prepared_dir = load_config()

    manifest_dir = axon_prepared_dir / "manifests"

    static_csv = manifest_dir / "axon_static_manifest.csv"
    video_csv = manifest_dir / "axon_video_manifest.csv"

    reports_dir = project_root / "reports" / "axon_manifest_check"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("========== CONFIG ==========")
    print(f"Project root      : {project_root}")
    print(f"Axon raw dir      : {axon_raw_dir}")
    print(f"Axon prepared dir : {axon_prepared_dir}")
    print(f"Static CSV        : {static_csv}")
    print(f"Video CSV         : {video_csv}")

    static_result = check_static_manifest(project_root, static_csv)
    video_result = check_video_manifest(axon_raw_dir, video_csv)

    # Sauvegarder échantillons pour vérification manuelle
    save_random_samples(
        static_result["rows"],
        reports_dir / "static_random_samples.csv",
        n=40,
    )

    save_random_samples(
        video_result["rows"],
        reports_dir / "video_random_samples.csv",
        n=40,
    )

    save_problem_rows(
        reports_dir / "static_missing_files.csv",
        static_result["missing"],
    )

    save_problem_rows(
        reports_dir / "video_missing_files.csv",
        video_result["missing"],
    )

    print("\n\n========== OUTPUTS ==========")
    print(f"Random static samples : {reports_dir / 'static_random_samples.csv'}")
    print(f"Random video samples  : {reports_dir / 'video_random_samples.csv'}")

    if static_result["missing"]:
        print(f"Static missing files  : {reports_dir / 'static_missing_files.csv'}")

    if video_result["missing"]:
        print(f"Video missing files   : {reports_dir / 'video_missing_files.csv'}")

    print("\n[OK] Vérification terminée.")


if __name__ == "__main__":
    main()