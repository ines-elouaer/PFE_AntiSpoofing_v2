import json
import os

null_count = 0
ok_count = 0
files_checked = 0

for root, dirs, files in os.walk("reports/eval"):
    for f in files:
        if f != "test_scores.json":
            continue

        path = os.path.join(root, f)
        files_checked += 1

        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            cm = data.get("artifacts", {}).get("confusion_matrix_csv", None)

            if cm is None:
                null_count += 1
                print(f"❌ Missing CM: {path}")
            else:
                ok_count += 1
                print(f"✅ OK: {path}")

        except Exception as e:
            print(f"⚠️ Error reading {path}: {e}")

print("\n===== SUMMARY =====")
print(f"Total files checked: {files_checked}")
print(f"Missing confusion matrix: {null_count}")
print(f"Valid confusion matrix: {ok_count}")