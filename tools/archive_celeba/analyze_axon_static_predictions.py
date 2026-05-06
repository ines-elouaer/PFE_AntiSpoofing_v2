from pathlib import Path
import pandas as pd

p = Path(r"E:\PFE_AntiSpoofing_v2\reports\axon_static_celeba_eval\predictions.csv")
df = pd.read_csv(p)

spoof = df[df["label"] == 1]
real = df[df["label"] == 0]

print("\n=== DECISIONS GLOBALES ===")
print(df["banking_decision"].value_counts())

print("\n=== SPOOF : DECISION BANCAIRE ===")
print(spoof["banking_decision"].value_counts())

print("\n=== REAL : DECISION BANCAIRE ===")
print(real["banking_decision"].value_counts())

bad = spoof[spoof["banking_decision"] == "ACCEPT"].copy()
retry = spoof[spoof["banking_decision"] == "RETRY"].copy()

print("\n=== ATTAQUES ACCEPTEES A TORT ===")
print("Nombre:", len(bad))
print(bad["attack_type"].value_counts())

print("\n=== ATTAQUES EN RETRY ===")
print("Nombre:", len(retry))
print(retry["attack_type"].value_counts())

out_dir = Path(r"E:\PFE_AntiSpoofing_v2\reports\axon_static_celeba_eval")
bad.to_csv(out_dir / "dangerous_false_accepts.csv", index=False)
retry.to_csv(out_dir / "retry_spoof_cases.csv", index=False)

print("\nFichiers sauvegardés :")
print(out_dir / "dangerous_false_accepts.csv")
print(out_dir / "retry_spoof_cases.csv")
