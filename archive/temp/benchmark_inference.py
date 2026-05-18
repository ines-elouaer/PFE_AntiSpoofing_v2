import time
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

def build_model():
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model

@torch.no_grad()
def benchmark(device="cuda", runs=200, warmup=50, batch_size=1, img_size=224):
    model = build_model().to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    
    for _ in range(warmup):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) * 1000 / runs
    print(f"Device={device} | batch={batch_size} | avg inference = {avg_ms:.2f} ms")

def main():
    if torch.cuda.is_available():
        benchmark(device="cuda", batch_size=1)
        benchmark(device="cuda", batch_size=8)
    benchmark(device="cpu", batch_size=1)

if __name__ == "__main__":
    main()
