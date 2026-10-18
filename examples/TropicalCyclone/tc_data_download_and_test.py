from torchgeo.datasets import TropicalCyclone
from pathlib import Path

dataset = TropicalCyclone(
    root=str(Path(__file__).resolve().parent / "data"),
    split="train",
)

print("Dataset size:", len(dataset))

sample = dataset[0]

print(sample.keys())

for k, v in sample.items():
    if hasattr(v, "shape"):
        print(k, v.shape)
    else:
        print(k, v)