from torchgeo.datasets import TropicalCyclone

dataset = TropicalCyclone(
    root="examples/TropicalCyclone/data",
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