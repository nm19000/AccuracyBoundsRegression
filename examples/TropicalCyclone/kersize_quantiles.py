import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

from accuracy_bounds.classificationproblems.utils import wavelet_array_features
from accuracy_bounds.regressionproblems.feasible_sets import feasible_sets_features_nn_xy
from accuracy_bounds.regressionproblems.kersize_compute import compute_kernel_size
from utils import tc_uncertainty2


SOURCE_DIR = Path(
    "data/extracted/nasa_tropical_storm_competition_train_source"
)

LABEL_DIR = Path(
    "data/extracted/nasa_tropical_storm_competition_train_labels"
)


# --------------------------------------------------
# Load exactly 1K storms once
# --------------------------------------------------

def load_storm_subset(N=1000, seed=124):

    rng = np.random.default_rng(seed)

    label_folders = [
        f for f in LABEL_DIR.iterdir()
        if f.is_dir()
    ]

    # Randomly choose 1K storms
    selected_folders = rng.choice(
        label_folders,
        size=min(N, len(label_folders)),
        replace=False
    )

    images = []
    labels = []

    for i, label_folder in enumerate(selected_folders):

        if i % 100 == 0:
            print(f"Loading image {i}/{len(selected_folders)}")

        storm_id = label_folder.name.replace(
            "nasa_tropical_storm_competition_train_labels_",
            ""
        )

        # Read wind speed label
        with open(label_folder / "labels.json") as f:
            label = json.load(f)

        wind_speed = float(label["wind_speed"])

        # Find image
        image_folder = (
            SOURCE_DIR /
            f"nasa_tropical_storm_competition_train_source_{storm_id}"
        )

        image_file = image_folder / "image.jpg"

        if not image_file.exists():
            continue

        img = Image.open(image_file)
        img = img.convert("RGB")
        img = img.resize((366, 366))
        img = np.array(img)

        images.append(img)
        labels.append(wind_speed)

    return (
        np.stack(images),
        np.array(labels).reshape(-1, 1)
    )


# --------------------------------------------------
# Load the 1K storm sample ONCE
# --------------------------------------------------

SAMPLE_SIZE = 1000

train_images, train_labels = load_storm_subset(
    N=SAMPLE_SIZE,
    seed=123
)

print("\nLoaded storm sample")
print("images:", train_images.shape)
print("labels:", train_labels.shape)


# --------------------------------------------------
# Compute wavelet features ONCE
# --------------------------------------------------

print("\nComputing wavelet features...")

features = wavelet_array_features(
    train_images,
    wavelet="haar"
)


# --------------------------------------------------
# Standardize features ONCE
# --------------------------------------------------

scaler = StandardScaler()

features = scaler.fit_transform(features)


# --------------------------------------------------
# Random subset experiments
# --------------------------------------------------

Q_GRID = np.array(
    [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
    ],
    dtype=float,
)


results = []


for q in Q_GRID:

    print(f"\nRunning experiment for q={q}")

    # --------------------------------------------------
    # Feasible sets
    # --------------------------------------------------

    result = feasible_sets_features_nn_xy(
        features,
        train_labels,
        N_max=300,
        qx=q,
        target_uncertainty=tc_uncertainty2,
    )


    # --------------------------------------------------
    # Kernel size
    # --------------------------------------------------

    kersize, M_prime, Nk_stats = compute_kernel_size(
        result["feasible_sets"]
    )

    print("kernel size:", kersize)
    print("M prime:", M_prime)
    print("tau_x:", result["tau_x"])
    print("Nk stats:", Nk_stats)


    results.append(
        {
            "q": q,
            "kernel_size": float(kersize),
            "M_prime": int(M_prime),
            "tau_x": float(result["tau_x"]),
            "Nk_stats": Nk_stats,
        }
    )


# --------------------------------------------------
# Save results
# --------------------------------------------------

with open(
    "tc_train_kernel_results.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )


# --------------------------------------------------
# Summary
# --------------------------------------------------

kernel_sizes = np.array(
    [
        r["kernel_size"]
        for r in results
    ]
)

print("\nKernel sizes:")
print(kernel_sizes)