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
# Load a limited pool of storms
# --------------------------------------------------

def load_storm_subset(N=20000, seed=123):

    rng = np.random.default_rng(seed)

    label_folders = [
        f for f in LABEL_DIR.iterdir()
        if f.is_dir()
    ]


    # randomly choose storms to load
    selected_folders = rng.choice(
        label_folders,
        size=min(N, len(label_folders)),
        replace=False
    )


    images = []
    labels = []


    for i, label_folder in enumerate(selected_folders):

        if i % 1000 == 0:
            print(
                f"Loading image {i}/{len(selected_folders)}"
            )


        storm_id = label_folder.name.replace(
            "nasa_tropical_storm_competition_train_labels_",
            ""
        )


        # read wind speed label

        with open(label_folder / "labels.json") as f:
            label = json.load(f)

        wind_speed = float(label["wind_speed"])



        # find image

        image_folder = (
            SOURCE_DIR /
            f"nasa_tropical_storm_competition_train_source_{storm_id}"
        )

        image_file = image_folder / "image.jpg"


        if not image_file.exists():
            continue



        img = Image.open(image_file)

        img = img.convert("RGB")
        img = img.resize((366,366))

        img = np.array(img)



        images.append(img)
        labels.append(wind_speed)



    return (
        np.stack(images),
        np.array(labels).reshape(-1,1)
    )



# --------------------------------------------------
# Load only 20K storms
# --------------------------------------------------

POOL_SIZE = 20000

all_images, all_labels = load_storm_subset(
    N=POOL_SIZE,
    seed=123
)


print("\nLoaded storm pool")
print("images:", all_images.shape)
print("labels:", all_labels.shape)



# --------------------------------------------------
# Random subset experiments
# --------------------------------------------------

num_runs = 20
subset_size = 1000


results = []


for run in range(num_runs):

    print("\n============================")
    print(f"Experiment {run+1}/{num_runs}")
    print("============================")


    rng = np.random.default_rng(seed=run)



    # choose random storms for this run

    indices = rng.choice(
        len(all_images),
        size=subset_size,
        replace=False
    )


    train_images = all_images[indices]
    train_labels = all_labels[indices]


    print(
        "Subset size:",
        len(train_images)
    )


    # --------------------------------------------------
    # Wavelet features
    # --------------------------------------------------

    features = wavelet_array_features(
        train_images,
        wavelet="haar"
    )


    # --------------------------------------------------
    # Standardize wavelet space
    # --------------------------------------------------

    scaler = StandardScaler()

    features = scaler.fit_transform(features)



    # --------------------------------------------------
    # Feasible sets
    # --------------------------------------------------

    result = feasible_sets_features_nn_xy(
        features,
        train_labels,
        N_max=300,
        qx=0.1,
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
            "run": run,
            "seed": run,
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
    "tc_test_kernel_results.json",
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


print("\n============================")
print("Summary")
print("============================")


print(
    "Mean kernel size:",
    kernel_sizes.mean()
)


print(
    "Std kernel size:",
    kernel_sizes.std()
)


print(
    "95% interval:",
    np.percentile(
        kernel_sizes,
        [2.5,97.5]
    )
)