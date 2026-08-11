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
    "data/extracted/nasa_tropical_storm_competition_test_source"
)

LABEL_DIR = Path(
    "data/extracted/nasa_tropical_storm_competition_test_labels"
)


test_images = []
test_labels = []

# small test subset
N = 1000

count = 0

for label_folder in LABEL_DIR.iterdir():

    if count >= N:
        break

    if not label_folder.is_dir():
        continue

    # get storm id
    storm_id = label_folder.name.replace(
        "nasa_tropical_storm_competition_test_labels_",
        ""
    )

    # read label
    with open(label_folder / "labels.json") as f:
        label = json.load(f)

    wind_speed = float(label["wind_speed"])


    # matching image directory
    image_folder = (
        SOURCE_DIR /
        f"nasa_tropical_storm_competition_test_source_{storm_id}"
    )

    image_file = image_folder / "image.jpg"

    if not image_file.exists():
        continue


    # load image
    img = Image.open(image_file)

    #Convert grayscale to RGB if necessary
    img = img.convert("RGB")

    # Resize to a common size
    img = img.resize((366, 366))

    img = np.array(img)

    test_images.append(img)
    test_labels.append(wind_speed)

    count += 1

print("Loaded", len(test_images), "images")


test_images = np.stack(test_images)

test_labels = np.array(test_labels).reshape(-1,1)


print("images:", test_images.shape)
print("labels:", test_labels.shape)


features = wavelet_array_features(
    test_images,
    wavelet="haar"
)

scaler = StandardScaler()

features = scaler.fit_transform(features)

kersize_list =[]
M_prime_list = []
Nk_stats_list = []
tau_x_list = []

for N_max in [10,20,30,40,50,100,200,300,450]:

    result = feasible_sets_features_nn_xy(
        features,
        test_labels,
        N_max=N_max,
        qx=0.1,
        target_uncertainty=tc_uncertainty2,
    )

    tau_x_list.append(result["tau_x"])

    kersize, M_prime, Nk_stats = compute_kernel_size(
        result["feasible_sets"]
    )


    kersize_list.append(kersize)
    M_prime_list.append(M_prime)
    Nk_stats_list.append(Nk_stats)
    tau_x_list.append(result["tau_x"])

print("kernel size:", kersize_list)
print("M prime:", M_prime_list)
print("Nk stats:", Nk_stats_list)
print("tau_x:", tau_x_list)