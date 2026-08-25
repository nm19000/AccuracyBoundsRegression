import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from accuracy_bounds.classificationproblems.utils import wavelet_array_features
from accuracy_bounds.classificationproblems.feasible_sets import feasible_sets_classification_nn
from accuracy_bounds.classificationproblems.kersize_compute import compute_kernel_size_one_hot

from pathlib import Path
from datasets import load_dataset

p = Path(
    "/Users/ng3/Library/CloudStorage/OneDrive-OakRidgeNationalLaboratory/CodeProjects/AccuracyBounds/AccuracyBoundsRegression/AccuracyBounds/examples/RESISC45/data/.data/data"
)

hf_dataset = load_dataset(
    "imagefolder",
    data_dir=str(p),
    split="train"
)


class RESISC45Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None, n_samples=3000, seed=42):
        if n_samples < len(hf_dataset):
            hf_dataset = hf_dataset.shuffle(seed=seed).select(range(n_samples))

        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = sample["image"]      # PIL Image
        label = sample["label"]      # integer

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.ToTensor()
])

hf_split = hf_dataset.train_test_split(test_size=0.2, seed=42)

hf_train = hf_split["train"]
hf_test = hf_split["test"]

def main():

    tau_x_train_list = []
    average_kersize_train_list = []
    M_prime_train_list = []
    Nk_stats_train_list = []

    tau_x_test_list = []
    average_kersize_test_list = []
    M_prime_test_list = []
    Nk_stats_test_list = []

    for s in range(0,20):

        train_data = RESISC45Dataset(hf_train, transform=transform, n_samples=3000, seed=s)
        test_data = RESISC45Dataset(hf_test, transform=transform, n_samples=3000, seed=s)

        train_images = []
        train_labels = []

        for img, label in train_data:
            train_images.append(img.numpy())   # (3,256, 256)
            train_labels.append(label)

        train_images = np.stack(train_images)
        train_labels = np.array(train_labels)

        train_features = wavelet_array_features(train_images, wavelet="haar")

        # N_max is maximum number of neighbors to consider for feasible sets, qx is the quantile for distance threshold
        feas_train = feasible_sets_classification_nn(train_features, train_labels, N_max=200,qx=0.05)
        tau_x_train= feas_train["tau_x"]
        feasible_sets_train= feas_train["feasible_sets"]

        average_kersize_train, M_prime_train, Nk_stats_train = compute_kernel_size_one_hot(feasible_sets_train)

        tau_x_train_list.append(tau_x_train)
        average_kersize_train_list.append(average_kersize_train)
        M_prime_train_list.append(M_prime_train)
        Nk_stats_train_list.append(Nk_stats_train)

        test_images = []
        test_labels = []

        for img, label in test_data:
            test_images.append(img.numpy())   # (3,256, 256)
            test_labels.append(label)

        test_images = np.stack(test_images)
        test_labels = np.array(test_labels)


        test_features = wavelet_array_features(test_images, wavelet="haar")

        feas_test = feasible_sets_classification_nn(test_features, test_labels, N_max=200,qx=0.05)
        tau_x_test= feas_test["tau_x"]
        feasible_sets_test= feas_test["feasible_sets"]

        average_kersize_test, M_prime_test, Nk_stats_test = compute_kernel_size_one_hot(feasible_sets_test)

        tau_x_test_list.append(tau_x_test)
        average_kersize_test_list.append(average_kersize_test)
        M_prime_test_list.append(M_prime_test)
        Nk_stats_test_list.append(Nk_stats_test)

    print("tau_x_train_list", tau_x_train_list)
    print("average_kersize_train_list", average_kersize_train_list)
    print("M_prime_train_list", M_prime_train_list)
    print("Nk_stats_train_list", Nk_stats_train_list)

    print("tau_x_test_list", tau_x_test_list)
    print("average_kersize_test_list", average_kersize_test_list)
    print("M_prime_test_list", M_prime_test_list)
    print("Nk_stats_test_list", Nk_stats_test_list)


if __name__=="__main__":

    main()