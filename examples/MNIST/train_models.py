"""
Train an EfficientNet on MNIST with optional label noise. Self-contained:
no project-internal modules and no pre-generated split/noise/stats files.

All settings are fixed constants below; only the seed and the label
noise ratio (default 0, applied to train, val and test) are passed in:
    python train_mnist.py --seed 42
    python train_mnist.py --seed 42 --noise_ratio 0.2

Data handling matches the MNIST notebook: torchvision download, official
60k/10k train/test split, empty val split, Normalize(0.5, 0.5), images
(1, 28, 28). Label noise uses the notebook's add_label_noise() -- exactly
int(ratio * N) labels flipped to a different class; with the same noise
seed the noisy TRAIN labels are identical to the notebook's ablation.
Test gets the same ratio and the same noise seed.

Model selection: there is no val split, so the final model is the LAST
epoch (best_model.pt = latest_model.pt).

Evaluation: every epoch, val and test are evaluated against both the
clean and the noisy labels (accuracy, loss, confusion matrices) and stored
in epoch_stats.pkl. Test performance is only tracked -- it is never used
for model selection. The final test of the selected model is printed as a
table and plotted in final_test_results.png.

Outputs in experiments/mnist_seed<seed>_noise<ratio>/:
    train.log, metrics.json, epoch_stats.pkl, training_curves.png,
    final_test_results.png, best_model.pt, latest_model.pt

epoch_stats.pkl: epoch_stats["epochs"][epoch][split] is an
evaluate_split() dict ("against_clean_label"/"against_provided_label" ->
"accuracy", "confusion_matrix", ...); epoch 0 is the untrained model.
"""
import argparse
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST



# =====================================================================
# Fixed configuration -- only the seed and the label noise ratio are set
# from the command line
# =====================================================================
DATASET = "mnist"
DATA_ROOT = "./data/raw"
EXPERIMENTS_DIR = "./experiments"
TRAIN_SUBSET_SIZE = None        # None = all 60k train images (as in the MNIST notebook)
NOISE_SEED = None               # None = use the run seed
MODEL_NAME = "efficientnet_b0"  # any timm model name
PRETRAINED = False
IMG_SIZE = 28
NATIVE_SIZE = 28
IN_CHANS = 1
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY_N_STEPS = 20


# =====================================================================
# Utilities
# =====================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(run_dir):
    logger = logging.getLogger(f"train.{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def fmt_num(value):
    return "n/a" if value is None else f"{value:.4f}"


def add_label_noise(labels, frac, num_classes=10, seed=42):
    """Flip exactly int(frac * N) labels to a different, uniformly random class."""
    rng = np.random.default_rng(seed)
    y = np.array(labels)
    idx = rng.choice(len(y), int(frac * len(y)), replace=False)
    y[idx] = (y[idx] + rng.integers(1, num_classes, len(idx))) % num_classes
    return y, idx


# =====================================================================
# Data
# =====================================================================
def build_mnist_split_data(data_root, train_subset_size=None, seed=42):
    """Same as the MNIST notebook: uint8 images (N, 28, 28) + labels per split."""
    data_path = Path(data_root) / "MNIST"
    data_path.mkdir(parents=True, exist_ok=True)
    train_raw = MNIST(data_path, train=True, download=True)
    test_raw = MNIST(data_path, train=False, download=True)

    train_x, train_y = train_raw.data, train_raw.targets
    if train_subset_size is not None:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(train_x), train_subset_size, replace=False))
        train_x, train_y = train_x[idx], train_y[idx]

    return {
        "classes": [str(i) for i in range(10)],
        "train": (train_x, train_y.tolist()),
        "val": (train_x[:0], []),
        "test": (test_raw.data, test_raw.targets.tolist()),
    }


class NoisyLabelDataset(Dataset):
    """In-memory uint8 MNIST images plus clean and (possibly noisy) provided labels.
    Returns dict batches: image, label (provided), clean_label, is_noisy."""

    def __init__(self, images, clean_labels, noisy_labels=None, transform=None):
        self.images = images
        self.clean_labels = np.asarray(clean_labels, dtype=np.int64)
        self.labels = (self.clean_labels.copy() if noisy_labels is None
                       else np.asarray(noisy_labels, dtype=np.int64))
        self.is_noisy = self.labels != self.clean_labels
        self.transform = transform

    def __len__(self):
        return len(self.clean_labels)

    def _load(self, idx):
        return self.images[idx].unsqueeze(0).float() / 255.0  # (1, 28, 28) in [0, 1]

    def __getitem__(self, idx):
        img = self._load(idx)
        if self.transform:
            img = self.transform(img)
        return {
            "image": img,
            "label": int(self.labels[idx]),
            "clean_label": int(self.clean_labels[idx]),
            "is_noisy": bool(self.is_noisy[idx]),
        }


def get_transforms(in_chans, img_size, native_size, train):
    ops = []
    if img_size != native_size:
        ops.append(T.Resize((img_size, img_size), antialias=True))
    # no flips: they change the meaning of digits
    ops.append(T.Normalize(mean=[0.5] * in_chans, std=[0.5] * in_chans))
    return T.Compose(ops)


# =====================================================================
# Evaluation
# =====================================================================
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    preds, provided, clean, is_noisy, loss_p, loss_c = [], [], [], [], [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        y = batch["label"].to(device)
        y_clean = batch["clean_label"].to(device)
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu().numpy())
        provided.append(y.cpu().numpy())
        clean.append(y_clean.cpu().numpy())
        is_noisy.append(batch["is_noisy"].numpy())
        loss_p.append(F.cross_entropy(logits, y, reduction="none").cpu().numpy())
        loss_c.append(F.cross_entropy(logits, y_clean, reduction="none").cpu().numpy())
    return tuple(np.concatenate(a) for a in (preds, provided, clean, is_noisy, loss_p, loss_c))


def _mean_or_none(a):
    return float(a.mean()) if len(a) else None


def compute_metrics(preds, labels, class_names):
    k = list(range(len(class_names)))
    return {
        "accuracy": float((preds == labels).mean()),
        "confusion_matrix": confusion_matrix(labels, preds, labels=k).tolist(),
        "classification_report": classification_report(
            labels, preds, labels=k, target_names=class_names, output_dict=True, zero_division=0),
    }


def compute_breakdown_by_noise(preds, clean, provided, is_noisy):
    out = {}
    for name, m in (("clean_subset", ~is_noisy), ("noisy_subset", is_noisy)):
        out[name] = {"count": int(m.sum()),
                     "acc_vs_clean_label": _mean_or_none(preds[m] == clean[m]),
                     "acc_vs_provided_label": _mean_or_none(preds[m] == provided[m])}
    return out


def compute_loss_breakdown(loss_clean, loss_provided, is_noisy):
    out = {}
    for name, m in (("clean_subset", ~is_noisy), ("noisy_subset", is_noisy)):
        out[name] = {"count": int(m.sum()),
                     "loss_vs_clean_label": _mean_or_none(loss_clean[m]),
                     "loss_vs_provided_label": _mean_or_none(loss_provided[m])}
    return out


def summarize(preds, provided, clean, is_noisy, loss_p, loss_c, class_names):
    return {
        "against_provided_label": compute_metrics(preds, provided, class_names),
        "against_clean_label": compute_metrics(preds, clean, class_names),
        "noisy_fraction": float(is_noisy.mean()),
        "noise_breakdown": compute_breakdown_by_noise(preds, clean, provided, is_noisy),
        "loss_vs_provided_label": float(loss_p.mean()),
        "loss_vs_clean_label": float(loss_c.mean()),
        "loss_breakdown": compute_loss_breakdown(loss_c, loss_p, is_noisy),
    }


def empty_summary(class_names):
    n = len(class_names)
    empty_metrics = {"accuracy": None, "confusion_matrix": [[0] * n for _ in range(n)],
                     "classification_report": {}}
    acc_sub = {"count": 0, "acc_vs_clean_label": None, "acc_vs_provided_label": None}
    loss_sub = {"count": 0, "loss_vs_clean_label": None, "loss_vs_provided_label": None}
    return {
        "against_provided_label": dict(empty_metrics),
        "against_clean_label": dict(empty_metrics),
        "noisy_fraction": None,
        "noise_breakdown": {"clean_subset": dict(acc_sub), "noisy_subset": dict(acc_sub)},
        "loss_vs_provided_label": None,
        "loss_vs_clean_label": None,
        "loss_breakdown": {"clean_subset": dict(loss_sub), "noisy_subset": dict(loss_sub)},
    }


def evaluate_split(model, loader, class_names, device):
    if len(loader.dataset) == 0:
        return empty_summary(class_names)
    return summarize(*run_inference(model, loader, device), class_names)


def evaluate_epoch_snapshot(model, loaders, class_names, device):
    """Full evaluate_split() over train (eval transform), val and test."""
    return {split: evaluate_split(model, loaders[split], class_names, device)
            for split in ("train", "val", "test")}


# =====================================================================
# Plots
# =====================================================================
def plot_training_curves(epoch_stats, path):
    epochs = sorted(epoch_stats["epochs"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"train": "#7209b7", "val": "#ef8a17", "test": "#118ab2"}
    for split, c in colors.items():
        for key, ls, tag in (("against_clean_label", "-", "clean"),
                             ("against_provided_label", "--", "provided")):
            acc = [epoch_stats["epochs"][e][split][key]["accuracy"] for e in epochs]
            if all(a is None for a in acc):
                continue
            ax[0].plot(epochs, [np.nan if a is None else a for a in acc], ls, color=c,
                       marker="o", ms=3, label=f"{split} (vs {tag})")
        for key, ls, tag in (("loss_vs_clean_label", "-", "clean"),
                             ("loss_vs_provided_label", "--", "provided")):
            loss = [epoch_stats["epochs"][e][split][key] for e in epochs]
            if all(l is None for l in loss):
                continue
            ax[1].plot(epochs, [np.nan if l is None else l for l in loss], ls, color=c,
                       marker="o", ms=3, label=f"{split} (vs {tag})")
    ax[0].set_ylabel("accuracy")
    ax[0].set_ylim(0, 1.02)
    ax[1].set_ylabel("loss")
    for a in ax:
        a.set_xlabel("epoch (0 = before training)")
        a.legend(fontsize=8)
    fig.suptitle(epoch_stats["run_name"])
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def curve_lists(epoch_stats):
    """(epoch, value) tuples for every combination of split (train, test),
    labels (clean, noisy) and metric (loss, accuracy). Taken from the
    per-epoch snapshots, so train uses the eval transform and epoch 0 is the
    untrained model."""
    epochs = sorted(epoch_stats["epochs"])
    keys = {("clean", "loss"): ("loss_vs_clean_label",),
            ("noisy", "loss"): ("loss_vs_provided_label",),
            ("clean", "accuracy"): ("against_clean_label", "accuracy"),
            ("noisy", "accuracy"): ("against_provided_label", "accuracy")}
    lists = {}
    for split in ("train", "test"):
        for (labels, metric), path in keys.items():
            vals = []
            for e in epochs:
                v = epoch_stats["epochs"][e][split]
                for k in path:
                    v = v[k]
                vals.append((e, None if v is None else round(float(v), 4)))
            lists[f"{split}_{labels}_{metric}"] = vals
    return lists


def final_test_table(test_results, epoch):
    """Text table of the final test results vs. clean and noisy labels."""
    tb, lb = test_results["noise_breakdown"], test_results["loss_breakdown"]
    rows = [
        ("all test samples", sum(v["count"] for v in tb.values()),
         test_results["against_clean_label"]["accuracy"], test_results["against_provided_label"]["accuracy"],
         test_results["loss_vs_clean_label"], test_results["loss_vs_provided_label"]),
        ("  label unchanged", tb["clean_subset"]["count"],
         tb["clean_subset"]["acc_vs_clean_label"], tb["clean_subset"]["acc_vs_provided_label"],
         lb["clean_subset"]["loss_vs_clean_label"], lb["clean_subset"]["loss_vs_provided_label"]),
        ("  label flipped", tb["noisy_subset"]["count"],
         tb["noisy_subset"]["acc_vs_clean_label"], tb["noisy_subset"]["acc_vs_provided_label"],
         lb["noisy_subset"]["loss_vs_clean_label"], lb["noisy_subset"]["loss_vs_provided_label"]),
    ]
    lines = [f"FINAL TEST RESULTS (model from epoch {epoch})",
             f"{'':18s} {'n':>6s} | {'acc clean':>9s} {'acc noisy':>9s} | {'loss clean':>10s} {'loss noisy':>10s}"]
    for name, n, ac, an, lc, ln in rows:
        lines.append(f"{name:18s} {n:6d} | {fmt_num(ac):>9s} {fmt_num(an):>9s} | {fmt_num(lc):>10s} {fmt_num(ln):>10s}")
    return lines


def plot_final_test(test_results, class_names, path, title):
    """Accuracy/loss vs. clean and noisy test labels + both confusion matrices."""
    fig, ax = plt.subplots(1, 4, figsize=(22, 5), gridspec_kw={"width_ratios": [1, 1, 1.4, 1.4]})
    keys = ["vs clean labels", "vs noisy labels"]
    colors = ["#118ab2", "#ef8a17"]

    accs = [test_results["against_clean_label"]["accuracy"], test_results["against_provided_label"]["accuracy"]]
    losses = [test_results["loss_vs_clean_label"], test_results["loss_vs_provided_label"]]
    for a, vals, name in ((ax[0], accs, "accuracy"), (ax[1], losses, "loss")):
        bars = a.bar(keys, vals, color=colors)
        a.bar_label(bars, labels=[f"{v:.4f}" for v in vals], padding=3)
        a.set_title(f"Test {name}")
        a.set_ylim(0, 1.1 if name == "accuracy" else max(vals) * 1.2)
        a.spines[["top", "right"]].set_visible(False)

    for a, key, name in ((ax[2], "against_clean_label", "clean"), (ax[3], "against_provided_label", "noisy")):
        cm = np.array(test_results[key]["confusion_matrix"])
        a.imshow(cm, cmap="Blues")
        k = len(class_names)
        a.set_xticks(range(k), class_names, rotation=45, ha="right", fontsize=8)
        a.set_yticks(range(k), class_names, fontsize=8)
        for i in range(k):
            for j in range(k):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_xlabel("predicted")
        a.set_ylabel(f"{name} label")
        a.set_title(f"Confusion matrix vs {name} labels")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =====================================================================
# Training
# =====================================================================
def build_args(seed, noise_ratio=0.0):
    """Fixed configuration as a namespace; only the seed and the train noise ratio vary."""
    args = argparse.Namespace(
        dataset=DATASET, data_root=DATA_ROOT, experiments_dir=EXPERIMENTS_DIR, seed=seed,
        train_subset_size=TRAIN_SUBSET_SIZE,
        train_noise_ratio=noise_ratio, test_noise_ratio=noise_ratio,
        noise_seed=NOISE_SEED, model_name=MODEL_NAME, pretrained=PRETRAINED,
        img_size=IMG_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY, num_workers=NUM_WORKERS, device=DEVICE,
        log_every_n_steps=LOG_EVERY_N_STEPS,
    )
    args.run_name = f"mnist_seed{seed}_noise{noise_ratio:g}"
    return args


def run(args):
    native_size, in_chans = NATIVE_SIZE, IN_CHANS
    noise_seed = args.seed if args.noise_seed is None else args.noise_seed

    run_dir = Path(args.experiments_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(run_dir)
    logger.info(f"Starting run '{args.run_name}' (dataset={args.dataset})")
    logger.info(f"Config: {json.dumps(vars(args), indent=2, default=str)}")

    seed_everything(args.seed)

    # ---------------- data ----------------
    split_data = build_mnist_split_data(args.data_root, args.train_subset_size, args.seed)
    class_names = split_data["classes"]
    num_classes = len(class_names)

    sizes = {s: len(split_data[s][1]) for s in ("train", "val", "test")}
    logger.info(f"Split sizes: {sizes}")
    val_is_empty = sizes["val"] == 0
    if val_is_empty:
        logger.info("Val split is EMPTY -- val is never evaluated; the final model is the "
                    "last epoch (test is never used for model selection).")

    # label noise: train uses noise_seed exactly as in the notebooks;
    # all splits use the same noise seed, so each split's noisy labels match
    # add_label_noise(labels, ratio, seed=noise_seed) in the notebooks
    noise_cfg = {"train": (args.train_noise_ratio, noise_seed),
                 "val": (args.train_noise_ratio, noise_seed),
                 "test": (args.test_noise_ratio, noise_seed)}
    noisy = {}
    for split, (ratio, s) in noise_cfg.items():
        labels = split_data[split][1]
        noisy[split] = add_label_noise(labels, ratio, num_classes, seed=s)[0] if ratio > 0 and labels else None
        if noisy[split] is not None:
            n_flip = int((noisy[split] != np.asarray(labels)).sum())
            logger.info(f"{split} label noise: {n_flip}/{len(labels)} labels flipped (ratio={ratio})")
        else:
            logger.info(f"No {split} label noise")

    train_tf = get_transforms(in_chans, args.img_size, native_size, train=True)
    eval_tf = get_transforms(in_chans, args.img_size, native_size, train=False)

    def make_ds(split, tf):
        images, labels = split_data[split]
        return NoisyLabelDataset(images, labels, noisy[split], tf)

    datasets = {"train_aug": make_ds("train", train_tf),
                "train": make_ds("train", eval_tf),     # same data, no augmentation, for stats
                "val": make_ds("val", eval_tf),
                "test": make_ds("test", eval_tf)}

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(datasets["train_aug"], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, generator=g)
    loaders = {s: DataLoader(datasets[s], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers) for s in ("train", "val", "test")}

    # ---------------- model ----------------
    logger.info(f"Building model '{args.model_name}' (pretrained={args.pretrained}, "
                f"in_chans={in_chans}, num_classes={num_classes})")
    model = timm.create_model(args.model_name, pretrained=args.pretrained,
                              num_classes=num_classes, in_chans=in_chans).to(args.device)
    logger.info(f"Device: {args.device}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    criterion = nn.CrossEntropyLoss(reduction="none")

    best_ckpt_path = run_dir / "best_model.pt"
    best_val_loss_ckpt_path = run_dir / "best_model_val_loss.pt"
    latest_ckpt_path = run_dir / "latest_model.pt"
    best_val_acc, best_val_loss = -1.0, float("inf")
    best_val_loss_epoch = None

    epoch_stats = {"run_name": args.run_name, "dataset": args.dataset, "classes": class_names,
                   "config": vars(args), "epochs": {}}
    logger.info("Snapshotting epoch 0 (untrained model) train/val/test stats ...")
    epoch_stats["epochs"][0] = evaluate_epoch_snapshot(model, loaders, class_names, args.device)

    history = []
    noise_active = args.train_noise_ratio > 0
    n_batches = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        ep_preds, ep_provided, ep_clean, ep_noisy, ep_loss_p, ep_loss_c = [], [], [], [], [], []
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            imgs = batch["image"].to(args.device)
            labels = batch["label"].to(args.device)
            clean_labels = batch["clean_label"].to(args.device)

            opt.zero_grad()
            logits = model(imgs)
            loss_p = criterion(logits, labels)
            loss = loss_p.mean()
            loss.backward()
            opt.step()

            with torch.no_grad():
                loss_c = F.cross_entropy(logits, clean_labels, reduction="none")
            preds = logits.argmax(1).detach()
            running_loss += loss.item() * imgs.size(0)
            running_correct += (preds == labels).sum().item()
            total += imgs.size(0)

            ep_preds.append(preds.cpu().numpy())
            ep_provided.append(labels.cpu().numpy())
            ep_clean.append(clean_labels.cpu().numpy())
            ep_noisy.append(batch["is_noisy"].numpy())
            ep_loss_p.append(loss_p.detach().cpu().numpy())
            ep_loss_c.append(loss_c.cpu().numpy())

            if args.log_every_n_steps and step % args.log_every_n_steps == 0:
                logger.info(f"epoch {epoch}/{args.epochs} step {step}/{n_batches} "
                            f"running_loss={running_loss / total:.4f} "
                            f"running_acc(vs provided)={running_correct / total:.4f} "
                            f"lr={opt.param_groups[0]['lr']:.2e}")
        sched.step()

        # running (augmented, mid-epoch) train stats
        tr_preds, tr_provided, tr_clean, tr_noisy, tr_loss_p, tr_loss_c = (
            np.concatenate(a) for a in (ep_preds, ep_provided, ep_clean, ep_noisy, ep_loss_p, ep_loss_c))
        train_loss = float(tr_loss_p.mean())
        train_loss_clean = float(tr_loss_c.mean())
        train_acc_provided = float((tr_preds == tr_provided).mean())
        train_acc_clean = float((tr_preds == tr_clean).mean())
        train_breakdown = compute_breakdown_by_noise(tr_preds, tr_clean, tr_provided, tr_noisy)
        train_loss_breakdown = compute_loss_breakdown(tr_loss_c, tr_loss_p, tr_noisy)

        # full train(eval tf)/val/test snapshot -- also provides val/test metrics
        snap = evaluate_epoch_snapshot(model, loaders, class_names, args.device)
        epoch_stats["epochs"][epoch] = snap
        val, test = snap["val"], snap["test"]
        val_acc = val["against_provided_label"]["accuracy"]
        val_loss = val["loss_vs_provided_label"]
        test_acc = test["against_provided_label"]["accuracy"]
        test_loss = test["loss_vs_provided_label"]
        epoch_time = time.time() - t0

        log_parts = [f"epoch {epoch}/{args.epochs} COMPLETE",
                     f"train_loss={train_loss:.4f} train_acc(vs provided)={train_acc_provided:.4f}"]
        if noise_active:
            log_parts.append(f"train_acc(vs clean)={train_acc_clean:.4f} "
                             f"noisy-subset acc vs true={fmt_num(train_breakdown['noisy_subset']['acc_vs_clean_label'])} "
                             f"vs given={fmt_num(train_breakdown['noisy_subset']['acc_vs_provided_label'])} "
                             f"(n={train_breakdown['noisy_subset']['count']})")
        if val_is_empty:
            log_parts.append("val n/a (empty split)")
        else:
            log_parts.append(f"val acc vs clean={fmt_num(val['against_clean_label']['accuracy'])} "
                             f"vs noisy={fmt_num(val_acc)} | "
                             f"val loss vs clean={fmt_num(val['loss_vs_clean_label'])} "
                             f"vs noisy={fmt_num(val_loss)}")
        log_parts.append(f"time={epoch_time:.1f}s")
        logger.info(" | ".join(log_parts))

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_clean": train_loss_clean,
            "train_loss_breakdown": train_loss_breakdown,
            "train_acc_vs_provided": train_acc_provided,
            "train_acc_vs_clean": train_acc_clean,
            "train_breakdown": train_breakdown,
            "val_loss": val_loss,
            "val_loss_clean": val["loss_vs_clean_label"],
            "val_loss_breakdown": val["loss_breakdown"],
            "val_acc": val_acc,
            "val_breakdown": val["noise_breakdown"],
            "val_acc_clean": val["against_clean_label"]["accuracy"],
            "test_loss": test_loss,
            "test_loss_clean": test["loss_vs_clean_label"],
            "test_acc": test_acc,
            "test_acc_clean": test["against_clean_label"]["accuracy"],
            "time_sec": epoch_time,
        })

        # ---------------- checkpoints ----------------
        state = model.state_dict()
        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": state, "epoch": epoch, "val_acc": val_acc}, best_ckpt_path)
            logger.info(f"New best model (val_acc={val_acc:.4f}) saved to {best_ckpt_path}")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss, best_val_loss_epoch = val_loss, epoch
            torch.save({"model_state": state, "epoch": epoch, "val_loss": val_loss}, best_val_loss_ckpt_path)
            logger.info(f"New best model by val loss ({val_loss:.4f}) saved to {best_val_loss_ckpt_path}")
        torch.save({"model_state": state, "epoch": epoch}, latest_ckpt_path)

    # ---------------- final evaluation ----------------
    if args.epochs == 0:
        logger.info("No training epochs were run (--epochs 0); evaluating the untrained model.")
        best_epoch = 0
    else:
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=args.device)
            logger.info(f"Final model: best-by-val-accuracy checkpoint (epoch {ckpt['epoch']})")
        else:
            # no val split -> no model selection; never select on the test set
            ckpt = torch.load(latest_ckpt_path, map_location=args.device)
            torch.save(ckpt, best_ckpt_path)
            logger.info(f"Final model: last epoch ({ckpt['epoch']}) -- no val split for model selection")
        model.load_state_dict(ckpt["model_state"])
        best_epoch = ckpt["epoch"]
    test_results = evaluate_split(model, loaders["test"], class_names, args.device)

    metrics = {
        "run_name": args.run_name,
        "config": vars(args),
        "dataset": args.dataset,
        "classes": class_names,
        "split_sizes": sizes,
        "history": history,
        "best_val_acc": best_val_acc if best_val_acc > -1.0 else None,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if best_val_loss_epoch is not None else None,
        "best_val_loss_epoch": best_val_loss_epoch,
        "test_results": test_results,
        "curves": curve_lists(epoch_stats),
        "train_noise_ratio": args.train_noise_ratio,
        "test_noise_ratio": args.test_noise_ratio,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(run_dir / "epoch_stats.pkl", "wb") as f:
        pickle.dump(epoch_stats, f)
    plot_training_curves(epoch_stats, run_dir / "training_curves.png")
    plot_final_test(test_results, class_names, run_dir / "final_test_results.png",
                    f"{args.run_name} -- final test (epoch {best_epoch})")
    logger.info(f"Saved metrics.json, epoch_stats.pkl, training_curves.png and "
                f"final_test_results.png to {run_dir}")

    for line in final_test_table(test_results, best_epoch):
        logger.info(line)

    logger.info("Per-epoch curves as (epoch, value) tuples (epoch 0 = untrained model):")
    for name, values in curve_lists(epoch_stats).items():
        logger.info(f"{name} = {values}")
    logger.info(f"Run '{args.run_name}' complete. See {run_dir}")
    return metrics


def main():
    p = argparse.ArgumentParser(description="Train an EfficientNet on MNIST (all settings fixed except seed and noise ratio)")
    p.add_argument("--seed", type=int, default=42,
                   help="model init, data loader shuffling, MNIST train subset and (by default) label noise")
    p.add_argument("--noise_ratio", type=float, default=0.0,
                   help="fraction of labels flipped to a wrong class, applied to train, val "
                        "and test (same noise seed); default 0 = clean labels")
    a = p.parse_args()
    if not 0.0 <= a.noise_ratio <= 1.0:
        p.error("--noise_ratio must be in [0, 1]")
    run(build_args(a.seed, a.noise_ratio))


if __name__ == "__main__":
    main()