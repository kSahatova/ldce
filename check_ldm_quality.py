"""
check_ldm_quality.py
====================
Evaluate the generation quality of a trained LDM against a real dataset
using a classification model as the quality signal.

Produces
--------
  1. Classifier accuracy on class-conditional generated samples (overall + per-class)
  2. Classifier confidence distribution: real vs generated images
  3. FID score between generated and real test images (requires pytorch-fid)
  4. Pixel statistics per generated class
  5. Visual grids of generated samples (one row per class)

The script can be driven either via individual CLI flags or by pointing to
an existing LDCE run-config YAML (e.g. assets/configs/config_ldce_mnist.yaml)
which already contains paths for the LDM, classifier, and dataset.

Usage
-----
    # Via individual flags
    python check_ldm_quality.py \\
        --ldm_cfg   assets/configs/ldm_mnist.yaml \\
        --ldm_ckpt  assets/checkpoints/mnist_ldm/mnist_ldm_best.ckpt \\
        --clf_ckpt  assets/checkpoints/mnist_classifier/classifier_mnist.pth \\
        --dataset   fmnist \\
        --classes   0 2 4 6 \\
        --n_per_class 20 \\
        --output_dir results/ldm_quality

    # Via an existing LDCE run-config YAML
    python check_ldm_quality.py \\
        --run_cfg assets/configs/config_ldce_mnist.yaml \\
        --output_dir results/ldm_quality
"""

import argparse
import contextlib
import importlib.abc
import importlib.machinery
import os
import sys
import shutil
import types

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "taming-transformers"))

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.classifiers import CNNtorch, SimpleCNNtorch


@contextlib.contextmanager
def _src_stub_context():
    """Stub 'src.*' imports so torch.load can unpickle checkpoints saved from a
    codebase where classifier classes lived under src/."""

    class _AnyObj:
        def __setstate__(self, state: dict) -> None:
            self.__dict__.update(state if isinstance(state, dict) else {})

    class _SrcStub(types.ModuleType):
        def __getattr__(self, _name: str):
            return _AnyObj

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "src" or fullname.startswith("src."):
                return importlib.machinery.ModuleSpec(fullname, self)

        def create_module(self, spec):
            return _SrcStub(spec.name)

        def exec_module(self, module):
            sys.modules[module.__name__] = module

    finder = _Finder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
    HAS_FID = True
except ImportError:
    HAS_FID = False


# ── Dataset helpers ───────────────────────────────────────────────────────────

_DATASET_REGISTRY = {
    "mnist":  datasets.MNIST,
    "fmnist": datasets.FashionMNIST,
}

_DATASET_NAMES = {
    "mnist":  "MNIST",
    "fmnist": "FashionMNIST",
}


def build_real_dataset(
    dataset_name: str,
    image_size: int,
    data_root: str,
    classes: list,
    max_samples: int | None = None,
) -> torch.utils.data.Dataset:
    """Return a filtered test-set dataset for the given classes.

    Each sample is returned as a ``(3, H, W)`` float32 tensor in ``[0, 1]``.
    """
    ds_cls = _DATASET_REGISTRY[dataset_name.lower()]
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    raw = ds_cls(root=data_root, train=False, download=True, transform=tf)

    # Filter to requested classes
    targets = torch.tensor(raw.targets) if hasattr(raw, "targets") else torch.tensor([t for _, t in raw])
    indices = [i for i, t in enumerate(targets) if t.item() in classes]

    if max_samples is not None:
        indices = indices[:max_samples]

    subset = Subset(raw, indices)

    # Wrap to repeat greyscale → 3-channel
    class RGBWrapper(torch.utils.data.Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, label = self.base[idx]
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            return img, label

    return RGBWrapper(subset)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_ldm(cfg_path: str, ckpt_path: str, device: str):
    config = OmegaConf.load(cfg_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [LDM] Missing keys  : {len(missing)}")
    if unexpected:
        print(f"  [LDM] Unexpected keys: {len(unexpected)}")
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded LDM  ({n_params:.1f} M params) from {ckpt_path}")
    return model


def load_classifier(ckpt_path: str, clf_args: dict, device: str):
    """Load a CNNtorch or SimpleCNNtorch classifier from a checkpoint."""
    # Determine class from presence of 'in_conv_channels' key
    if "in_conv_channels" in clf_args:
        model = SimpleCNNtorch(**clf_args)
    else:
        model = CNNtorch(**clf_args)

    # Use src stub so torch.load can unpickle checkpoints whose classes
    # were originally stored under a 'src' package.
    with _src_stub_context():
        sd = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = sd.get("state_dict", sd)
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded CLF  ({n_params:.3f} M params) from {ckpt_path}")
    return model


# ── LDM sampling ──────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(
    ldm_model,
    classes: list,
    n_per_class: int,
    ddim_steps: int,
    guidance_scale: float,
    ddim_eta: float,
    null_class: int,
    device: str,
) -> tuple[torch.Tensor, list[int]]:
    """Generate ``n_per_class`` samples for each class in ``classes``.

    Returns
    -------
    samples : (N, 3, H, W) float tensor in [0, 1]
    labels  : list of int length N — true class index for each sample
    """
    ldm_model.eval()
    sampler = DDIMSampler(ldm_model)

    all_samples = []
    all_labels = []

    shape = [ldm_model.channels, ldm_model.image_size, ldm_model.image_size]

    # Null/unconditional conditioning for CFG
    null_tensor = torch.full((n_per_class,), null_class, dtype=torch.long, device=device)
    with ldm_model.ema_scope():
        uc = ldm_model.get_learned_conditioning({"class_label": null_tensor})

    for cls in classes:
        cls_tensor = torch.full((n_per_class,), cls, dtype=torch.long, device=device)
        with ldm_model.ema_scope():
            c = ldm_model.get_learned_conditioning({"class_label": cls_tensor})
            z, _ = sampler.sample(
                S=ddim_steps,
                batch_size=n_per_class,
                shape=shape,
                conditioning=c,
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=uc,
                verbose=False,
                eta=ddim_eta,
            )
        x = ldm_model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0).cpu()
        all_samples.append(x)
        all_labels.extend([cls] * n_per_class)

    return torch.cat(all_samples, dim=0), all_labels


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def classifier_accuracy(
    clf,
    samples: torch.Tensor,
    labels: list[int],
    clf_input_size: int,
    clf_input_channels: int,
    class_map: dict | None,
    device: str,
) -> dict:
    """Run classifier on generated samples and return accuracy statistics.

    Parameters
    ----------
    class_map : optional dict mapping classifier output index → dataset class index.
                If None, classifier indices are assumed to match dataset classes directly.
    """
    clf.eval()
    correct = 0
    total = 0
    classes = sorted(set(labels))
    per_class_correct = {c: 0 for c in classes}
    per_class_total = {c: 0 for c in classes}
    all_probs = []

    for i in range(samples.shape[0]):
        img = samples[i:i+1].to(device)
        img = F.interpolate(img, size=(clf_input_size, clf_input_size),
                            mode="bilinear", align_corners=False)
        img = img[:, :clf_input_channels]

        logits = clf(img)
        probs = F.softmax(logits, dim=1) if not clf.activate_softmax else logits.exp()
        all_probs.append(probs.cpu())

        pred_idx = logits.argmax(dim=1).item()
        # Map classifier index → dataset class index if needed
        pred_cls = class_map[pred_idx] if class_map else pred_idx
        true_cls = labels[i]

        per_class_total[true_cls] += 1
        if pred_cls == true_cls:
            correct += 1
            per_class_correct[true_cls] += 1
        total += 1

    overall_acc = correct / total if total > 0 else 0.0
    per_class_acc = {
        c: per_class_correct[c] / per_class_total[c]
        for c in classes if per_class_total[c] > 0
    }
    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "all_probs": torch.cat(all_probs, dim=0),
        "correct": correct,
        "total": total,
    }


@torch.no_grad()
def real_classifier_accuracy(
    clf,
    dataloader: DataLoader,
    clf_input_size: int,
    clf_input_channels: int,
    class_map: dict | None,
    device: str,
) -> dict:
    """Compute classifier accuracy on the real test dataset for comparison."""
    clf.eval()
    correct = 0
    total = 0
    classes_seen = set()
    per_class_correct = {}
    per_class_total = {}
    all_probs = []

    for imgs, labels_batch in dataloader:
        imgs = imgs.to(device)
        imgs = F.interpolate(imgs, size=(clf_input_size, clf_input_size),
                             mode="bilinear", align_corners=False)
        imgs = imgs[:, :clf_input_channels]

        logits = clf(imgs)
        probs = F.softmax(logits, dim=1) if not clf.activate_softmax else logits.exp()
        all_probs.append(probs.cpu())
        preds = logits.argmax(dim=1)

        for pred_idx, true_lbl in zip(preds.tolist(), labels_batch.tolist()):
            pred_cls = class_map[pred_idx] if class_map else pred_idx
            classes_seen.add(true_lbl)
            per_class_total[true_lbl] = per_class_total.get(true_lbl, 0) + 1
            per_class_correct[true_lbl] = per_class_correct.get(true_lbl, 0)
            if pred_cls == true_lbl:
                correct += 1
                per_class_correct[true_lbl] += 1
            total += 1

    overall_acc = correct / total if total > 0 else 0.0
    per_class_acc = {
        c: per_class_correct[c] / per_class_total[c]
        for c in classes_seen if per_class_total.get(c, 0) > 0
    }
    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "all_probs": torch.cat(all_probs, dim=0),
        "correct": correct,
        "total": total,
    }


def pixel_stats_per_class(samples: torch.Tensor, labels: list[int]) -> dict:
    classes = sorted(set(labels))
    stats = {}
    for c in classes:
        mask = [i for i, l in enumerate(labels) if l == c]
        s = samples[mask]
        stats[c] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "non_degenerate": bool(s.view(len(mask), -1).std(dim=1).gt(0.01).all()),
        }
    return stats


def compute_fid(
    real_dataset: torch.utils.data.Dataset,
    generated_samples: torch.Tensor,
    output_dir: str,
    device: str,
    batch_size: int = 64,
) -> float | None:
    if not HAS_FID:
        print("  Skipping FID — install pytorch-fid: pip install pytorch-fid")
        return None

    real_dir = os.path.join(output_dir, "_fid_real")
    gen_dir = os.path.join(output_dir, "_fid_gen")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)

    # Save real images
    for i, (img, _) in enumerate(real_dataset):
        save_image(img, os.path.join(real_dir, f"{i:06d}.png"))

    # Save generated images
    for i in range(generated_samples.shape[0]):
        save_image(generated_samples[i], os.path.join(gen_dir, f"{i:06d}.png"))

    try:
        fid = calculate_fid_given_paths(
            [real_dir, gen_dir],
            batch_size=batch_size,
            device=device,
            dims=2048,
        )
    except Exception as exc:
        print(f"  FID computation failed: {exc}")
        fid = None
    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(gen_dir, ignore_errors=True)

    return fid


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_sample_grid(samples: torch.Tensor, labels: list[int], classes: list, n_per_row: int, path: str):
    rows = []
    for c in classes:
        idx = [i for i, l in enumerate(labels) if l == c][:n_per_row]
        rows.append(samples[idx])
    grid_tensor = torch.cat(rows, dim=0)
    grid = make_grid(grid_tensor, nrow=n_per_row, padding=2, pad_value=0.5)
    save_image(grid, path)


def plot_confidence_comparison(gen_probs: torch.Tensor, real_probs: torch.Tensor,
                                classes: list, output_path: str):
    """Plot max-confidence histograms for generated vs real images."""
    gen_conf = gen_probs.max(dim=1).values.numpy()
    real_conf = real_probs.max(dim=1).values.numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(real_conf, bins=40, alpha=0.6, label="Real images", color="steelblue", density=True)
    ax.hist(gen_conf, bins=40, alpha=0.6, label="Generated images", color="darkorange", density=True)
    ax.set_xlabel("Max classifier confidence")
    ax.set_ylabel("Density")
    ax.set_title("Classifier Confidence: Real vs Generated")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(real_acc: dict, gen_acc: dict, classes: list, output_path: str):
    x = np.arange(len(classes))
    width = 0.35
    real_vals = [real_acc.get(c, 0.0) * 100 for c in classes]
    gen_vals = [gen_acc.get(c, 0.0) * 100 for c in classes]

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.2), 4))
    ax.bar(x - width / 2, real_vals, width, label="Real images", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, gen_vals, width, label="Generated images", color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Per-class Classifier Accuracy: Real vs Generated")
    ax.legend()
    for xi, (rv, gv) in enumerate(zip(real_vals, gen_vals)):
        ax.text(xi - width / 2, rv + 1, f"{rv:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + width / 2, gv + 1, f"{gv:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Config resolution helpers ─────────────────────────────────────────────────

def resolve_config(args) -> dict:
    """Merge CLI args and optional run-config YAML into a single settings dict."""
    settings = {}

    if args.run_cfg:
        run_cfg = OmegaConf.load(args.run_cfg)
        run_cfg_dir = os.path.dirname(os.path.abspath(args.run_cfg))

        def _abs(p):
            return p if os.path.isabs(p) else os.path.join(run_cfg_dir, p)

        settings["ldm_cfg"] = _abs(run_cfg.diffusion_model.cfg_path)
        settings["ldm_ckpt"] = _abs(run_cfg.diffusion_model.ckpt_path)
        settings["clf_ckpt"] = _abs(run_cfg.classifier_model.ckpt_path)
        settings["clf_args"] = OmegaConf.to_container(run_cfg.classifier_model.args, resolve=True)
        settings["clf_input_size"] = run_cfg.classifier_model.get("input_size", 28)
        raw_map = OmegaConf.to_container(run_cfg.classifier_model.get("class_map", {}), resolve=True)
        settings["class_map"] = {int(k): int(v) for k, v in raw_map.items()} if raw_map else None
        settings["classes"] = list(run_cfg.data.get("classes", list(range(10))))
        settings["data_root"] = _abs(run_cfg.data.root)
        settings["image_size"] = run_cfg.data.image_size
        settings["dataset"] = run_cfg.get("dataset", "mnist")
        settings["ddim_steps"] = run_cfg.get("ddim_steps", 200)
        settings["guidance_scale"] = run_cfg.get("scale", 3.0)
        settings["ddim_eta"] = run_cfg.get("ddim_eta", 0.0)
        # Dataset name from the LDM config if available
        ldm_cfg = OmegaConf.load(settings["ldm_cfg"])
        ds_name = ldm_cfg.get("data", {}).get("name", "MNIST")
        settings["dataset"] = "fmnist" if "Fashion" in ds_name else "mnist"
    else:
        settings["ldm_cfg"] = args.ldm_cfg
        settings["ldm_ckpt"] = args.ldm_ckpt
        settings["clf_ckpt"] = args.clf_ckpt
        settings["clf_args"] = dict(
            input_channels=args.clf_input_channels,
            num_classes=args.clf_num_classes,
            softmax_flag=True,
        )
        settings["clf_input_size"] = args.clf_input_size
        settings["class_map"] = None
        settings["classes"] = args.classes
        settings["data_root"] = args.data_root
        settings["image_size"] = args.image_size
        settings["dataset"] = args.dataset
        settings["ddim_steps"] = args.ddim_steps
        settings["guidance_scale"] = args.guidance_scale
        settings["ddim_eta"] = args.ddim_eta

    return settings


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LDM generation quality check")

    # Shortcut: single run-config YAML (mirrors config_ldce_mnist.yaml)
    p.add_argument("--run_cfg", default=None,
                   help="LDCE run-config YAML that provides all paths (optional)")

    # Individual model paths (used when --run_cfg is not given)
    p.add_argument("--ldm_cfg",  default=None, help="LDM architecture YAML")
    p.add_argument("--ldm_ckpt", default=None, help="LDM checkpoint (.ckpt)")
    p.add_argument("--clf_ckpt", default=None, help="Classifier checkpoint (.pth)")
    p.add_argument("--clf_input_channels", type=int, default=1)
    p.add_argument("--clf_num_classes",    type=int, default=10)
    p.add_argument("--clf_input_size",     type=int, default=28,
                   help="Spatial size expected by the classifier (default: 28)")

    # Data
    p.add_argument("--dataset",    default="mnist", choices=list(_DATASET_REGISTRY),
                   help="Dataset to compare against (default: mnist)")
    p.add_argument("--data_root",  default="./data")
    p.add_argument("--image_size", type=int, default=32,
                   help="Input resolution for the LDM VAE (default: 32)")
    p.add_argument("--classes",    type=int, nargs="+", default=list(range(10)),
                   help="Which classes to evaluate (default: 0–9)")
    p.add_argument("--max_real_samples", type=int, default=None,
                   help="Max real images per class for FID / accuracy baselines")

    # Sampling
    p.add_argument("--n_per_class",     type=int,   default=50)
    p.add_argument("--ddim_steps",      type=int,   default=200)
    p.add_argument("--guidance_scale",  type=float, default=3.0)
    p.add_argument("--ddim_eta",        type=float, default=0.0)
    p.add_argument("--null_class",      type=int,   default=None,
                   help="Null/unconditional class index (auto-detected from config if omitted)")

    # Output
    p.add_argument("--output_dir", default="results/ldm_quality")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=64)

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.run_cfg is None and (args.ldm_cfg is None or args.ldm_ckpt is None):
        print("Error: provide either --run_cfg or both --ldm_cfg and --ldm_ckpt")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    cfg = resolve_config(args)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nDataset   : {_DATASET_NAMES.get(cfg['dataset'], cfg['dataset'])}")
    print(f"Classes   : {cfg['classes']}")
    print(f"Samples   : {args.n_per_class} per class")
    print(f"DDIM steps: {cfg['ddim_steps']}  |  CFG scale: {cfg['guidance_scale']}")

    # ── Load LDM ──────────────────────────────────────────────────────────────
    print("\nLoading LDM ...")
    ldm = load_ldm(cfg["ldm_cfg"], cfg["ldm_ckpt"], device)

    # Determine null class: either from CLI or from cond_stage n_classes - 1
    if args.null_class is not None:
        null_class = args.null_class
    else:
        try:
            ldm_cfg_data = OmegaConf.load(cfg["ldm_cfg"])
            null_class = ldm_cfg_data.model.params.cond_stage_config.params.n_classes - 1
        except Exception:
            null_class = max(cfg["classes"]) + 1
    print(f"  Null class index: {null_class}")

    # ── Load classifier ───────────────────────────────────────────────────────
    clf = None
    if cfg.get("clf_ckpt"):
        print("\nLoading classifier ...")
        clf = load_classifier(cfg["clf_ckpt"], cfg["clf_args"], device)
        clf_input_channels = cfg["clf_args"].get("input_channels", 1)
    else:
        print("\nNo classifier checkpoint provided — skipping accuracy metrics.")

    # ── Load real dataset ─────────────────────────────────────────────────────
    print(f"\nLoading real {_DATASET_NAMES.get(cfg['dataset'])} test set ...")
    real_ds = build_real_dataset(
        dataset_name=cfg["dataset"],
        image_size=cfg["image_size"],
        data_root=cfg["data_root"],
        classes=cfg["classes"],
        max_samples=args.max_real_samples,
    )
    real_loader = DataLoader(real_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)
    print(f"  Real samples: {len(real_ds)}")

    # ── Generate samples ──────────────────────────────────────────────────────
    print("\nGenerating samples ...")
    gen_samples, gen_labels = generate_samples(
        ldm_model=ldm,
        classes=cfg["classes"],
        n_per_class=args.n_per_class,
        ddim_steps=cfg["ddim_steps"],
        guidance_scale=cfg["guidance_scale"],
        ddim_eta=cfg["ddim_eta"],
        null_class=null_class,
        device=device,
    )
    print(f"  Generated: {gen_samples.shape[0]} images  shape={tuple(gen_samples.shape[1:])}")

    # Save sample grid
    grid_path = os.path.join(args.output_dir, "ldm_quality_samples.png")
    save_sample_grid(gen_samples, gen_labels, cfg["classes"],
                     n_per_row=min(args.n_per_class, 10), path=grid_path)
    print(f"  Saved sample grid → {grid_path}")

    # ── Pixel statistics ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pixel Statistics (generated samples)")
    print("=" * 60)
    px_stats = pixel_stats_per_class(gen_samples, gen_labels)
    for c, s in px_stats.items():
        flag = "" if s["non_degenerate"] else "  *** DEGENERATE ***"
        print(f"  Class {c:2d}: mean={s['mean']:.4f}  std={s['std']:.4f}"
              f"  min={s['min']:.4f}  max={s['max']:.4f}{flag}")
    all_stds = gen_samples.view(gen_samples.shape[0], -1).std(dim=1)
    degenerate = int((all_stds < 0.01).sum().item())
    if degenerate:
        print(f"\n  WARNING: {degenerate} / {gen_samples.shape[0]} images appear degenerate (std < 0.01)")

    # ── Classifier evaluation ─────────────────────────────────────────────────
    gen_results = None
    real_results = None

    if clf is not None:
        print("\n" + "=" * 60)
        print("  Classifier Accuracy")
        print("=" * 60)

        print("\n  [Real images]")
        real_results = real_classifier_accuracy(
            clf=clf,
            dataloader=real_loader,
            clf_input_size=cfg["clf_input_size"],
            clf_input_channels=clf_input_channels,
            class_map=cfg.get("class_map"),
            device=device,
        )
        print(f"  Overall accuracy : {real_results['overall_accuracy'] * 100:.1f}%"
              f"  ({real_results['correct']}/{real_results['total']})")
        for c in cfg["classes"]:
            acc = real_results["per_class_accuracy"].get(c, float("nan")) * 100
            print(f"    Class {c:2d}: {acc:.1f}%")

        print("\n  [Generated images]")
        gen_results = classifier_accuracy(
            clf=clf,
            samples=gen_samples,
            labels=gen_labels,
            clf_input_size=cfg["clf_input_size"],
            clf_input_channels=clf_input_channels,
            class_map=cfg.get("class_map"),
            device=device,
        )
        print(f"  Overall accuracy : {gen_results['overall_accuracy'] * 100:.1f}%"
              f"  ({gen_results['correct']}/{gen_results['total']})")
        for c in cfg["classes"]:
            acc = gen_results["per_class_accuracy"].get(c, float("nan")) * 100
            print(f"    Class {c:2d}: {acc:.1f}%")

        # Gap between real and generated accuracy
        gap = (real_results["overall_accuracy"] - gen_results["overall_accuracy"]) * 100
        print(f"\n  Real–Generated accuracy gap: {gap:+.1f}%")

        # Plots
        conf_path = os.path.join(args.output_dir, "confidence_distribution.png")
        plot_confidence_comparison(
            gen_probs=gen_results["all_probs"],
            real_probs=real_results["all_probs"],
            classes=cfg["classes"],
            output_path=conf_path,
        )
        print(f"  Saved confidence plot → {conf_path}")

        acc_path = os.path.join(args.output_dir, "per_class_accuracy.png")
        plot_per_class_accuracy(
            real_acc=real_results["per_class_accuracy"],
            gen_acc=gen_results["per_class_accuracy"],
            classes=cfg["classes"],
            output_path=acc_path,
        )
        print(f"  Saved accuracy bar chart → {acc_path}")

    # ── FID ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FID Score")
    print("=" * 60)
    fid = compute_fid(
        real_dataset=real_ds,
        generated_samples=gen_samples,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
    )
    if fid is not None:
        print(f"  FID : {fid:.2f}")
    else:
        print("  FID : not computed")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Dataset        : {_DATASET_NAMES.get(cfg['dataset'])} (classes {cfg['classes']})")
    print(f"  Generated      : {gen_samples.shape[0]} samples  ({args.n_per_class}/class)")
    print(f"  DDIM steps     : {cfg['ddim_steps']}   CFG scale: {cfg['guidance_scale']}")
    print(f"  Degenerate imgs: {degenerate}")
    if gen_results:
        print(f"  CLF acc (real) : {real_results['overall_accuracy'] * 100:.1f}%")
        print(f"  CLF acc (gen)  : {gen_results['overall_accuracy'] * 100:.1f}%")
    if fid is not None:
        print(f"  FID            : {fid:.2f}")
    print(f"  Output dir     : {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
