"""
MNIST LDCE – generate counterfactual explanations for a digit classifier.

Run from the REPOSITORY ROOT:
    python ldce_mnist.py

What this script does
─────────────────────
For each MNIST test image the script:
  1. Encodes the image with the diffusion model's first stage (identity for MNIST).
  2. Adds noise up to timestep t_enc = strength × ddim_steps.
  3. Denoises with CCMDDIMSampler, guided by:
       • the classifier gradient  (push toward target digit class)
       • an Lp distance penalty   (stay close to the original image)
  4. Saves the original and counterfactual images plus a .pth metadata dict.

Outputs (in output_dir/results/)
─────────────────────────────────
  original/00000.png          – input digit images
  counterfactual/00000.png    – generated counterfactuals
  00000.pth                   – metadata dict with predictions and distances

All settings live in mnist_ldce/config.yaml.
"""

import os
import sys
import random
import pathlib

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from sampling_helpers import disabled_train, get_model, _unmap_img
from ldm.models.diffusion.cc_ddim import CCMDDIMSampler
from ldm.models.classifiers import SimpleCNNtorch, CNNtorch
from utils.preprocessor import Normalizer

from mnist_ldce.dataset import MNISTForLDCE, DIGIT_NAMES, MNIST_CLOSEST_CLASS

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = r"D:/VSCodeProjects/ldce/mnist_ldce/config.yaml"

# Null/unconditional class index in the DDPM's ClassEmbedder.
# Must equal n_classes - 1 from ddpm_mnist.yaml (10 digits → index 10).
UNCOND_CLASS_IDX = 10


# ─────────────────────────────────────────────────────────────────────────────
# Classifier components
# ─────────────────────────────────────────────────────────────────────────────

class ResizeWrapper(nn.Module):
    """Resize spatial dims and reduce to `out_channels` before classification.

    The pipeline operates at `pipeline_size` × `pipeline_size` (e.g. 32×32)
    while the classifier was trained at `clf_size` × `clf_size` (e.g. 28×28)
    on `out_channels`-channel images (e.g. 1 for grayscale CNNtorch).
    """

    def __init__(self, model: nn.Module, clf_size: int, out_channels: int = 1):
        super().__init__()
        self.model       = model
        self.clf_size    = clf_size
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, size=(self.clf_size, self.clf_size),
            mode='bilinear', align_corners=False,
        )
        return self.model(x[:, :self.out_channels])


def load_classifier(classifier_args: dict, ckpt_path: str,
                    device: torch.device) -> nn.Module:
    """Load the pretrained MNIST classifier from a checkpoint.

    Handles checkpoints saved from a codebase where the class lived under
    a 'src/' package by installing a temporary import stub.
    """
    import importlib.abc
    import importlib.machinery
    import types

    class _AnyObj:
        def __setstate__(self, state: dict) -> None:
            self.__dict__.update(state if isinstance(state, dict) else {})

    class _SrcStub(types.ModuleType):
        def __getattr__(self, _name: str):
            return _AnyObj

    class _SrcFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):  # noqa: ARG002
            if fullname == 'src' or fullname.startswith('src.'):
                return importlib.machinery.ModuleSpec(fullname, self)
        def create_module(self, spec):
            return _SrcStub(spec.name)
        def exec_module(self, module):
            sys.modules[module.__name__] = module

    sys.meta_path.insert(0, _SrcFinder())
    try:
        checkpoint = torch.load(
            ckpt_path, weights_only=False, map_location=torch.device(device)
        )
    finally:
        sys.meta_path.pop(0)

    model = CNNtorch(classifier_args['input_channels'],
                     classifier_args['num_classes'])
    model.load_state_dict(checkpoint)
    return model.to(device).eval()


def build_classifier_pipeline(cfg: dict, device: torch.device) -> nn.Module:
    """Load classifier and apply resize + normalisation wrappers as configured."""
    model = load_classifier(
        cfg['classifier_model']['args'],
        cfg['classifier_model']['ckpt_path'],
        device,
    )

    clf_size      = cfg['classifier_model'].get('input_size', cfg['data']['image_size'])
    pipeline_size = cfg['data']['image_size']
    in_channels   = cfg['classifier_model']['args'].get('input_channels', 1)

    if clf_size != pipeline_size or in_channels != 3:
        print(f"  Classifier: resize {pipeline_size}→{clf_size}, "
              f"channels 3→{in_channels}")
        model = ResizeWrapper(model, clf_size, out_channels=in_channels)

    # if cfg['classifier_model'].get('mnist_normalisation', False):
    #     mean  = [0.1307] * in_channels
    #     std   = [0.3081] * in_channels
    #     model = Normalizer(model, mean, std)

    model = model.to(device).eval()
    model.train = disabled_train
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Sampler & dataloader builders
# ─────────────────────────────────────────────────────────────────────────────

def build_sampler(model, classifier: nn.Module,
                  cfg: dict) -> tuple[CCMDDIMSampler, int]:
    """Construct CCMDDIMSampler, build its noise schedule, return (sampler, t_enc)."""
    sampler = CCMDDIMSampler(
        model,
        classifier,
        seg_model                   = None,
        classifier_wrapper          = cfg['classifier_model'].get('classifier_wrapper', True),
        record_intermediate_results = cfg.get('record_intermediate_results', False),
        verbose                     = True,
        **cfg['sampler'],
    )
    sampler.make_schedule(
        ddim_num_steps = cfg['ddim_steps'],
        ddim_eta       = cfg['ddim_eta'],
        verbose        = False,
    )
    t_enc = int(cfg['strength'] * len(sampler.ddim_timesteps))
    return sampler, t_enc


def completed_ids(out_dir: str) -> set:
    """Return the set of MNIST indices that already have a saved .pth result."""
    return {
        int(p.stem)
        for p in pathlib.Path(out_dir).glob('*.pth')
        if p.stem.isdigit()
    }


def build_dataloader(cfg: dict, out_dir: str,
                     restart_idx: int = 0) -> torch.utils.data.DataLoader:
    """Build the MNIST dataset and DataLoader from config, skipping completed samples."""
    done     = completed_ids(out_dir)
    dataset  = MNISTForLDCE(
        root           = cfg['data']['root'],
        split          = cfg['data']['split'],
        image_size     = cfg['data']['image_size'],
        restart_idx    = restart_idx,
        max_samples    = cfg['data'].get('max_samples', None),
        filter_classes = cfg['data'].get('filter_classes', None),
        skip_ids       = done,
    )
    print(f"Dataset: {len(dataset)} samples  "
          f"(split='{cfg['data']['split']}', "
          f"filter={cfg['data'].get('filter_classes', 'all')}, "
          f"skipping {len(done)} already completed)")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size  = cfg['data']['batch_size'],
        shuffle     = False,
        num_workers = 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Target class selection
# ─────────────────────────────────────────────────────────────────────────────

def pick_target_class(label: int, method: str, logits: torch.Tensor,
                      fixed_target: int = None) -> int:
    """Return the counterfactual target class for one input digit.

    Args:
        label        : true digit class (0–9)
        method       : 'closest' | 'random' | 'second_best' | 'fixed'
        logits       : (10,) classifier logits (used only by 'second_best')
        fixed_target : target class when method == 'fixed'
    """
    if method == 'closest':
        return MNIST_CLOSEST_CLASS[label][0]
    if method == 'random':
        return random.choice([c for c in range(10) if c != label])
    if method == 'second_best':
        for cls in logits.argsort(descending=True).tolist():
            if cls != label:
                return cls
        return MNIST_CLOSEST_CLASS[label][0]
    if method == 'fixed':
        if fixed_target is None:
            raise ValueError("target_class_method='fixed' requires 'target_class' in config")
        if fixed_target == label:
            raise ValueError(
                f"fixed target_class ({fixed_target}) matches source label ({label})"
            )
        return fixed_target
    raise ValueError(f"Unknown target_class_method: '{method}'")


def get_target_classes(labels: torch.Tensor, method: str,
                       logits: torch.Tensor, fixed_target: int,
                       device: torch.device) -> torch.Tensor:
    """Vectorised wrapper: returns (B,) target class tensor."""
    if method == 'fixed':
        return torch.full_like(labels, fixed_target)
    return torch.tensor(
        [pick_target_class(labels[j].item(), method, logits[j].cpu(), fixed_target)
         for j in range(len(labels))],
        device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Classifier evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(classifier: nn.Module, cfg: dict,
                        device: torch.device) -> dict:
    """Evaluate classifier top-1 accuracy on the full split (no class filter)."""
    clf_size = cfg['classifier_model'].get('input_size', cfg['data']['image_size'])
    split    = cfg['data']['split']

    dataset = MNISTForLDCE(
        root       = cfg['data']['root'],
        split      = split,
        image_size = clf_size,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['data']['batch_size'],
        shuffle=False, num_workers=0,
    )

    per_class_correct = {c: 0 for c in range(10)}
    per_class_total   = {c: 0 for c in range(10)}

    with torch.inference_mode():
        for images, labels, _ in loader:
            preds = classifier(images.to(device)).argmax(dim=1).cpu()
            for pred, label in zip(preds.tolist(), labels.tolist()):
                per_class_total[label]   += 1
                per_class_correct[label] += int(pred == label)

    n_correct = sum(per_class_correct.values())
    n_total   = sum(per_class_total.values())
    accuracy  = n_correct / n_total if n_total > 0 else 0.0
    per_class_acc = {
        c: (per_class_correct[c] / per_class_total[c]
            if per_class_total[c] > 0 else 0.0)
        for c in range(10)
    }

    print(f"\nClassifier evaluation — MNIST {split}  ({n_total} images)")
    print(f"  Overall accuracy : {accuracy * 100:.2f}%  ({n_correct}/{n_total})")
    for c in range(10):
        bar = '█' * int(per_class_acc[c] * 20)
        print(f"    {DIGIT_NAMES[c]:>7} ({c}):  {per_class_acc[c]*100:6.2f}%  {bar}")
    print()

    return {'accuracy': accuracy, 'per_class_acc': per_class_acc,
            'n_correct': n_correct, 'n_total': n_total}


# ─────────────────────────────────────────────────────────────────────────────
# Counterfactual generation (one batch)
# ─────────────────────────────────────────────────────────────────────────────

def generate_counterfactual_batch(
        model, sampler: CCMDDIMSampler,
        images: torch.Tensor, target_y: torch.Tensor,
        scale: float, t_enc: int, seed: int,
) -> tuple[torch.Tensor, list]:
    """Encode → add noise → guided denoise → decode for one batch.

    Returns:
        cf_images : (B, C, H, W) counterfactual images in [0, 1]
        probs     : list of per-image target-class probabilities (or empty list)
    """
    sampler.init_images = images.clone()

    init_latent = model.get_first_stage_encoding(
        model.encode_first_stage(_unmap_img(images))
    )

    batch_size = target_y.shape[0]

    with torch.no_grad(), model.ema_scope():
        uc = model.get_learned_conditioning({
            model.cond_stage_key: torch.tensor(
                batch_size * [UNCOND_CLASS_IDX]
            ).to(model.device)
        })
        c = model.get_learned_conditioning({
            model.cond_stage_key: target_y.to(model.device)
        })

        torch.manual_seed(seed)
        z_enc = sampler.stochastic_encode(
            init_latent,
            torch.tensor([t_enc] * batch_size).to(init_latent.device),
            noise=torch.randn_like(init_latent),
        )

        torch.manual_seed(seed)
        out = sampler.decode(
            z_enc, c, t_enc,
            unconditional_guidance_scale = scale,
            unconditional_conditioning   = uc,
            y                            = target_y.to(model.device),
            latent_t_0                   = False,
        )

    cf_images = torch.clamp(
        (model.decode_first_stage(out["x_dec"]) + 1.0) / 2.0, 0.0, 1.0
    )
    probs = out.get("prob") or []
    return cf_images, probs


# ─────────────────────────────────────────────────────────────────────────────
# Result saving (one batch)
# ─────────────────────────────────────────────────────────────────────────────

def save_batch(
        out_dir: str,
        unique_ids: torch.Tensor,
        src_images: torch.Tensor,
        cf_images:  torch.Tensor,
        labels:     torch.Tensor,
        tgt_classes: torch.Tensor,
        logits_in:  torch.Tensor,
        logits_out: torch.Tensor,
        probs:      list,
        save_only_successful: bool = True,
) -> int:
    """Persist originals, counterfactuals and metadata for one batch.

    Args:
        save_only_successful: when True, skip samples where the counterfactual
                              did not flip the classifier to the target class.
    Returns:
        Number of samples actually saved.
    """
    softmax_in  = logits_in.softmax(dim=1).cpu()
    softmax_out = logits_out.softmax(dim=1).cpu()
    in_pred     = logits_in.argmax(dim=1).cpu()
    out_pred    = logits_out.argmax(dim=1).cpu()

    saved = 0
    for j in range(len(unique_ids)):
        uidx      = unique_ids[j].item()
        src       = src_images[j].cpu()
        cf        = cf_images[j].cpu()
        tgt       = tgt_classes[j].item()
        lbl       = labels[j].item()
        diff      = src - cf

        if save_only_successful and out_pred[j].item() != tgt:
            continue

        data_dict = {
            "unique_id":         uidx,
            "image":             src,
            "gen_image":         cf,
            "source":            DIGIT_NAMES[lbl],
            "target":            DIGIT_NAMES[tgt],
            "in_pred":           DIGIT_NAMES[in_pred[j].item()],
            "out_pred":          DIGIT_NAMES[out_pred[j].item()],
            "in_confid":         softmax_in[j].max().item(),
            "out_confid":        softmax_out[j].max().item(),
            "in_tgt_confid":     softmax_in[j, tgt].item(),
            "out_tgt_confid":    softmax_out[j, tgt].item(),
            "target_confidence": probs[j] if probs else softmax_out[j, tgt].item(),
            "closeness_l1":      torch.norm(diff, p=1).item(),
            "closeness_l2":      torch.norm(diff, p=2).item(),
        }

        fname = str(uidx).zfill(5)
        torch.save(data_dict,
                   os.path.join(out_dir, f'{fname}.pth'))
        save_image(src.clamp(0, 1),
                   os.path.join(out_dir, 'original', f'{fname}.png'))
        save_image(cf.clamp(0, 1),
                   os.path.join(out_dir, 'counterfactual', f'{fname}.png'))
        saved += 1

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg    = load_config(CONFIG_PATH)
    seed   = cfg.get('seed', 0)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Output directories ────────────────────────────────────────────────────
    out_dir         = os.path.join(cfg['output_dir'], "results")
    resume_ckpt     = os.path.join(out_dir, "last_saved_id.pth")
    pathlib.Path(os.path.join(out_dir, 'original')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(out_dir, 'counterfactual')).mkdir(parents=True, exist_ok=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    last_data_idx = 0
    if cfg.get('resume', False) and os.path.exists(resume_ckpt):
        last_data_idx = torch.load(resume_ckpt, map_location='cpu').get("last_data_idx", -1) + 1
        print(f"Resuming from sample index {last_data_idx}")

    # ── Models ────────────────────────────────────────────────────────────────
    print("Loading MNIST classifier …")
    classifier = build_classifier_pipeline(cfg, device)

    print("Loading MNIST diffusion model …")
    diff_model = get_model(
        cfg_path  = cfg['diffusion_model']['cfg_path'],
        ckpt_path = cfg['diffusion_model']['ckpt_path'],
    ).to(device).eval()

    # ── Optional classifier quality check ─────────────────────────────────────
    if cfg.get('evaluate_classifier', True):
        evaluate_classifier(classifier, cfg, device)

    # ── Sampler & data ────────────────────────────────────────────────────────
    sampler, t_enc = build_sampler(diff_model, classifier, cfg)
    data_loader    = build_dataloader(cfg, out_dir, restart_idx=last_data_idx)

    target_method        = cfg.get('target_class_method', 'closest')
    fixed_target         = cfg.get('target_class', None)
    log_rate             = cfg.get('log_rate', 10)
    batch_size           = cfg['data']['batch_size']
    save_only_successful = cfg.get('save_only_successful', True)

    total_saved = 0

    # ── Generation loop ───────────────────────────────────────────────────────
    for i, (images, labels, unique_ids) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            logits_in = classifier(images)

        tgt_classes = get_target_classes(
            labels, target_method, logits_in, fixed_target, device
        )

        for j in range(len(labels)):
            print(f"  [{i * batch_size + j:05d}] "
                  f"{DIGIT_NAMES[labels[j].item()]} → "
                  f"{DIGIT_NAMES[tgt_classes[j].item()]}")

        cf_images, probs = generate_counterfactual_batch(
            diff_model, sampler, images, tgt_classes,
            cfg['scale'], t_enc, seed,
        )

        with torch.inference_mode():
            logits_out = classifier(cf_images)

        print(f"  Input  preds: {logits_in.argmax(1).tolist()},  "
              f"conf: {[f'{v:.2f}' for v in logits_in.softmax(1).max(1).values.tolist()]}")
        print(f"  Output preds: {logits_out.argmax(1).tolist()},  "
              f"conf: {[f'{v:.2f}' for v in logits_out.softmax(1).max(1).values.tolist()]}")

        n_saved = save_batch(
            out_dir, unique_ids, images, cf_images,
            labels, tgt_classes, logits_in, logits_out, probs,
            save_only_successful=save_only_successful,
        )
        total_saved += n_saved
        print(f"  Saved {n_saved}/{len(labels)} this batch  "
              f"(total: {total_saved})")

        if (i + 1) % log_rate == 0:
            last_idx = unique_ids[-1].item()
            torch.save({"last_data_idx": last_idx}, resume_ckpt)
            print(f"  Checkpoint saved (last index: {last_idx})")


if __name__ == "__main__":
    main()
