"""
Inference check for a class-conditional pixel-space DDPM trained on MNIST.

Loads a checkpoint produced by train_ddpm.py, generates one sample per digit
class (0–9) using DDIM, and saves a grid image to disk.

Usage:
    python check_mnist_ddpm.py --ckpt ./checkpoints/mnist_ddpm/last.ckpt
    python check_mnist_ddpm.py --ckpt ./checkpoints/mnist_ddpm/last.ckpt \\
        --cfg mnist_ldce/ddpm_mnist.yaml \\
        --ddim_steps 200 \\
        --scale 3.0 \\
        --output samples_check.png \\
        --device cuda
"""

import argparse
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import make_grid

# Make sure the repo root and taming-transformers are on the path
sys.path.insert(0, ".")
sys.path.insert(0, "./taming-transformers")

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


NULL_CLASS = 10   # must match n_classes - 1 in ddpm_mnist.yaml


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(cfg_path: str, ckpt_path: str, device: str) -> torch.nn.Module:
    config = OmegaConf.load(cfg_path)
    model = instantiate_from_config(config.model)

    state = torch.load(ckpt_path, map_location="cpu")
    # PyTorch-Lightning checkpoints store weights under 'state_dict'
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device)
    model.eval()
    return model


# ── Sampling ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_classes(
    model,
    classes: list,
    ddim_steps: int,
    scale: float,
    ddim_eta: float,
    device: str,
) -> torch.Tensor:
    """Generate one sample per class label.

    Returns a (N, 3, H, W) float tensor in [0, 1].
    """
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    batch_size = len(classes)
    class_tensor = torch.tensor(classes, dtype=torch.long, device=device)
    null_tensor  = torch.tensor([NULL_CLASS] * batch_size, dtype=torch.long, device=device)

    with model.ema_scope():
        c  = model.get_learned_conditioning({model.cond_stage_key: class_tensor})
        uc = model.get_learned_conditioning({model.cond_stage_key: null_tensor})

        shape = [model.channels, model.image_size, model.image_size]  # [3, 32, 32]
        samples, _ = sampler.sample(
            S=ddim_steps,
            batch_size=batch_size,
            shape=shape,
            conditioning=c,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            verbose=False,
        )

    # IdentityFirstStage decode is a no-op, but we call it for generality
    x = model.decode_first_stage(samples)
    x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)   # [-1, 1] → [0, 1]
    return x.cpu()


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def pixel_stats(samples: torch.Tensor) -> dict:
    """Basic sanity checks on the generated pixel values."""
    return {
        "min":  float(samples.min()),
        "max":  float(samples.max()),
        "mean": float(samples.mean()),
        "std":  float(samples.std()),
    }


def check_value_range(samples: torch.Tensor) -> bool:
    """Verify pixels are in [0, 1] after clamping (should always pass)."""
    return bool((samples >= 0.0).all() and (samples <= 1.0).all())


def check_not_degenerate(samples: torch.Tensor, std_threshold: float = 0.01) -> bool:
    """Flag suspiciously flat / all-identical images."""
    per_image_std = samples.view(samples.shape[0], -1).std(dim=1)
    return bool((per_image_std > std_threshold).all())


def channel_variance_check(samples: torch.Tensor) -> dict:
    """
    MNIST digits are replicated across 3 channels in train_ddpm.py.
    Inter-channel variance should be low for a well-trained model.
    """
    ch_diff = (samples[:, 0] - samples[:, 1]).abs().mean().item()
    return {"mean_inter_channel_diff": ch_diff}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inference check for a MNIST DDPM checkpoint")
    p.add_argument("--ckpt",       required=True,  help="Path to the .ckpt file")
    p.add_argument("--cfg",        default="mnist_ldce/ddpm_mnist.yaml",
                   help="Model architecture YAML (default: mnist_ldce/ddpm_mnist.yaml)")
    p.add_argument("--ddim_steps", type=int,   default=200, help="DDIM denoising steps (default: 200)")
    p.add_argument("--scale",      type=float, default=3.0, help="Classifier-free guidance scale (default: 3.0)")
    p.add_argument("--eta",        type=float, default=0.0, help="DDIM eta — 0 = deterministic (default: 0.0)")
    p.add_argument("--output",     default="samples_check.png", help="Output grid image path")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device  : {args.device}")
    print(f"Config  : {args.cfg}")
    print(f"Ckpt    : {args.ckpt}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("\nLoading model …")
    model = load_model(args.cfg, args.ckpt, args.device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

    # ── Generate one sample per digit 0–9 ────────────────────────────────────
    classes = list(range(10))
    print(f"\nSampling classes {classes} with {args.ddim_steps} DDIM steps, scale={args.scale} …")
    samples = sample_classes(
        model,
        classes=classes,
        ddim_steps=args.ddim_steps,
        scale=args.scale,
        ddim_eta=args.eta,
        device=args.device,
    )
    print(f"  Output shape: {tuple(samples.shape)}")   # (10, 3, 32, 32)

    # ── Checks ────────────────────────────────────────────────────────────────
    print("\nChecks:")
    stats = pixel_stats(samples)
    print(f"  Pixel stats  : min={stats['min']:.4f}  max={stats['max']:.4f}  "
          f"mean={stats['mean']:.4f}  std={stats['std']:.4f}")

    range_ok  = check_value_range(samples)
    nonflat   = check_not_degenerate(samples)
    ch_info   = channel_variance_check(samples)

    print(f"  Value range [0,1]          : {'PASS' if range_ok  else 'FAIL'}")
    print(f"  Non-degenerate (std > 0.01): {'PASS' if nonflat   else 'FAIL — images look flat/identical'}")
    print(f"  Inter-channel diff (expect ~0 for MNIST): {ch_info['mean_inter_channel_diff']:.5f}")

    all_pass = range_ok and nonflat
    print(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    # ── Save grid ─────────────────────────────────────────────────────────────
    grid = make_grid(samples, nrow=5, padding=2)          # 2 rows of 5 digits
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(grid_np).save(args.output)
    print(f"\nSaved sample grid → {args.output}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
