"""
MNIST LDCE – generate counterfactual explanations for a digit classifier.

Run from the REPOSITORY ROOT:
    python -m mnist_ldce.run

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

Requirements
─────────────
  • A pretrained MNIST DDPM checkpoint matching the architecture in
    mnist_ldce/ddpm_mnist.yaml  (ckpt_path in config.yaml).
  • A pretrained MNIST digit classifier (10 output logits) whose checkpoint
    path is set in config.yaml.  Edit load_user_classifier() below to match
    how your checkpoint is saved.

All other settings (sampler hyper-params, batch size, etc.) live in
mnist_ldce/config.yaml.
"""

import os
import copy
import random
import pathlib

import yaml
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

# ── Repo-internal imports (must run from repository root) ─────────────────────
from sampling_helpers import disabled_train, get_model, _unmap_img
from ldm.models.diffusion.cc_ddim import CCMDDIMSampler
from utils.preprocessor import GenericPreprocessing, Normalizer

from mnist_ldce.dataset import MNISTForLDCE, DIGIT_NAMES, MNIST_CLOSEST_CLASS

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = r"D:/VSCodeProjects/ldce/mnist_ldce/config.yaml"

# Index of the null/unconditional class in the DDPM's ClassEmbedder.
# Must be n_classes - 1 from ddpm_mnist.yaml (10 digits → index 10).
UNCOND_CLASS_IDX = 10


# ─────────────────────────────────────────────────────────────────────────────
# Classifier loading  ← EDIT THIS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def load_user_classifier(ckpt_path: str, device: torch.device) -> nn.Module:
    """Load YOUR pretrained MNIST classifier.

    The classifier must:
      • accept (B, 3, H, W) float tensors in [0, 1] range
      • return (B, 10) logits (one per digit class)

    If your model was trained on single-channel images (1, H, W) you can add a
    grayscale conversion inside the model or replace `image.repeat(3, 1, 1)` in
    dataset.py with keeping 1 channel and updating this wrapper accordingly.

    Edit one of the examples below to match how your checkpoint is saved,
    then remove the NotImplementedError.
    """

    # ── Example A: the entire model object was saved ──────────────────────────
    # model = torch.load(ckpt_path, map_location='cpu')

    # ── Example B: only the state_dict was saved ──────────────────────────────
    # from your_package import YourMNISTNet
    # model = YourMNISTNet()
    # model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    # ── Example C: checkpoint dict with 'state_dict' or 'model' key ──────────
    # ckpt = torch.load(ckpt_path, map_location='cpu')
    # model = YourMNISTNet()
    # model.load_state_dict(ckpt.get('state_dict', ckpt.get('model', ckpt)))

    raise NotImplementedError(
        "\nPlease implement load_user_classifier() in mnist_ldce/run.py.\n"
        f"Checkpoint expected at: {ckpt_path}\n"
        "The model must accept (B, 3, H, W) tensors in [0, 1] and return (B, 10) logits."
    )

    return model.to(device).eval()  # noqa: unreachable – replace with your code


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def pick_target_class(label: int, method: str, logits: torch.Tensor) -> int:
    """Return the counterfactual target class for one input digit.

    Args:
        label  : true digit class (0–9)
        method : 'closest' | 'random' | 'second_best'
        logits : (10,) classifier logits for this image (used by 'second_best')
    """
    if method == 'closest':
        return MNIST_CLOSEST_CLASS[label][0]
    elif method == 'random':
        return random.choice([c for c in range(10) if c != label])
    elif method == 'second_best':
        sorted_classes = logits.argsort(descending=True).tolist()
        for cls in sorted_classes:
            if cls != label:
                return cls
        return MNIST_CLOSEST_CLASS[label][0]
    else:
        raise ValueError(f"Unknown target_class_method: '{method}'")


# ─────────────────────────────────────────────────────────────────────────────
# MNIST-specific sample generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_samples_mnist(
        model, sampler, target_y: torch.Tensor,
        ddim_steps: int, scale: float,
        init_latent: torch.Tensor, t_enc: int,
        seed: int = 0) -> dict:
    """MNIST-adapted version of sampling_helpers.generate_samples().

    The only functional difference from the original is that the unconditional
    class index is UNCOND_CLASS_IDX (= 10) instead of the ImageNet-hardcoded
    1000.  Everything else is identical.

    Returns a dict with keys: 'samples', 'probs', 'videos', 'masks', 'cgs'.
    """
    torch.cuda.empty_cache()
    batch_size = target_y.shape[0]

    with torch.no_grad():
        with model.ema_scope():
            # Unconditional conditioning: null class (index 10)
            uc = model.get_learned_conditioning({
                model.cond_stage_key: torch.tensor(
                    batch_size * [UNCOND_CLASS_IDX]
                ).to(model.device)
            })
            # Target-class conditioning
            c = model.get_learned_conditioning({
                model.cond_stage_key: target_y.to(model.device)
            })

            # Forward diffusion: add noise up to timestep t_enc
            torch.manual_seed(seed)
            noise = torch.randn_like(init_latent)
            z_enc = sampler.stochastic_encode(
                init_latent,
                torch.tensor([t_enc] * batch_size).to(init_latent.device),
                noise=noise
            )

            torch.manual_seed(seed)

            # Reverse diffusion with classifier + distance guidance
            out = sampler.decode(
                z_enc, c, t_enc,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                y=target_y.to(model.device),
                latent_t_0=False,
            )

    samples = out["x_dec"]
    prob    = out.get("prob")
    vid     = out.get("video")
    mask    = out.get("mask")
    cg      = out.get("concensus_regions")

    # Decode from latent / pixel space and clip to [0, 1]
    # For IdentityFirstStage this is just (x + 1) / 2 (un-mapping from [-1,1])
    x_samples = model.decode_first_stage(samples)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    return {
        "samples": [x_samples],
        "probs":   [prob]  if prob is not None else None,
        "videos":  [vid]   if vid  is not None else None,
        "masks":   [mask]  if mask is not None else None,
        "cgs":     [cg]    if cg   is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config(CONFIG_PATH)

    seed = cfg.get('seed', 0)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Output directories ────────────────────────────────────────────────────
    out_dir         = os.path.join(cfg['output_dir'], "results")
    checkpoint_path = os.path.join(out_dir, "last_saved_id.pth")
    os.makedirs(out_dir, exist_ok=True)

    # ── Resume support ────────────────────────────────────────────────────────
    last_data_idx = 0
    if cfg.get('resume', False) and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        last_data_idx = ckpt.get("last_data_idx", -1) + 1
        print(f"Resuming from sample index {last_data_idx}")

    # ── Diffusion model ───────────────────────────────────────────────────────
    print("Loading MNIST diffusion model …")
    model = get_model(
        cfg_path  = cfg['diffusion_model']['cfg_path'],
        ckpt_path = cfg['diffusion_model']['ckpt_path'],
    ).to(device).eval()

    # ── Classifier ───────────────────────────────────────────────────────────
    print("Loading MNIST classifier …")
    classifier_raw = load_user_classifier(
        ckpt_path = cfg['classifier_model']['ckpt_path'],
        device    = device,
    )

    # The CCMDDIMSampler calls get_classifier_logits(pred_x0) internally, which
    # maps pred_x0 from [-1, 1] → [0, 1] before calling self.classifier(x).
    # So the classifier always receives [0, 1] images.
    #
    # If your classifier additionally requires per-channel MNIST normalisation
    # (mean=0.1307, std=0.3081), enable mnist_normalisation in config.yaml.
    if cfg['classifier_model'].get('mnist_normalisation', False):
        mean = [0.1307] * 3
        std  = [0.3081] * 3
        classifier_model = Normalizer(classifier_raw, mean, std)
    else:
        classifier_model = classifier_raw

    classifier_model = classifier_model.to(device).eval()
    classifier_model.train = disabled_train  # freeze train/eval toggle

    # ── CCMDDIMSampler ────────────────────────────────────────────────────────
    sampler_cfg = cfg['sampler']
    ddim_steps  = cfg['ddim_steps']
    ddim_eta    = cfg['ddim_eta']
    scale       = cfg['scale']
    strength    = cfg['strength']

    sampler = CCMDDIMSampler(
        model,
        classifier_model,
        seg_model                  = None,
        classifier_wrapper         = cfg['classifier_model'].get('classifier_wrapper', True),
        record_intermediate_results= cfg.get('record_intermediate_results', False),
        verbose                    = True,
        **sampler_cfg,
    )
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    t_enc      = int(strength * len(sampler.ddim_timesteps))
    batch_size = cfg['data']['batch_size']

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    dataset = MNISTForLDCE(
        root        = cfg['data']['root'],
        split       = cfg['data']['split'],
        image_size  = cfg['data']['image_size'],
        restart_idx = last_data_idx,
    )
    print(f"Dataset: {len(dataset)} samples (split='{cfg['data']['split']}')")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    target_method = cfg.get('target_class_method', 'closest')
    log_rate      = cfg.get('log_rate', 10)

    # ── Generation loop ───────────────────────────────────────────────────────
    for i, (image, label, unique_data_idx) in enumerate(data_loader):
        image = image.to(device)   # (B, 3, 32, 32) in [0, 1]
        label = label.to(device)

        cur_batch = image.shape[0]

        # Initial classifier predictions (input images are already [0, 1])
        with torch.inference_mode():
            logits_in = classifier_model(image)   # (B, 10)
            in_class_pred = logits_in.argmax(dim=1)
            in_confid     = logits_in.softmax(dim=1).max(dim=1).values

        # Select counterfactual target classes
        tgt_classes = torch.tensor([
            pick_target_class(
                label[j].item(),
                method=target_method,
                logits=logits_in[j].cpu(),
            )
            for j in range(cur_batch)
        ]).to(device)

        in_confid_tgt = logits_in.softmax(dim=1)[
            torch.arange(cur_batch), tgt_classes
        ]

        for j in range(cur_batch):
            src_name = DIGIT_NAMES[label[j].item()]
            tgt_name = DIGIT_NAMES[tgt_classes[j].item()]
            print(f"  [{i * batch_size + j:05d}] {src_name} → {tgt_name}")

        # Set reference images for the distance regulariser (must be [0, 1])
        sampler.init_images = image.clone()
        sampler.init_labels = label

        # Encode into pixel / latent space: _unmap_img maps [0,1] → [-1,1]
        init_latent = model.get_first_stage_encoding(
            model.encode_first_stage(_unmap_img(image))
        )

        # Generate counterfactuals
        out = generate_samples_mnist(
            model, sampler, tgt_classes,
            ddim_steps, scale,
            init_latent = init_latent,
            t_enc       = t_enc,
            seed        = seed,
        )

        all_samples = out["samples"]   # list of (B, 3, 32, 32) tensors in [0, 1]
        all_probs   = out["probs"]

        # Evaluate output images
        with torch.inference_mode():
            logits_out    = classifier_model(all_samples[0])
            out_class_pred = logits_out.argmax(dim=1)
            out_confid     = logits_out.softmax(dim=1).max(dim=1).values
            out_confid_tgt = logits_out.softmax(dim=1)[
                torch.arange(cur_batch), tgt_classes
            ]

        print(f"  Input  preds: {in_class_pred.tolist()}, "
              f"conf: {[f'{v:.2f}' for v in in_confid.tolist()]}")
        print(f"  Output preds: {out_class_pred.tolist()}, "
              f"conf: {[f'{v:.2f}' for v in out_confid.tolist()]}")

        # Save results
        pathlib.Path(os.path.join(out_dir, 'original')).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(os.path.join(out_dir, 'counterfactual')).mkdir(
            parents=True, exist_ok=True)

        for j in range(cur_batch):
            uidx      = unique_data_idx[j].item()
            src_image = sampler.init_images[j].cpu()
            gen_image = all_samples[0][j].cpu()
            diff      = src_image - gen_image

            data_dict = {
                "unique_id":         uidx,
                "image":             src_image,
                "gen_image":         gen_image,
                "source":            DIGIT_NAMES[label[j].item()],
                "target":            DIGIT_NAMES[tgt_classes[j].item()],
                "in_pred":           DIGIT_NAMES[in_class_pred[j].item()],
                "out_pred":          DIGIT_NAMES[out_class_pred[j].item()],
                "in_confid":         in_confid[j].cpu().item(),
                "out_confid":        out_confid[j].cpu().item(),
                "in_tgt_confid":     in_confid_tgt[j].cpu().item(),
                "out_tgt_confid":    out_confid_tgt[j].cpu().item(),
                "target_confidence": (
                    all_probs[0][j]
                    if all_probs is not None
                    else out_confid_tgt[j].cpu().item()
                ),
                "closeness_l1": int(
                    torch.norm(diff, p=1, dim=-1).mean().item()
                ),
                "closeness_l2": int(
                    torch.norm(diff, p=2, dim=-1).mean().item()
                ),
            }

            torch.save(
                data_dict,
                os.path.join(out_dir, f'{str(uidx).zfill(5)}.pth')
            )
            save_image(
                src_image.clamp(0, 1),
                os.path.join(out_dir, 'original', f'{str(uidx).zfill(5)}.png')
            )
            save_image(
                gen_image.clamp(0, 1),
                os.path.join(out_dir, 'counterfactual', f'{str(uidx).zfill(5)}.png')
            )

        # Periodic checkpoint for resuming
        if (i + 1) % log_rate == 0:
            last_idx = unique_data_idx[-1].item()
            torch.save({"last_data_idx": last_idx}, checkpoint_path)
            print(f"  Checkpoint saved (last sample index: {last_idx})")

        del out


if __name__ == "__main__":
    main()
