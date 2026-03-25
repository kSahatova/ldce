"""
MNIST LDCE with a Latent Diffusion Model (LDM).

This script is the entry point for the paper-aligned LDCE-cls experiment on
MNIST. It reuses all logic from ldce_mnist.py and only redirects the config
to mnist_ldce/config_ldm.yaml, which points to the class-conditional LDM
(ldm_mnist.yaml) trained on top of a KL-autoencoder (autoencoder_mnist_kl.yaml).

Prerequisites (run in order from the repository root):
    1. python train_vae_mnist.py   --output_dir ./checkpoints/mnist_vae
    2. python train_ldm_mnist.py   --output_dir ./checkpoints/mnist_ldm
    3. python ldce_mnist_ldm.py

Why a separate entry point?
────────────────────────────
ldce_mnist.py has CONFIG_PATH as a module-level constant. Overriding it here
before calling main() keeps the original file untouched while allowing both
variants (pixel-space DDPM and latent LDM) to coexist in the repo.
"""

import ldce_mnist

# Point to the LDM run config instead of the pixel-space DDPM config.
ldce_mnist.CONFIG_PATH = r"D:/VSCodeProjects/ldce/mnist_ldce/config_ldm.yaml"

# The LDM's ClassEmbedder also uses 11 classes (digits 0–9 + null class 10),
# so UNCOND_CLASS_IDX stays at 10 — no change needed.

if __name__ == "__main__":
    ldce_mnist.main()
