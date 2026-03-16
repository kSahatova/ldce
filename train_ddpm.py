"""
Train a class-conditional pixel-space DDPM on MNIST or FashionMNIST.

The model architecture is defined in mnist_ldce/ddpm_mnist.yaml and is
compatible with the LDCE pipeline in ldce_mnist.py.

Usage (from the repository root):
    # MNIST
    python train_ddpm.py --dataset mnist --output_dir ./checkpoints/mnist_ddpm

    # FashionMNIST
    python train_ddpm.py --dataset fashion_mnist --output_dir ./checkpoints/fashion_mnist_ddpm

    # Resume from a checkpoint
    python train_ddpm.py --dataset mnist --resume_ckpt ./checkpoints/mnist_ddpm/last.ckpt

Key design choices
──────────────────
  • Batch format: dicts with 'image' (B, H, W, C) in [-1, 1] and
    'class_label' (B,) – exactly what LatentDiffusion.get_input() expects.
  • Classifier-free guidance (CFG) dropout: during training, class labels are
    randomly replaced with the null/unconditional class index (10) at rate
    `cfg_dropout`. This enables CFG at inference time.
  • IdentityFirstStage: no VAE; the UNet operates directly on pixel tensors.
  • Checkpoints are saved as standard PyTorch Lightning .ckpt files with a
    'state_dict' key, compatible with sampling_helpers.get_model().
"""

import argparse
import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from ldm.util import instantiate_from_config


# ── Constants ─────────────────────────────────────────────────────────────────

# Index reserved for the unconditional / null class in the ClassEmbedder.
# Must match n_classes - 1 in ddpm_mnist.yaml (10 digits → index 10).
NULL_CLASS = 10

FASHION_MNIST_NAMES = {
    0: 'T-shirt/top', 1: 'Trouser',  2: 'Pullover', 3: 'Dress',   4: 'Coat',
    5: 'Sandal',      6: 'Shirt',    7: 'Sneaker',  8: 'Bag',      9: 'Ankle boot',
}

MNIST_NAMES = {i: str(i) for i in range(10)}


# ── Dataset ───────────────────────────────────────────────────────────────────

class DDPMDataset(Dataset):
    """Wraps torchvision MNIST / FashionMNIST for LatentDiffusion training.

    Returns dicts:
        'image'       – (H, W, C) float32 tensor in [-1, 1]   ← b h w c layout
        'class_label' – int64 scalar (or NULL_CLASS after CFG dropout)
    """

    def __init__(
        self,
        dataset_name: str,          # 'mnist' or 'fashion_mnist'
        split: str,                 # 'train' or 'test'
        image_size: int = 32,
        root: str = './data',
        cfg_dropout: float = 0.1,   # probability of replacing label with NULL_CLASS
    ):
        self.cfg_dropout = cfg_dropout

        use_fashion = (dataset_name == 'fashion_mnist')
        cls = datasets.FashionMNIST if use_fashion else datasets.MNIST

        self.data = cls(
            root=root,
            train=(split == 'train'),
            download=True,
        )

        # Resize 28×28 → image_size×image_size, convert to float [0, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),          # (1, H, W) in [0, 1]
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        pil_image, label = self.data[idx]

        image = self.transform(pil_image)   # (1, H, W) in [0, 1]
        image = image.repeat(3, 1, 1)       # (3, H, W) grayscale → pseudo-RGB
        image = image * 2.0 - 1.0          # [0, 1] → [-1, 1]
        image = image.permute(1, 2, 0)     # (3, H, W) → (H, W, 3)  ← b h w c

        # Classifier-free guidance dropout: randomly replace label with null class
        if self.cfg_dropout > 0 and torch.rand(1).item() < self.cfg_dropout:
            label = NULL_CLASS

        return {
            'image':       image.contiguous(),
            'class_label': torch.tensor(label, dtype=torch.long),
        }


# ── Lightning DataModule ───────────────────────────────────────────────────────

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, image_size: int, root: str,
                 batch_size: int, num_workers: int, cfg_dropout: float):
        super().__init__()
        self.dataset_name = dataset_name
        self.image_size   = image_size
        self.root         = root
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.cfg_dropout  = cfg_dropout

    def setup(self, stage=None):
        self.train_ds = DDPMDataset(
            self.dataset_name, 'train', self.image_size, self.root, self.cfg_dropout
        )
        self.val_ds = DDPMDataset(
            self.dataset_name, 'test', self.image_size, self.root, cfg_dropout=0.0
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=(self.num_workers > 0)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=(self.num_workers > 0)
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Train a DDPM on MNIST or FashionMNIST')

    p.add_argument('--dataset', choices=['mnist', 'fashion_mnist'], default='mnist',
                   help='Which dataset to train on (default: mnist)')
    p.add_argument('--cfg_path', default='mnist_ldce/ddpm_mnist.yaml',
                   help='Path to the model architecture YAML (default: mnist_ldce/ddpm_mnist.yaml)')
    p.add_argument('--output_dir', default='./checkpoints/ddpm',
                   help='Directory where checkpoints are saved')
    p.add_argument('--data_root', default='./data',
                   help='Root directory for dataset download (default: ./data)')

    # Training hyper-parameters
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=2e-5,
                   help='Learning rate (overrides the value in the yaml)')
    p.add_argument('--image_size',  type=int,   default=32,
                   help='Spatial resolution fed to the UNet (default: 32)')
    p.add_argument('--cfg_dropout', type=float, default=0.1,
                   help='Fraction of labels replaced with null class during training '
                        'to enable classifier-free guidance at inference (default: 0.1)')
    p.add_argument('--num_workers', type=int,   default=4)

    # Checkpointing / resuming
    p.add_argument('--resume_ckpt', default=None,
                   help='Path to a .ckpt file to resume training from')
    p.add_argument('--save_every_n_epochs', type=int, default=5,
                   help='Save a checkpoint every N epochs (default: 5)')

    # Hardware
    p.add_argument('--gpus', type=int, default=1,
                   help='Number of GPUs to use (0 = CPU, default: 1)')
    p.add_argument('--precision', choices=['32', '16', 'bf16'], default='32',
                   help='Floating-point precision (default: 32)')

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model from YAML ───────────────────────────────────────────────────
    print(f"Loading model architecture from: {args.cfg_path}")
    config = OmegaConf.load(args.cfg_path)

    # Override the learning rate from the command line
    config.model.base_learning_rate = args.lr

    model = instantiate_from_config(config.model)

    if args.resume_ckpt:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
        if missing:
            print(f"  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # ── Data ──────────────────────────────────────────────────────────────────
    datamodule = MNISTDataModule(
        dataset_name=args.dataset,
        image_size=args.image_size,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cfg_dropout=args.cfg_dropout,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f'{args.dataset}_ddpm_{{epoch:04d}}',
        save_top_k=-1,                        # keep all periodic checkpoints
        every_n_epochs=args.save_every_n_epochs,
        save_last=True,                       # always keep last.ckpt
        verbose=True,
    )
    best_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f'{args.dataset}_ddpm_best',
        monitor='val/loss_simple_ema',
        mode='min',
        save_top_k=1,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # ── Trainer ───────────────────────────────────────────────────────────────
    accelerator = 'gpu' if args.gpus > 0 else 'cpu'
    devices     = args.gpus if args.gpus > 0 else 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=args.precision,
        callbacks=[checkpoint_cb, best_cb, lr_monitor],
        default_root_dir=args.output_dir,
        log_every_n_steps=50,
        val_check_interval=1.0,           # validate once per epoch
    )

    dataset_label = 'FashionMNIST' if args.dataset == 'fashion_mnist' else 'MNIST'
    print(f"\nTraining on {dataset_label} for {args.epochs} epochs")
    print(f"  batch_size  = {args.batch_size}")
    print(f"  image_size  = {args.image_size}")
    print(f"  cfg_dropout = {args.cfg_dropout}")
    print(f"  output_dir  = {args.output_dir}\n")

    trainer.fit(model, datamodule=datamodule)

    print(f"\nTraining complete.  Best checkpoint: {best_cb.best_model_path}")
    print(f"Last checkpoint   : {checkpoint_cb.last_model_path}")
    print(
        f"\nTo use a checkpoint with the LDCE pipeline, update config.yaml:\n"
        f"  diffusion_model:\n"
        f"    ckpt_path: \"{checkpoint_cb.last_model_path}\"\n"
    )


if __name__ == '__main__':
    main()
