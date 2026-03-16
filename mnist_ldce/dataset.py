"""
MNIST dataset wrapper compatible with the LDCE pipeline.

Returns (image, label, unique_idx) tuples, just like ImageNet/CelebA datasets
do in data/datasets.py, so the main run script can process them uniformly.

Images are 3-channel [0, 1] tensors (grayscale channel repeated × 3) resized
to `image_size` × `image_size` pixels.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# Human-readable names for the 10 MNIST digit classes
DIGIT_NAMES = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
}

# For each digit, an ordered list of visually similar alternative digits.
# The first element is used as the default counterfactual target class.
# Derived from typical CNN confusion patterns on MNIST.
MNIST_CLOSEST_CLASS = {
    0: [6, 8, 9],
    1: [7, 4, 2],
    2: [7, 3, 1],
    3: [8, 9, 5],
    4: [9, 1, 7],
    5: [6, 3, 8],
    6: [0, 8, 5],
    7: [1, 2, 4],
    8: [9, 0, 3],
    9: [4, 0, 8],
}


class MNISTForLDCE(Dataset):
    """MNIST test/train split returning (image, label, unique_idx) triples.

    Args:
        root       : Directory where MNIST is (or will be) downloaded.
        split      : 'test' (10 000 images) or 'train' (60 000 images).
        image_size : Square output resolution. MNIST 28 × 28 is resized here.
                     Must be a multiple of 32 for the UNet (32, 64, …).
        restart_idx: Skip the first `restart_idx` dataset entries (resume support).
    """

    def __init__(self, root: str = './data', split: str = 'test',
                 image_size: int = 32, restart_idx: int = 0):
        self.mnist = datasets.MNIST(
            root=root, train=(split == 'train'), download=True
        )
        self.restart_idx = restart_idx
        self.indices = list(range(restart_idx, len(self.mnist)))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),          # → (1, H, W)  in [0, 1]
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        actual_idx = self.indices[idx]
        pil_image, label = self.mnist[actual_idx]

        image = self.transform(pil_image)   # (1, H, W)
        image = image.repeat(3, 1, 1)       # (3, H, W)  – grayscale → pseudo-RGB

        return image, label, actual_idx
