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


def select_class_balanced(mnist_dataset, classes, total: int) -> list:
    """Return up to `total` dataset indices balanced across `classes`.

    Scans the full split once and picks floor(total / n_classes) examples per
    class (plus extras from the first classes if total is not divisible), so
    the returned list is as class-balanced as possible while respecting the
    requested total count.

    Args:
        mnist_dataset : torchvision MNIST dataset object.
        classes       : List of digit classes to include (e.g. [0, 3, 4, 5, 9]).
        total         : Maximum total number of indices to return.

    Returns:
        Sorted list of dataset indices.
    """
    classes = list(classes)
    per_class = total // len(classes)
    remainder = total % len(classes)

    buckets = {c: [] for c in classes}
    for idx in range(len(mnist_dataset)):
        _, label = mnist_dataset[idx]
        if label in buckets and len(buckets[label]) < per_class + (1 if classes.index(label) < remainder else 0):
            buckets[label].append(idx)
        if all(
            len(v) >= per_class + (1 if classes.index(k) < remainder else 0)
            for k, v in buckets.items()
        ):
            break

    indices = sorted(idx for bucket in buckets.values() for idx in bucket)
    return indices


class MNISTForLDCE(Dataset):
    """MNIST test/train split returning (image, label, unique_idx) triples.

    Args:
        root          : Directory where MNIST is (or will be) downloaded.
        split         : 'test' (10 000 images) or 'train' (60 000 images).
        image_size    : Square output resolution. MNIST 28 × 28 is resized here.
                        Must be a multiple of 32 for the UNet (32, 64, …).
        restart_idx   : Skip the first `restart_idx` dataset entries (resume support).
        filter_classes: If given, only include images whose label is in this list.
                        Combined with max_samples for class-balanced selection.
        skip_ids      : Set of MNIST dataset indices to exclude (already completed).
    """

    def __init__(self, root: str = './data', split: str = 'test',
                 image_size: int = 32, restart_idx: int = 0,
                 max_samples: int = None, filter_classes: list = None,
                 skip_ids: set = None):
        self.mnist = datasets.MNIST(
            root=root, train=(split == 'train'), download=True
        )
        self.restart_idx = restart_idx
        skip_ids = skip_ids or set()

        if filter_classes is not None:
            total = max_samples if max_samples is not None else len(self.mnist)
            all_indices = select_class_balanced(self.mnist, filter_classes, total)
            self.indices = [i for i in all_indices if i >= restart_idx and i not in skip_ids]
        else:
            self.indices = [i for i in range(restart_idx, len(self.mnist))
                            if i not in skip_ids]
            if max_samples is not None:
                self.indices = self.indices[:max_samples]

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
