import torch


@torch.jit.script
def sigmoid_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x * torch.log(x + 1e-30) + (1 - x) * torch.log(1 - x + 1e-30))


def normalize_img_to_0255(img):
    return (img - img.min()) / (img.max() - img.min()) * 255
