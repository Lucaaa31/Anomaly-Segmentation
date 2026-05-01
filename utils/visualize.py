"""Visualization helpers: Cityscapes color palette and semantic / panoptic renderers.

These are pure rendering utilities (no torch model calls). They consume already-computed
predictions / targets and produce numpy RGB arrays or save matplotlib figures to disk.
"""

import matplotlib.pyplot as plt
import numpy as np

from utils.constants import NUM_CS_CLASSES


def cs_color_map():
    """Cityscapes 19-class palette (train_id -> RGB). Index 255 stays black."""
    palette = np.array([
        [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
        [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
        [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
        [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
    ], dtype=np.uint8)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:NUM_CS_CLASSES] = palette
    return lut


def colorize_cs(seg):
    return cs_color_map()[seg.clip(0, 255)]


def save_semantic_vis(img, pred, target, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Image")
    axes[1].imshow(colorize_cs(pred))
    axes[1].set_title("Prediction")
    axes[2].imshow(colorize_cs(target))
    axes[2].set_title("Ground Truth")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def panoptic_to_rgb(sem_pred, inst_pred, num_classes, seed=0):
    """Render a panoptic prediction as an RGB image: a random hue per semantic class,
    with black borders between adjacent segments. Mirrors eomt/inference.ipynb."""
    h, w = sem_pred.shape
    sem_ids = np.unique(sem_pred)
    rng = np.random.default_rng(seed)
    color_for = {
        int(s): np.array([0, 0, 0], dtype=np.uint8) if s == -1 or s == num_classes
        else (np.array(plt.cm.hsv(rng.random())[:3]) * 255).astype(np.uint8)
        for s in sem_ids
    }
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for s in sem_ids:
        out[sem_pred == s] = color_for[int(s)]

    combined = sem_pred.astype(np.int64) * 100000 + inst_pred.astype(np.int64)
    border = np.zeros((h, w), dtype=bool)
    border[1:, :] |= combined[1:, :] != combined[:-1, :]
    border[:-1, :] |= combined[1:, :] != combined[:-1, :]
    border[:, 1:] |= combined[:, 1:] != combined[:, :-1]
    border[:, :-1] |= combined[:, 1:] != combined[:, :-1]
    out[border] = 0
    return out


def save_panoptic_vis(img, sem_pred, inst_pred, num_classes, path):
    """Save side-by-side image + panoptic visualization to disk."""
    out = panoptic_to_rgb(sem_pred, inst_pred, num_classes)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Image")
    axes[1].imshow(out)
    axes[1].set_title("Panoptic prediction (COCO classes)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
