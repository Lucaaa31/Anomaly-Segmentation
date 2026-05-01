"""Per-image EoMT inference helpers (semantic and panoptic).

Both helpers run on a single image (CHW tensor) and return logits / preds aligned to
the original image size, using the model's own windowing / padding utilities defined
in `eomt/training/lightning_module.py`. They are thin wrappers — anything that can be
shared between the eval pipeline and other downstream tasks (e.g. anomaly scoring on
saved logits) lives here.
"""

import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast


def _autocast_ctx(device):
    if device.type == "cuda":
        return autocast(dtype=torch.float16, device_type="cuda")
    return autocast(dtype=torch.float32, device_type="cpu", enabled=False)


@torch.no_grad()
def infer_semantic_logits(model, img, device):
    """Windowed semantic inference. Returns per-pixel logits [num_classes, H, W] on `device`.
    Works for both semantic-trained and panoptic-trained EoMT (mask-classification head)."""
    imgs = [img.to(device)]
    img_sizes = [img.shape[-2:]]
    crops, origins = model.window_imgs_semantic(imgs)

    with _autocast_ctx(device):
        mask_logits_per_layer, class_logits_per_layer = model(crops)

    mask_logits = F.interpolate(
        mask_logits_per_layer[-1], model.img_size, mode="bilinear"
    )
    crop_logits = model.to_per_pixel_logits_semantic(
        mask_logits, class_logits_per_layer[-1]
    )
    return model.revert_window_logits_semantic(crop_logits, origins, img_sizes)[0]


@torch.no_grad()
def infer_panoptic(model, img, device):
    """Panoptic inference (semantic + instance ids). Returns numpy arrays sem[H,W], inst[H,W]."""
    imgs = [img.to(device)]
    img_sizes = [img.shape[-2:]]
    transformed = model.resize_and_pad_imgs_instance_panoptic(imgs)

    with _autocast_ctx(device):
        mask_logits_per_layer, class_logits_per_layer = model(transformed)

    mask_logits = F.interpolate(
        mask_logits_per_layer[-1], model.img_size, mode="bilinear"
    )
    mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
        mask_logits, img_sizes
    )
    preds = model.to_per_pixel_preds_panoptic(
        mask_logits,
        class_logits_per_layer[-1],
        model.stuff_classes,
        model.mask_thresh,
        model.overlap_thresh,
    )[0].cpu().numpy()
    return preds[..., 0], preds[..., 1]
