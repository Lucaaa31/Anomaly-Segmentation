"""Step 4 - Compare EoMT checkpoints on Cityscapes val (semantic mIoU + qualitative viz).

Evaluates two pretrained EoMT models:
  - Cityscapes-trained (semantic, 19 classes) -> direct mIoU.
  - COCO-trained (panoptic, 133 classes) -> remapped to Cityscapes train_ids before mIoU.

The COCO->Cityscapes class map is the bridge that lets us run both models through the
same evaluation pipeline. Predictions whose COCO continuous id has no Cityscapes
counterpart are treated as ignore (they do not contribute to mIoU). This is the
"strict" choice: the COCO model gets no credit for unmappable predictions, which is
honest about what semantic content is shared between the two label spaces.

Usage (run from repo root):s

    python utils/eval_semantic.py --data-path /path/with/cityscapes_zips
    python utils/eval_semantic.py --data-path ... --model cs --num-vis 8
    python utils/eval_semantic.py --data-path ... --model coco
"""

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp.autocast_mode import autocast


REPO_ROOT = Path(__file__).resolve().parent.parent
EOMT_DIR = REPO_ROOT / "eomt"
sys.path.insert(0, str(EOMT_DIR))

from torchmetrics.classification import MulticlassJaccardIndex  # noqa: E402

NUM_CS_CLASSES = 19
IGNORE_INDEX = 255

CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

# COCO panoptic continuous id -> Cityscapes train_id.
# Continuous ids come from datasets.coco_panoptic.CLASS_MAPPING (orig COCO id -> 0..132):
# things 0..79, stuff 80..132. Anything not listed here is unmappable -> ignore.
# Notes: "rider" and "pole" have no COCO equivalent. "traffic sign" is mapped from
# stop sign (the closest COCO category). "terrain" is approximated as grass/dirt/mountain.
COCO_TO_CS = {
    # things
    0: 11,    # person
    1: 18,    # bicycle
    2: 13,    # car
    3: 17,    # motorcycle
    5: 15,    # bus
    6: 16,    # train
    7: 14,    # truck
    9: 6,     # traffic light
    11: 7,    # stop sign       -> traffic sign  (loose)
    # stuff
    100: 0,   # road
    109: 3,   # wall-brick      -> wall
    110: 3,   # wall-stone      -> wall
    111: 3,   # wall-tile       -> wall
    112: 3,   # wall-wood       -> wall
    116: 8,   # tree-merged     -> vegetation
    117: 4,   # fence-merged    -> fence
    119: 10,  # sky-other-merged
    123: 1,   # pavement-merged -> sidewalk
    124: 9,   # mountain-merged -> terrain (loose)
    125: 9,   # grass-merged    -> terrain (loose)
    126: 9,   # dirt-merged     -> terrain (loose)
    129: 2,   # building-other-merged -> building
    131: 3,   # wall-other-merged     -> wall
}


def build_coco_to_cs_lut(num_coco_classes, device):
    """[num_coco_classes] long tensor: index = COCO continuous id, value = CS train_id (or 255)."""
    lut = torch.full((num_coco_classes,), IGNORE_INDEX, dtype=torch.long, device=device)
    for c, t in COCO_TO_CS.items():
        if c < num_coco_classes:
            lut[c] = t
    return lut


def build_model_and_data(config_path, ckpt_path, data_path, device, setup_data=True):
    """Mirror eomt/inference.ipynb: build encoder, network, lit module from YAML config,
    then load the .bin checkpoint into the lit module.

    If setup_data is False, the data module is returned without .setup() — useful when
    only metadata (num_classes, img_size, stuff_classes) is needed and the underlying
    zips are not available (e.g. building the COCO model on a Cityscapes-only path)."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # data module
    dm_path = config["data"]["class_path"]
    dm_mod, dm_cls = dm_path.rsplit(".", 1)
    DataCls = getattr(importlib.import_module(dm_mod), dm_cls)
    data = DataCls(
        path=str(data_path),
        batch_size=1,
        num_workers=0,
        check_empty_targets=False,
        **config["data"].get("init_args", {}),
    )
    if setup_data:
        data = data.setup()

    # encoder
    enc_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    em, ec = enc_cfg["class_path"].rsplit(".", 1)
    EncCls = getattr(importlib.import_module(em), ec)
    encoder = EncCls(img_size=data.img_size, **enc_cfg.get("init_args", {}))

    # network (masked_attn_enabled=False matches inference.ipynb)
    net_cfg = config["model"]["init_args"]["network"]
    nm, nc = net_cfg["class_path"].rsplit(".", 1)
    NetCls = getattr(importlib.import_module(nm), nc)
    net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
    network = NetCls(
        masked_attn_enabled=False,
        num_classes=data.num_classes,
        encoder=encoder,
        **net_kwargs,
    )

    # lightning module
    lm, lc = config["model"]["class_path"].rsplit(".", 1)
    LitCls = getattr(importlib.import_module(lm), lc)
    model_kwargs = {
        k: v for k, v in config["model"]["init_args"].items() if k != "network"
    }
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = LitCls(
        img_size=data.img_size,
        num_classes=data.num_classes,
        network=network,
        **model_kwargs,
    ).eval().to(device)

    # weights
    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    state_dict = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"  [warn] missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"  [warn] unexpected keys: {len(incompatible.unexpected_keys)}")

    return model, data


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


def per_pixel_target(target_dict, ignore_idx=IGNORE_INDEX):
    """Build [H, W] CS train_id map from a Cityscapes val target dict (masks + labels)."""
    masks, labels = target_dict["masks"], target_dict["labels"]
    out = torch.full(masks.shape[-2:], ignore_idx, dtype=torch.long, device=masks.device)
    for i, m in enumerate(masks):
        out[m] = labels[i].long()
    return out


def evaluate_semantic(
    model, dataset, device, lut=None, max_images=-1,
    num_vis=0, vis_dir=None, vis_prefix="vis", log_every=50,
):
    """Compute per-class IoU on `dataset` using `model`.

    If `lut` is provided, raw model predictions are remapped through it
    (COCO continuous id -> Cityscapes train_id). Pixels whose prediction is
    unmappable (lut value == IGNORE_INDEX) are excluded from the metric by
    masking the target to ignore. Returns a [NUM_CS_CLASSES] IoU tensor on CPU.

    If `num_vis > 0` and `vis_dir` is given, saves up to `num_vis` semantic
    visualizations as `{vis_prefix}_{idx:04d}.png` (uses the remapped pred when
    a lut is given, so the saved image is in CS color space)."""
    metric = MulticlassJaccardIndex(
        num_classes=NUM_CS_CLASSES, ignore_index=IGNORE_INDEX, average=None
    ).to(device)
    n = len(dataset) if max_images <= 0 else min(max_images, len(dataset))
    saved = 0
    for i in range(n):
        img, tgt = dataset[i]
        target = per_pixel_target(tgt).to(device)
        logits = infer_semantic_logits(model, img, device)
        pred = logits.argmax(0)

        if lut is not None:
            pred_cs = lut[pred]
            unmapped = pred_cs.eq(IGNORE_INDEX)
            target_for_metric = torch.where(unmapped, torch.full_like(target, IGNORE_INDEX), target)
            pred_for_metric = torch.where(unmapped, torch.zeros_like(pred_cs), pred_cs)
        else:
            target_for_metric = target
            pred_for_metric = pred

        metric.update(pred_for_metric[None], target_for_metric[None])

        if saved < num_vis and vis_dir is not None:
            save_semantic_vis(
                img, pred_for_metric.cpu().numpy(), target.cpu().numpy(),
                vis_dir / f"{vis_prefix}_{i:04d}.png",
            )
            saved += 1

        if log_every and (i + 1) % log_every == 0:
            print(f"  {vis_prefix} {i+1}/{n}")

    return metric.compute().cpu()


def evaluate_cs_model(args, device, vis_dir):
    print(f"[cs] loading {args.cs_config}")
    model, data = build_model_and_data(
        Path(args.cs_config), Path(args.cs_ckpt), Path(args.data_path), device
    )
    dataset = data.val_dataloader().dataset
    return evaluate_semantic(
        model, dataset, device,
        max_images=args.max_images,
        num_vis=args.num_vis, vis_dir=vis_dir, vis_prefix="cs",
    )


def evaluate_coco_model(args, device, vis_dir):
    print(f"[coco] loading {args.coco_config}")
    model, data = build_model_and_data(
        Path(args.coco_config), Path(args.coco_ckpt), Path(args.data_path), device,
        setup_data=False,
    )
    print("[coco] loading Cityscapes val for evaluation")
    CityscapesSemantic = getattr(
        importlib.import_module("datasets.cityscapes_semantic"), "CityscapesSemantic"
    )
    cs_data = CityscapesSemantic(
        path=str(args.data_path), batch_size=1, num_workers=0,
        check_empty_targets=False,
    ).setup()
    cs_dataset = cs_data.val_dataloader().dataset

    # Panoptic qualitative samples (separate from the semantic mIoU loop).
    for i in range(min(args.num_vis, len(cs_dataset))):
        img, _ = cs_dataset[i]
        sem, inst = infer_panoptic(model, img, device)
        save_panoptic_vis(img, sem, inst, data.num_classes, vis_dir / f"coco_panoptic_{i:04d}.png")

    lut = build_coco_to_cs_lut(data.num_classes, device)
    return evaluate_semantic(
        model, cs_dataset, device, lut=lut,
        max_images=args.max_images,
        num_vis=args.num_vis, vis_dir=vis_dir, vis_prefix="coco_remap",
    )


def print_iou_table(name, iou_per_class):
    print(f"\n=== {name} ===")
    print(f"{'class':<16} {'IoU':>7}")
    print("-" * 25)
    for cname, v in zip(CS_CLASS_NAMES, iou_per_class.tolist()):
        print(f"{cname:<16} {v*100:>6.2f}")
    valid = iou_per_class[~torch.isnan(iou_per_class)]
    print("-" * 25)
    print(f"{'mIoU':<16} {valid.mean().item()*100:>6.2f}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-path", required=True,
                   help="Directory containing leftImg8bit_trainvaltest.zip and gtFine_trainvaltest.zip")
    p.add_argument("--cs-ckpt", default=str(REPO_ROOT / "models" / "eomt_cityscapes.bin"))
    p.add_argument("--coco-ckpt", default=str(REPO_ROOT / "models" / "eomt_coco.bin"))
    p.add_argument("--cs-config",
                   default=str(EOMT_DIR / "configs" / "dinov2" / "cityscapes" / "semantic" / "eomt_base_640.yaml"))
    p.add_argument("--coco-config",
                   default=str(EOMT_DIR / "configs" / "dinov2" / "coco" / "panoptic" / "eomt_base_640_2x.yaml"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", choices=["both", "cs", "coco"], default="both")
    p.add_argument("--num-vis", type=int, default=4, help="Qualitative samples to save per model")
    p.add_argument("--max-images", type=int, default=-1, help="Cap eval to N images (default: all val)")
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "step4"))
    args = p.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu")
    print(f"device: {device}")

    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    if args.model in ("both", "cs"):
        results["cs"] = evaluate_cs_model(args, device, vis_dir)
        print_iou_table("EoMT Cityscapes-trained (semantic) on Cityscapes val", results["cs"])
    if args.model in ("both", "coco"):
        results["coco"] = evaluate_coco_model(args, device, vis_dir)
        print_iou_table("EoMT COCO-trained (panoptic) on Cityscapes val (remapped)", results["coco"])

    summary = out_dir / "step4_miou.txt"
    with open(summary, "w") as f:
        for name, iou in results.items():
            f.write(f"{name}\n")
            for c, v in zip(CS_CLASS_NAMES, iou.tolist()):
                f.write(f"  {c:<16} {v*100:.2f}\n")
            valid = iou[~torch.isnan(iou)]
            f.write(f"  {'mIoU':<16} {valid.mean().item()*100:.2f}\n\n")
    print(f"\nSummary written to {summary}")
    print(f"Visualizations in    {vis_dir}")


if __name__ == "__main__":
    main()
