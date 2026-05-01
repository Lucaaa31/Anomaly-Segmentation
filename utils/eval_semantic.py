"""Step 4 - Compare EoMT checkpoints on Cityscapes val (semantic mIoU + qualitative viz).

Evaluates EoMT models (CS-trained, COCO-trained, COCO->CS fine-tuned) on the SAME
semantic mIoU pipeline. The COCO->CS class map (in `utils.class_remap`) is the bridge
that lets all three be compared on the same metric: predictions whose COCO continuous
id has no Cityscapes counterpart are treated as ignore (do not contribute to mIoU).

This module is intentionally narrow — it owns only the things that touch the mIoU
metric itself (per-pixel target conversion, the per-class IoU loop, the printable
table, and the CLI orchestration). Model / data construction lives in `utils.build`,
inference primitives in `utils.inference`, COCO->CS remap in `utils.class_remap`,
and visualizations in `utils.visualize`.

A `label_space` flag selects between "strict" (full 19 CS classes; default) and
"common" (drop pole and traffic sign from the GT, merge rider into person — the
intersection of the COCO and CS class spaces, useful as a fairer cross-dataset
comparison reported on a reduced set of classes).

Usage (run from repo root):

    python utils/eval_semantic.py --data-path /path/with/cityscapes_zips
    python utils/eval_semantic.py --data-path ... --model cs --num-vis 8
    python utils/eval_semantic.py --data-path ... --model coco
    python utils/eval_semantic.py --data-path ... --label-space common
"""

import argparse
import sys
from pathlib import Path

import torch

# Bootstrap so `python utils/eval_semantic.py ...` (run as a script) can resolve
# the `utils.X` absolute imports below. When imported as a library (e.g. from a
# notebook), the caller has already put REPO_ROOT on sys.path and this is a no-op.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchmetrics.classification import MulticlassJaccardIndex  # noqa: E402

from utils.build import build_model_and_data  # noqa: E402
from utils.class_remap import (  # noqa: E402
    build_coco_to_cs_lut,
    common_pred_remap,
    common_target_remap,
    mask_inactive_to_nan,
)
from utils.constants import IGNORE_INDEX, NUM_CS_CLASSES  # noqa: E402
from utils.inference import infer_panoptic, infer_semantic_logits  # noqa: E402
from utils.visualize import save_panoptic_vis, save_semantic_vis  # noqa: E402

EOMT_DIR = REPO_ROOT / "eomt"

CS_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]


def per_pixel_target(target_dict, ignore_idx=IGNORE_INDEX):
    """Build [H, W] CS train_id map from a Cityscapes val target dict (masks + labels)."""
    masks, labels = target_dict["masks"], target_dict["labels"]
    out = torch.full(masks.shape[-2:], ignore_idx, dtype=torch.long, device=masks.device)
    for i, m in enumerate(masks):
        out[m] = labels[i].long()
    return out


def evaluate_semantic(
    model, dataset, device, lut=None, label_space="strict", max_images=-1,
    num_vis=0, vis_dir=None, vis_prefix="vis", log_every=50,
):
    """Compute per-class IoU on `dataset` using `model`.

    If `lut` is provided, raw model predictions are remapped through it
    (COCO continuous id -> Cityscapes train_id). Pixels whose prediction is
    unmappable (lut value == IGNORE_INDEX) are excluded from the metric by
    masking the target to ignore.

    `label_space`:
        - "strict" (default): full 19 CS classes in the GT. Fair to the
          CS-trained model; harsh on the COCO-trained one because it cannot
          express rider/pole and only partially expresses traffic sign.
        - "common": GT is reduced to the intersection of the COCO and CS class
          spaces (pole and traffic sign -> ignore, rider -> person). Predictions
          are also rider->person remapped so a correct rider prediction counts
          as person. Drop classes appear as NaN in the per-class result and are
          skipped from the mean. Fairer cross-dataset comparison, but reported
          over fewer classes.
        - tuple/list of the above (e.g. ("strict","common")) -> compute both in
          one pass over the dataset and return a {space: tensor} dict.

    Returns a [NUM_CS_CLASSES] IoU tensor on CPU when `label_space` is a string,
    or a dict {space: tensor} when it is a sequence.

    If `num_vis > 0` and `vis_dir` is given, saves up to `num_vis` semantic
    visualizations as `{vis_prefix}_{idx:04d}.png` (uses the first label space
    in the list for the saved pred/target — typically 'strict')."""
    multi = not isinstance(label_space, str)
    spaces = tuple(label_space) if multi else (label_space,)
    for s in spaces:
        assert s in ("strict", "common"), f"unknown label_space: {s}"

    metrics = {
        s: MulticlassJaccardIndex(
            num_classes=NUM_CS_CLASSES, ignore_index=IGNORE_INDEX, average=None
        ).to(device)
        for s in spaces
    }
    n = len(dataset) if max_images <= 0 else min(max_images, len(dataset))
    saved = 0
    for i in range(n):
        img, tgt = dataset[i]
        target_orig = per_pixel_target(tgt).to(device)
        logits = infer_semantic_logits(model, img, device)
        pred_orig = logits.argmax(0)

        if lut is not None:
            pred_cs = lut[pred_orig]
            unmapped = pred_cs.eq(IGNORE_INDEX)

        vis_pred, vis_target = None, None
        for space in spaces:
            target = common_target_remap(target_orig) if space == "common" else target_orig

            if lut is not None:
                target_for_metric = torch.where(
                    unmapped, torch.full_like(target, IGNORE_INDEX), target
                )
                pred_for_metric = torch.where(unmapped, torch.zeros_like(pred_cs), pred_cs)
            else:
                target_for_metric = target
                pred_for_metric = pred_orig

            if space == "common":
                pred_for_metric = common_pred_remap(pred_for_metric)

            metrics[space].update(pred_for_metric[None], target_for_metric[None])
            if vis_pred is None:
                vis_pred, vis_target = pred_for_metric, target

        if saved < num_vis and vis_dir is not None:
            save_semantic_vis(
                img, vis_pred.cpu().numpy(), vis_target.cpu().numpy(),
                vis_dir / f"{vis_prefix}_{i:04d}.png",
            )
            saved += 1

        if log_every and (i + 1) % log_every == 0:
            print(f"  {vis_prefix} {i+1}/{n}")

    results = {s: mask_inactive_to_nan(metrics[s].compute().cpu(), s) for s in spaces}
    if multi:
        return results
    return results[spaces[0]]


def print_iou_table(name, iou_per_class):
    print(f"\n=== {name} ===")
    print(f"{'class':<16} {'IoU':>7}")
    print("-" * 25)
    for cname, v in zip(CS_CLASS_NAMES, iou_per_class.tolist()):
        if v != v:  # NaN: class dropped from this label space
            print(f"{cname:<16} {'—':>6}")
        else:
            print(f"{cname:<16} {v*100:>6.2f}")
    valid = iou_per_class[~torch.isnan(iou_per_class)]
    print("-" * 25)
    print(f"{'mIoU':<16} {valid.mean().item()*100:>6.2f}  ({valid.numel()} cls)")


# --- CLI orchestration -------------------------------------------------------

def evaluate_cs_model(args, device, vis_dir):
    print(f"[cs] loading {args.cs_config}")
    model, data = build_model_and_data(
        Path(args.cs_config), Path(args.cs_ckpt), Path(args.data_path), device
    )
    dataset = data.val_dataloader().dataset
    return evaluate_semantic(
        model, dataset, device,
        label_space=args.label_space,
        max_images=args.max_images,
        num_vis=args.num_vis, vis_dir=vis_dir, vis_prefix="cs",
    )


def evaluate_coco_model(args, device, vis_dir):
    import importlib

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
        label_space=args.label_space,
        max_images=args.max_images,
        num_vis=args.num_vis, vis_dir=vis_dir, vis_prefix="coco_remap",
    )


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
    p.add_argument("--label-space", choices=["strict", "common"], default="strict",
                   help="strict = full 19 CS classes; common = drop pole/traffic-sign and merge "
                        "rider->person (fairer cross-dataset comparison, fewer classes).")
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

    summary = out_dir / f"step4_miou_{args.label_space}.txt"
    with open(summary, "w") as f:
        f.write(f"label_space: {args.label_space}\n\n")
        for name, iou in results.items():
            f.write(f"{name}\n")
            for c, v in zip(CS_CLASS_NAMES, iou.tolist()):
                if v != v:
                    f.write(f"  {c:<16} —\n")
                else:
                    f.write(f"  {c:<16} {v*100:.2f}\n")
            valid = iou[~torch.isnan(iou)]
            f.write(f"  {'mIoU':<16} {valid.mean().item()*100:.2f}  ({valid.numel()} cls)\n\n")
    print(f"\nSummary written to {summary}")
    print(f"Visualizations in    {vis_dir}")


if __name__ == "__main__":
    main()
