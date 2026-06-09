"""Evaluate all 5 models on the FULL Cityscapes val split (500 imgs) and cache
per-class strict IoU to results/step4_results.txt.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Bootstrap so `python utils/eval_all_strict.py` resolves the `utils.X` imports.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing utils.build also puts models/eomt on sys.path (so `datasets.*`,
# `models.*`, `training.*` YAML class paths resolve).
from utils.build import build_model_and_data
from utils.class_remap import build_coco_to_cs_lut
from utils.eval_semantic import ( 
    CS_CLASS_NAMES,
    evaluate_semantic,
    per_pixel_target,
)

from PIL import Image  # noqa: E402
from torchvision.transforms import Compose, Normalize, Resize, ToTensor 
from torchvision.transforms.functional import to_pil_image 


# --- default paths (resolve relative to the repo root / Drive mount) ---------
CS_CONFIG = REPO_ROOT / "configs" / "dinov2" / "cityscapes" / "semantic" / "eomt_base_640.yaml"
COCO_CONFIG = REPO_ROOT / "configs" / "dinov2" / "coco" / "panoptic" / "eomt_base_640_2x.yaml"
CS_CKPT = REPO_ROOT / "models" / "eomt_cityscapes.bin"
COCO_CKPT = REPO_ROOT / "models" / "eomt_coco.bin"
PHASED_CKPT = REPO_ROOT / "models" / "coco_finetune" / "phased" / "eomt_finetuned_phased.bin"
FULL_CKPT = REPO_ROOT / "models" / "coco_finetune" / "full" / "eomt_finetuned_full.bin"
ERFNET_WEIGHTS = REPO_ROOT / "models" / "erfnet" / "erfnet_pretrained.pth"
DATA_PATH = REPO_ROOT / "dataset" / "Cityscapes"
OUT_FILE = REPO_ROOT / "results" / "step4_results.txt"

ERFNET_NUM_CLASSES = 20  # 19 Cityscapes train-ids + 1 void/ignore (id 19)
ERFNET_SIZE = (512, 1024)


# --- ERFNet helpers (mirror notebooks/Step7_AnomalyDetection_Erfnet.ipynb) ---

def load_my_state_dict(model, state_dict):
    """Copy weights handling the DataParallel 'module.' prefix (Romera/Step 7)."""
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            if name.startswith("module."):
                own_state[name.split("module.")[-1]].copy_(param)
            else:
                print(f"  [erfnet] not loaded: {name}")
                continue
        else:
            own_state[name].copy_(param)
    return model


def load_erfnet(weights_path, device):
    from models.erfnet.erfnet import ERFNet

    model = ERFNet(ERFNET_NUM_CLASSES)
    if device.type == "cuda":
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    state = torch.load(str(weights_path), map_location=lambda s, l: s)
    model = load_my_state_dict(model, state)
    return model.eval()


@torch.no_grad()
def evaluate_erfnet_strict(model, val_dataset, device, max_images=-1, log_every=50,
                           imagenet_norm=False):
    """Per-class IoU (%) for ERFNet over Cityscapes val using iouEval(20, ignore=19).

    The Romera/course ERFNet eval feeds images in [0,1] (Resize + ToTensor, NO
    ImageNet normalization). Adding ImageNet stats with these weights collapses
    the model (mIoU near 0). Set imagenet_norm=True only if your specific weights
    were trained with ImageNet normalization."""
    from utils.iouEval import iouEval

    steps = [Resize(ERFNET_SIZE, Image.BILINEAR), ToTensor()]
    if imagenet_norm:
        steps.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    tf = Compose(steps)
    iou = iouEval(ERFNET_NUM_CLASSES, ignoreIndex=19)

    n = len(val_dataset) if max_images <= 0 else min(max_images, len(val_dataset))
    for i in range(n):
        img, tgt = val_dataset[i]                       
        x = tf(to_pil_image(img)).unsqueeze(0).to(device)
        pred = model(x).argmax(1, keepdim=True)        

        gt = per_pixel_target(tgt).to(device)           
        gt = F.interpolate(
            gt[None, None].float(), size=ERFNET_SIZE, mode="nearest"
        )[0].long()                                     
        gt[gt == 255] = 19                              

        iou.addBatch(pred, gt[None])                    
        if log_every and (i + 1) % log_every == 0:
            print(f"  erfnet {i + 1}/{n}")

    mean_iou, per_class = iou.getIoU()                 
    return (per_class * 100).tolist(), mean_iou.item() * 100


# --- EoMT eval ---------------------------------------------------------------

def evaluate_eomt_strict(config_path, ckpt_path, val_dataset, data_path, device,
                         use_lut=False, max_images=-1, img_size=None):
    """Per-class IoU (%) for an EoMT checkpoint, strict label space.

    img_size overrides the encoder build size. The fine-tuned checkpoints were
    trained at 640x640, so their pos_embed has a 40x40 grid and must be loaded
    into a model built at 640; the provided checkpoints use the config default."""
    overrides = {"img_size": tuple(img_size)} if img_size else None
    model, data = build_model_and_data(
        Path(config_path), Path(ckpt_path), Path(data_path), device,
        setup_data=False, data_overrides=overrides,
    )
    lut = build_coco_to_cs_lut(data.num_classes, device) if use_lut else None
    iou = evaluate_semantic(
        model, val_dataset, device, lut=lut, label_space="strict",
        max_images=max_images,
    )                                                    # [19] fractions, no NaN in strict
    per_class = (iou * 100).tolist()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return per_class, float(np.nanmean(per_class))


# --- model registry ----------------------------------------------------------
# (display_name, kind, kwargs)  -- order is the plot order.

def model_registry(args):
    return [
        ("ERFNet", "erfnet", {"weights": args.erfnet_weights}),
        ("EoMT-COCO", "eomt", {"config": args.coco_config, "ckpt": args.coco_ckpt, "use_lut": True}),
        ("EoMT-CS", "eomt", {"config": args.cs_config, "ckpt": args.cs_ckpt, "use_lut": False}),
        ("EoMT-COCO-FT-phased", "eomt", {"config": args.cs_config, "ckpt": args.phased_ckpt, "use_lut": False, "img_size": (640, 640)}),
        ("EoMT-COCO-FT-full", "eomt", {"config": args.cs_config, "ckpt": args.full_ckpt, "use_lut": False, "img_size": (640, 640)}),
    ]


# --- output file -------------------------------------------------------------

def write_header(fh):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    fh.write(f"# Section 3.1 - per-class strict IoU on Cityscapes val (500 imgs)\n")
    fh.write(f"# generated: {ts}\n")
    fh.write(f"# label_space: strict | values in percent (%) | mIoU = mean of the 19 classes\n")
    fh.write(f"# eval: EoMT -> torchmetrics @ native res ; ERFNet -> iouEval @ 512x1024\n")
    fh.write("model," + ",".join(CS_CLASS_NAMES) + ",mIoU\n")
    fh.flush()


def write_row(fh, name, per_class, miou):
    vals = ",".join(f"{v:.2f}" for v in per_class)
    fh.write(f"{name},{vals},{miou:.2f}\n")
    fh.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", default=str(DATA_PATH))
    p.add_argument("--out", default=str(OUT_FILE))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-images", type=int, default=-1,
                   help="Cap eval to N images (default: all 500). Use a small N for a smoke test.")
    p.add_argument("--only", nargs="*", default=None,
                   help="Evaluate only these model names (default: all 5).")
    p.add_argument("--cs-config", default=str(CS_CONFIG))
    p.add_argument("--coco-config", default=str(COCO_CONFIG))
    p.add_argument("--cs-ckpt", default=str(CS_CKPT))
    p.add_argument("--coco-ckpt", default=str(COCO_CKPT))
    p.add_argument("--phased-ckpt", default=str(PHASED_CKPT))
    p.add_argument("--full-ckpt", default=str(FULL_CKPT))
    p.add_argument("--erfnet-weights", default=str(ERFNET_WEIGHTS))
    p.add_argument("--erfnet-imagenet-norm", action="store_true",
                   help="Apply ImageNet normalization to ERFNet inputs (default off: "
                        "Romera/course weights expect inputs in [0,1]).")
    args = p.parse_args()

    device = torch.device(
        args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu"
    )
    print(f"device: {device}")

    print("Loading Cityscapes val (500 imgs, shared by all models)")
    from datasets.cityscapes_semantic import CityscapesSemantic
    val_dataset = CityscapesSemantic(
        path=str(args.data_path), batch_size=1, num_workers=0, check_empty_targets=False,
    ).setup().val_dataloader().dataset
    print(f"  {len(val_dataset)} images")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    registry = model_registry(args)
    if args.only:
        registry = [m for m in registry if m[0] in set(args.only)]

    summary = {}
    with open(out_path, "w") as fh:        # overwrite: this file is the cache
        write_header(fh)
        for name, kind, kw in registry:
            print(f"\n=== {name} ({kind}) ===")
            t0 = time.time()
            try:
                if kind == "erfnet":
                    if not Path(kw["weights"]).exists():
                        raise FileNotFoundError(kw["weights"])
                    model = load_erfnet(kw["weights"], device)
                    per_class, miou = evaluate_erfnet_strict(
                        model, val_dataset, device, max_images=args.max_images,
                        imagenet_norm=args.erfnet_imagenet_norm,
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    if not Path(kw["ckpt"]).exists():
                        raise FileNotFoundError(kw["ckpt"])
                    per_class, miou = evaluate_eomt_strict(
                        kw["config"], kw["ckpt"], val_dataset, args.data_path, device,
                        use_lut=kw["use_lut"], max_images=args.max_images,
                        img_size=kw.get("img_size"),
                    )
            except Exception as e:         # keep going; cache whatever succeeds
                print(f"  [SKIP] {name}: {type(e).__name__}: {e}")
                fh.write(f"# {name}: FAILED ({type(e).__name__}: {e})\n")
                fh.flush()
                continue

            write_row(fh, name, per_class, miou)
            summary[name] = miou
            print(f"  mIoU(strict) = {miou:.2f}  ({time.time() - t0:.0f}s)")

    print("\n--- summary (strict mIoU) ---")
    for name, _, _ in registry:
        print(f"  {name:<22} {summary.get(name, float('nan')):.2f}")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()