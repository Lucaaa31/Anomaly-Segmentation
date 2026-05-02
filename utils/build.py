"""Build EoMT lightning module + Cityscapes / COCO data module from a YAML config.

Mirrors `eomt/inference.ipynb`: builds encoder + network + lit module via importlib
from the YAML class_paths, then loads a `.bin` checkpoint into the lit module.

Side effect: this module adds `eomt/` to sys.path at import time so the YAML
class_paths (e.g. `models.vit.ViT`, `datasets.cityscapes_semantic.CityscapesSemantic`)
resolve. Importing this module is therefore enough to use those modules elsewhere.
"""

import importlib
import sys
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
EOMT_DIR = REPO_ROOT / "eomt"
if str(EOMT_DIR) not in sys.path:
    sys.path.insert(0, str(EOMT_DIR))


def build_model_and_data(
    config_path, ckpt_path, data_path, device,
    setup_data=True, skip_class_head=False, data_overrides=None,
):
    """Build encoder, network, lit module from YAML config; load `.bin` weights.

    If `setup_data` is False, the data module is returned without `.setup()` — useful
    when only metadata (`num_classes`, `img_size`, `stuff_classes`) is needed and the
    underlying zips are not available (e.g. building the COCO model on a Cityscapes-only
    path).

    If `skip_class_head` is True, the checkpoint is filtered to drop tensors
    that cannot be transferred across class / query spaces:
    - keys containing 'class_head' or 'class_predictor' (different num_classes)
    - any other key whose tensor shape disagrees with the target model
      (e.g. `network.q.weight` when num_queries differs, `criterion.empty_weight`
      whose length depends on num_classes).
    Use this when loading weights from a checkpoint with a different number of
    classes / queries (e.g. COCO panoptic 200q/133cls -> Cityscapes semantic
    100q/19cls fine-tuning).

    `data_overrides` lets callers tweak the data-module init args coming from the YAML
    (e.g. {"img_size": (640, 640), "batch_size": 2, "num_workers": 2}).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # data module
    dm_path = config["data"]["class_path"]
    dm_mod, dm_cls = dm_path.rsplit(".", 1)
    DataCls = getattr(importlib.import_module(dm_mod), dm_cls)
    data_kwargs = {"batch_size": 1, "num_workers": 0, "check_empty_targets": False}
    data_kwargs.update(config["data"].get("init_args", {}))
    data_kwargs.update(data_overrides or {})
    data = DataCls(path=str(data_path), **data_kwargs)
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
    if skip_class_head:
        model_sd = model.state_dict()
        filtered, dropped_class_head, dropped_shape = {}, [], []
        for k, v in state_dict.items():
            if "class_head" in k or "class_predictor" in k:
                dropped_class_head.append(k)
                continue
            if k in model_sd and model_sd[k].shape != v.shape:
                dropped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
            filtered[k] = v
        state_dict = filtered
        if dropped_class_head:
            print(f"  [info] skip_class_head: dropped {len(dropped_class_head)} class-head key(s)")
        if dropped_shape:
            print(f"  [info] skip_class_head: dropped {len(dropped_shape)} shape-mismatched key(s):")
            for k, ck_shape, mdl_shape in dropped_shape:
                print(f"    {k}: ckpt {ck_shape} vs model {mdl_shape}")
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"  [warn] missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"  [warn] unexpected keys: {len(incompatible.unexpected_keys)}")

    return model, data
