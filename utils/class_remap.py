"""COCO panoptic -> Cityscapes semantic class remap (+ optional 'common' label space).

The bridge that lets a COCO-trained EoMT model be evaluated on the Cityscapes mIoU
pipeline. Predictions whose COCO continuous id has no Cityscapes counterpart are
treated as ignore (do not contribute to mIoU). This is the "strict" choice: the
COCO model gets no credit for unmappable predictions.

Continuous ids come from `eomt/datasets/coco_panoptic.py::CLASS_MAPPING` (orig COCO
id -> 0..132): things 0..79, stuff 80..132. Anything not listed here is unmappable.

Notes:
    - "rider" and "pole" have no COCO equivalent.
    - "traffic sign" is mapped from stop sign (the closest COCO category).
    - "terrain" is approximated as grass / dirt / mountain.

This module also exposes a 'common' label space (the intersection of what COCO and
Cityscapes can both express). Under the common space the GT is reduced before the
metric: `pole` and `traffic sign` (which COCO essentially cannot predict) are set
to ignore, and `rider` is merged into `person` (COCO does not distinguish them).
The strict numbers stay the headline result; the common-space numbers are a fairer
cross-dataset comparison reported on a reduced class set.
"""

import torch

from utils.constants import IGNORE_INDEX, NUM_CS_CLASSES


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


# CS classes dropped from the GT under the 'common' label space (COCO has no
# equivalent, so these would always score ~0 against a COCO-trained model).
CS_DROP_IN_COMMON = (5, 7)  # pole, traffic sign

# CS classes folded into a parent class under the 'common' label space.
# COCO does not distinguish riders from persons.
CS_MERGE_IN_COMMON = {12: 11}  # rider -> person


def common_target_remap(target, ignore_idx=IGNORE_INDEX):
    """Reduce a CS GT map to the 'common' label space (intersection with COCO).

    Sets `pole` and `traffic sign` pixels to ignore and merges `rider` into
    `person`. All other classes (and existing ignore pixels) are left unchanged.
    Use together with `common_pred_remap` on the predictions.
    """
    out = target.clone()
    for c in CS_DROP_IN_COMMON:
        out[target == c] = ignore_idx
    for src, dst in CS_MERGE_IN_COMMON.items():
        out[target == src] = dst
    return out


def cs_active_classes(label_space):
    """CS train_ids that should contribute to the mIoU under the given label space.

    Under 'common', the dropped classes (pole, traffic sign) and the merged
    source class (rider, folded into person) are NOT active: their IoU is
    meaningless / not what the model is being asked to predict, and including
    them in the per-class table or the mean drags the result down spuriously.
    """
    if label_space == "strict":
        return list(range(NUM_CS_CLASSES))
    if label_space == "common":
        inactive = set(CS_DROP_IN_COMMON) | set(CS_MERGE_IN_COMMON.keys())
        return [c for c in range(NUM_CS_CLASSES) if c not in inactive]
    raise ValueError(f"unknown label_space: {label_space}")


def mask_inactive_to_nan(iou_per_class, label_space):
    """Set the IoU of inactive classes (under `label_space`) to NaN.

    torchmetrics `MulticlassJaccardIndex` returns 0.0 (not NaN) for classes
    with no support, which silently inflates the mean when classes are dropped.
    This helper makes the dropped/merged classes show up as NaN so they can
    be skipped from the table and the mean cleanly.
    """
    active = set(cs_active_classes(label_space))
    out = iou_per_class.clone()
    for c in range(out.numel()):
        if c not in active:
            out[c] = float("nan")
    return out


def common_pred_remap(pred):
    """Apply the 'common' label space merges to predictions.

    Only the rider->person merge is applied: a correct `rider` prediction must
    count as a correct `person` prediction in the common space. The drop classes
    (pole, traffic sign) are intentionally NOT remapped here — leaving them lets
    a model's mispredictions on those classes still register as errors against
    whichever GT class they fell on.
    """
    out = pred.clone()
    for src, dst in CS_MERGE_IN_COMMON.items():
        out[pred == src] = dst
    return out
