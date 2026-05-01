"""COCO panoptic -> Cityscapes semantic class remap.

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
"""

import torch

from utils.constants import IGNORE_INDEX


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
