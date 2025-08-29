# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import math
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def focal_eiou_loss(preds, targets, gamma=0.5, reduction='none'):
    """
    preds: N * 4
    targets: N * 4
    """
    # preds in xyxy format
    preds_x1 = preds[:, 0]
    preds_y1 = preds[:, 1]
    preds_x2 = preds[:, 2]
    preds_y2 = preds[:, 3]
    preds_w = preds_x2 - preds_x1
    preds_h = preds_y2 - preds_y1
    
    # targets in xyxy format
    targets_x1 = targets[:, 0]
    targets_y1 = targets[:, 1]
    targets_x2 = targets[:, 2]
    targets_y2 = targets[:, 3]
    targets_w = targets_x2 - targets_x1
    targets_h = targets_y2 - targets_y1

    # intersection
    inter_x1 = torch.max(preds_x1, targets_x1)
    inter_y1 = torch.max(preds_y1, targets_y1)
    inter_x2 = torch.min(preds_x2, targets_x2)
    inter_y2 = torch.min(preds_y2, targets_y2)
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    
    inter_area = inter_w * inter_h
    preds_area = preds_w * preds_h
    targets_area = targets_w * targets_h
    
    union_area = preds_area + targets_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    # enclosing box
    enclose_x1 = torch.min(preds_x1, targets_x1)
    enclose_y1 = torch.min(preds_y1, targets_y1)
    enclose_x2 = torch.max(preds_x2, targets_x2)
    enclose_y2 = torch.max(preds_y2, targets_y2)
    
    enclose_w = torch.clamp(enclose_x2 - enclose_x1, min=0)
    enclose_h = torch.clamp(enclose_y2 - enclose_y1, min=0)
    
    # center distance
    preds_cx = (preds_x1 + preds_x2) / 2
    preds_cy = (preds_y1 + preds_y2) / 2
    targets_cx = (targets_x1 + targets_x2) / 2
    targets_cy = (targets_y1 + targets_y2) / 2
    
    center_dist_sq = (preds_cx - targets_cx) ** 2 + (preds_cy - targets_cy) ** 2
    
    # enclosing box diagonal
    enclose_c2 = enclose_w ** 2 + enclose_h ** 2 + 1e-6
    
    # width and height distance
    w_dist_sq = (preds_w - targets_w) ** 2
    h_dist_sq = (preds_h - targets_h) ** 2
    
    eiou = iou - (center_dist_sq / enclose_c2) - (w_dist_sq / (enclose_w ** 2 + 1e-6)) - (h_dist_sq / (enclose_h ** 2 + 1e-6))
    
    loss = (1 - eiou)
    
    # focal loss
    focal_loss = (iou.detach().abs()) ** gamma * loss
    
    if reduction == 'sum':
        return focal_loss.sum()
    elif reduction == 'mean':
        return focal_loss.mean()
    else:
        return focal_loss
