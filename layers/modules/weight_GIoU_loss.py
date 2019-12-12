#!/usr/bin/env python
# encoding: utf-8
"""
@file: weight_GIoU_loss.py
@time: on 2019/12/12 at 10:29
@author: Created by M.sc.SHI
@version: 0.10
@license: 1994-2019
@contact: <yukyewshek@outlook.com>
@software: PyCharm Professional
@Copyright: VINCENT.S PRODUCTION Co.,Ltd
"""
import torch


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'giou', 'iof']

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])

    wh = (rb - lt + 1).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

    if mode == 'iou':
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
        ious = overlap / (area1 + area2 - overlap)
        return ious
    elif mode == 'giou':
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
        clt = torch.min(bboxes1[:, :2], bboxes2[:, :2])
        crb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])

        area_c = (crb[:, 0] - clt[:, 0] + 1) * (
                crb[:, 1] - clt[:, 1] + 1)
        U = (area1 + area2 - overlap)
        ious = overlap / U
        gious = ious - (area_c - U) / area_c
        return gious
    else:
        ious = overlap / area1
        return ious
