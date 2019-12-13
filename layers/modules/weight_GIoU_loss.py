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


def iou_giou_loss(bboxes1, bboxes2):
    """Calculate iou/giou loss between two set of bboxes.
    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (m, 4)
    Returns:
        giou loss(Tensor): shape (1, m)
    """

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # 卡住wh的最小值为0(当两者无相交部分时,设定交集为0)

    overlap = wh[:, 0] * wh[:, 1]

    area_a = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area_b = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    iou = overlap / (area_a + area_b - overlap)

    area_c_lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    area_c_rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])

    area_c = (area_c_rb[:, 0] - area_c_lt[:, 0]) * (area_c_rb[:, 1] - area_c_lt[:, 1])
    U = (area_a + area_b - overlap)
    giou = iou - (area_c - U) / area_c

    return 1 - giou
