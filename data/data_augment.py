"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

TODO: implement data_augment for training

Ellis Brown, Max deGroot
"""

import torch
from torchvision import transforms
import cv2
import numpy as np
import random
import math
from utils.box_utils import matrix_iou
from PIL import Image
from configs.config import cfg


def _fix_crop(image, boxes, labels):
    height, width, _ = image.shape
    w, h = 512, 512
    for _ in range(50):
        l = random.randrange(250) + 900
        t = 700
        roi = np.array((l, t, l + w, t + h))

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
            .all(axis=1)
        boxes_t = boxes[mask].copy()
        labels_t = labels[mask].copy()
        if len(boxes_t) == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] -= roi[:2]

        return image_t, boxes_t, labels_t
    else:
        roi = np.array((900, 700, 900 + w, 700 + h))
        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        boxes_t = np.zeros((1, 4))
        labels_t = np.zeros((1, 1))

        # print("Warning, fix_crop with no target", boxes, labels)
        return image_t, boxes_t, labels_t

def _crop(image, boxes, labels):
    # height, width, _ = image.shape
    height, width = 512, 512

    # if len(boxes) == 0:
    #     return image, boxes, labels

    while True:
        mode = random.choice((
            # None,
            (0.001, None),
            (0.01, None),
            (0.1, None),
            # (0.3, None),
            # (0.5, None),
            # (0.7, None),
            # (0.9, None),
            # (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        if not boxes.any():
            mode = (None, None)

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.8, 1.2)
            min_ratio = max(0.5, scale * scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)
            if random.random() < 0.2:
                h, w = height, width

            if not boxes.any():
                l = random.randrange(int(w/2), image.shape[1] - int(w/2*3))
                t = random.randrange(int(h/2), image.shape[0] - int(h/2*3))
            else:
                l = random.randrange(image.shape[1] - w)
                t = random.randrange(image.shape[0] - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])

            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            if not boxes.any():
                return image_t, boxes, labels

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                     .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t, labels_t


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes, fill, p):
    if random.random() > p:
        return image, boxes

    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(0.9, 1.1)

        min_ratio = max(0.5, 1. / scale / scale)
        max_ratio = min(2, scale * scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale * ratio
        hs = scale / ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)

        expand_image = np.empty((h, w, depth), dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror(image, boxes):
    height, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

    if random.randrange(2):
        image = image[::-1, :]
        boxes = boxes.copy()
        boxes[:, 1::2] = height - boxes[:, 3::-2]

    return image, boxes


def preproc_for_test(image, resize_wh, mean):
    interp_methods = [
        cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
        cv2.INTER_LANCZOS4
    ]
    # interp_method = interp_methods[random.randrange(5)]
    interp_method = interp_methods[3]
    # image = Image.fromarray(image)
    # image = image.resize((resize_wh[0], resize_wh[1]))
    image = cv2.resize(
        image, (resize_wh[0], resize_wh[1]), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image -= mean
    # to rgb
    # image = image[:, :, (2, 1, 0)]
    return image.transpose(2, 0, 1)


class preproc(object):
    def __init__(self, resize_wh, rgb_means, p):
        self.means = rgb_means
        self.resize_wh = resize_wh
        self.p = p

    def __call__(self, image, targets):
        # image_t = preproc_for_test(image, self.resize_wh, self.means)
        # return torch.from_numpy(image_t), targets
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        # image_crop, boxes, labels = _fix_crop(image, boxes, labels)
        # if len(labels.shape) != 2:
        #     labels = np.expand_dims(labels, 1)
        # targets_crop = np.hstack((boxes, labels))
        # image = image_crop
        # targets = targets_crop

        # if len(boxes) == 0:
        #     #boxes = np.empty((0, 4))
        #     targets = np.zeros((1, 5))
        #     image = preproc_for_test(image, self.resize_wh, self.means)
        #     return torch.from_numpy(image), targets

        # image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image.shape
        boxes_o = targets_o[:, :-1]
        labels_o = targets_o[:, -1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o, 1)
        targets_o = np.hstack((boxes_o, labels_o))

        image_t, boxes, labels = _crop(image, boxes, labels)
        image_t = _distort(image_t)
        image_t, boxes = _expand(image_t, boxes, self.means, self.p)
        image_t, boxes = _mirror(image_t, boxes)
        #image_t, boxes = _mirror(image, boxes)

        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.resize_wh, self.means)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0]) * 1.
        b_h = (boxes[:, 3] - boxes[:, 1]) * 1.
        mask_b = np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        if len(boxes_t) == 0:
            #image = preproc_for_test(image_o, self.resize_wh, self.means)
            #return torch.from_numpy(image), targets_o
            return torch.from_numpy(image_t), targets_o

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return torch.from_numpy(image_t), targets_t


class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, resize_wh, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize_wh = resize_wh
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img, target=None):

        interp_methods = [
            cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
            cv2.INTER_NEAREST, cv2.INTER_LANCZOS4
        ]
        # interp_method = interp_methods[0]
        # img = img.astype(np.float32)
        #roi = np.array((0, 0, img.shape[1], img.shape[0]))
        # roi = np.array((850, 450, 850 + 832, 450 + 832))
        # roi = np.array((1100, 700, 1100 + 512, 700 + 512))
        # roi = np.array((350, 850, 350 + 512, 850 + 512))
        roi = np.array(cfg.DATASETS.ROI)
        img = img[roi[1]:roi[3], roi[0]:roi[2]].astype(np.float32)
        # img = cv2.resize(
        #     np.array(img), (self.resize_wh[0], self.resize_wh[1]),
        #     interpolation=interp_method).astype(np.float32)
        # img -= self.means
        # img = img.transpose(self.swap)
        # if target is not None:
        #     target[:, :2] = np.maximum(target[:, :2], roi[:2])
        #     target[:, :2] -= roi[:2]
        #     target[:, 2:4] = np.minimum(target[:, 2:4], roi[2:])
        #     target[:, 2:4] -= roi[:2]
        # pos = target[0, :].astype(np.uint32)
        # cv2.rectangle(img, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 2)
        # cv2.imshow("1", img)
        # cv2.waitKey(0)
        return torch.from_numpy(img), target
