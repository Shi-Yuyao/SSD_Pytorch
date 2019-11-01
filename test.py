import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import COCODetection, VOCDetection, CheckoutDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
from configs.config import cfg, cfg_from_file
import numpy as np
import time
import os
import sys
import pickle
import datetime
from models.model_builder import SSD
import yaml
import json
from data.checkout import CHECKOUT_CLASSES

ind_to_class = dict(zip(range(len(CHECKOUT_CLASSES)), CHECKOUT_CLASSES))

def arg_parse():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detection')
    parser.add_argument(
        '--weights',
        default='weights/refine_vgg_epoch_20_512.pth',
        type=str,
        help='Trained state_dict file path to open')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--save_folder',
        default='eval/',
        type=str,
        help='File path to save results')
    parser.add_argument(
        '--num_workers',
        default=16,
        type=int,
        help='Number of workers used in dataloading')
    parser.add_argument(
        '--retest', default=False, type=bool, help='test cache results')
    args = parser.parse_args()
    return args


def eval_net(val_dataset,
             val_loader,
             net,
             detector,
             cfg,
             transform,
             max_per_image=300,
             thresh=0.01,
             batch_size=1):
    net.eval()
    roi = np.array(cfg.DATASETS.ROI)
    num_images = len(val_dataset)
    num_classes = cfg.MODEL.NUM_CLASSES
    eval_save_folder = "./eval/"
    if not os.path.exists(eval_save_folder):
        os.mkdir(eval_save_folder)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    det_file = os.path.join(eval_save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        val_dataset.evaluate_detections(all_boxes, eval_save_folder)
        return

    json_result = []
    for idx, (imgs, _, img_info) in enumerate(val_loader):
        with torch.no_grad():
            t1 = time.time()
            x = imgs
            x = x.cuda()
            output = net(x)
            t4 = time.time()
            boxes, scores = detector.forward(output)
            t2 = time.time()
            for k in range(boxes.size(0)):
                img_json = dict()
                i = idx * batch_size + k
                boxes_ = boxes[k]
                scores_ = scores[k]
                boxes_ = boxes_.cpu().numpy()
                scores_ = scores_.cpu().numpy()
                img_id = img_info[k][2]
                img_wh = img_info[k][:2]
                img_json['img_id'] = img_id
                scale = np.array([img_wh[0], img_wh[1], img_wh[0], img_wh[1]])
                xscale = roi[2] - roi[0]
                yscale = roi[3] - roi[1]
                scale = np.array([xscale, yscale, xscale, yscale])
                boxes_ *= scale
                roi_offset = np.array(cfg.DATASETS.ROI[:2])
                boxes_[:, :2] += roi_offset
                boxes_[:, 2:] += roi_offset
                obj_json = []
                for j in range(1, num_classes):
                    inds = np.where(scores_[:, j] > thresh)[0]
                    if len(inds) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = boxes_[inds]
                    c_scores = scores_[inds, j]
                    c_dets = np.hstack((c_bboxes,
                                        c_scores[:, np.newaxis])).astype(
                                            np.float32, copy=False)
                    keep = nms(c_dets, cfg.TEST.NMS_OVERLAP, device=torch.cuda.current_device(), force_cpu=False)
                    keep = keep[:50]
                    c_dets = c_dets[keep, :]
                    for det in c_dets:
                        tmp_json = {"xmin": int(det[0]), "ymin": int(det[1]), "xmax": int(det[2]), "ymax": int(det[3]),
                                    "score": str(det[4]), "label": ind_to_class[j]}
                        obj_json.append(tmp_json)

                    all_boxes[j][i] = c_dets

                img_json["objects"] = obj_json
                json_result.append(img_json)

            t3 = time.time()
            detect_time = t2 - t1
            nms_time = t3 - t2
            forward_time = t4 - t1
            if idx % 10 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s'.format(
                    i + 1, num_images, forward_time, detect_time, nms_time))

    with open("test.json", 'w') as fp:
        json.dump(json_result, fp, indent=4)

def main():
    global args
    args = arg_parse()
    cfg_from_file(args.cfg_file)
    bgr_means = cfg.TRAIN.BGR_MEAN
    dataset_name = cfg.DATASETS.DATA_TYPE
    batch_size = cfg.TEST.BATCH_SIZE
    num_workers = args.num_workers
    if cfg.DATASETS.DATA_TYPE == 'VOC':
        trainvalDataset = VOCDetection
        top_k = 200
    elif cfg.DATASETS.DATA_TYPE == 'CHECKOUT':
        trainvalDataset = CheckoutDetection
        top_k = 50
    else:
        trainvalDataset = COCODetection
        top_k = 300
    dataroot = cfg.DATASETS.DATAROOT
    if cfg.MODEL.SIZE == '300':
        size_cfg = cfg.SMALL
    else:
        size_cfg = cfg.BIG
    valSet = cfg.DATASETS.VAL_TYPE
    num_classes = cfg.MODEL.NUM_CLASSES
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg.TRAIN.TRAIN_ON = False
    net = SSD(cfg)

    checkpoint = torch.load(args.weights, map_location='cpu')
    state_dict = checkpoint['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    # net.cuda()
    detector = Detect(cfg)
    ValTransform = BaseTransform(size_cfg.IMG_WH, bgr_means, (2, 0, 1))
    val_dataset = trainvalDataset(dataroot, valSet, ValTransform, None, "test")
    val_loader = data.DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate)
    top_k = 300
    thresh = cfg.TEST.CONFIDENCE_THRESH
    eval_net(
        val_dataset,
        val_loader,
        net,
        detector,
        cfg,
        ValTransform,
        top_k,
        thresh=thresh,
        batch_size=batch_size)


if __name__ == '__main__':
    st = time.time()
    main()
    print("final time", time.time() - st)
