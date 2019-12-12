import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import COCODetection, VOCDetection, CheckoutDetection, detection_collate, BaseTransform, preproc
from data import CHECKOUT_CLASSES
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
from configs.config import cfg, cfg_from_file, VOC_CLASSES, COCO_CLASSES
from utils.box_utils import draw_rects
import numpy as np
import time
import os
import sys
import pickle
import datetime
from models.model_builder import SSD
import yaml
import cv2


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detection')
    parser.add_argument(
        "--images",
        dest='images',
        help="Image / Directory containing images to perform detection upon",
        default="images",
        type=str)
    parser.add_argument(
        '--weights',
        default='weights/ssd_darknet_300.pth',
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
        default=8,
        type=int,
        help='Number of workers used in dataloading')
    parser.add_argument(
        '--retest', default=False, type=bool, help='test cache results')
    args = parser.parse_args()
    return args


def im_detect(img, net, detector, transform, thresh=0.01):
    with torch.no_grad():
        t0 = time.time()
        w, h = img.shape[1], img.shape[0]
        '''采用图片推理'''
        # x = transform(img)[0].unsqueeze(0)  # 将图片转化为tensor
        # x = x.cuda()  # 采用GPU
        '''采用随机tensor推理'''
        img.cuda()
        t1 = time.time()
        output = net(img)  # 网络推理
        t2 = time.time()
        boxes, scores = detector.forward(output)  # 检测并输出解码框和置信度
        t3 = time.time()
        transform_time = t1 - t0  # 计量transform时间
        inference_time = t2 - t1  # 计量inference时间
        detection_time = t3 - t2  # 计量detection时间
        # print("transform_time:", round(transform_time, 3),
        #       "inference_time:", round(inference_time, 3),
        #       "detect_time:", round(detection_time, 3))

        '''nms'''
        # max_conf, max_id = scores[0].topk(1, 1, True, True)
        # pos = max_id > 0
        # if len(pos) == 0:
        #     return np.empty((0, 6))
        # boxes = boxes[0][pos.view(-1, 1).expand(len(pos), 4)].view(-1, 4)
        # scores = max_conf[pos].view(-1, 1)
        # max_id = max_id[pos].view(-1, 1)
        # inds = scores > thresh
        # if len(inds) == 0:
        #     return np.empty((0, 6))
        # boxes = boxes[inds.view(-1, 1).expand(len(inds), 4)].view(-1, 4)
        # scores = scores[inds].view(-1, 1)
        # max_id = max_id[inds].view(-1, 1)
        # c_dets = torch.cat((boxes, scores, max_id.float()), 1).cpu().numpy()
        # img_classes = np.unique(c_dets[:, -1])
        # output = None
        # flag = False
        # for cls in img_classes:
        #     cls_mask = np.where(c_dets[:, -1] == cls)[0]
        #     image_pred_class = c_dets[cls_mask, :]
        #     keep = nms(image_pred_class, cfg.TEST.NMS_OVERLAP, force_cpu=False)
        #     keep = keep[:50]
        #     image_pred_class = image_pred_class[keep, :]
        #     if not flag:
        #         output = image_pred_class
        #         flag = True
        #     else:
        #         output = np.concatenate((output, image_pred_class), axis=0)
        # output[:, 0:2][output[:, 0:2] < 0] = 0
        # output[:, 2:4][output[:, 2:4] > 1] = 1
        # scale = np.array([w, h, w, h])
        # output[:, :4] = output[:, :4] * scale
        # t3 = time.time()
        # nms_time = t3 - t2  # 计算nms时间
        # print("nms_time:", round(nms_time, 3))

    return transform_time, inference_time, detection_time


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
        classes = VOC_CLASSES
        top_k = 200
    elif cfg.DATASETS.DATA_TYPE == 'CHECKOUT':
        trainvalDataset = CheckoutDetection
        classes = CHECKOUT_CLASSES
        top_k = 200
    else:
        trainvalDataset = COCODetection
        classes = COCO_CLASSES
        top_k = 300
    valSet = cfg.DATASETS.VAL_TYPE
    num_classes = cfg.MODEL.NUM_CLASSES
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg.TRAIN.TRAIN_ON = False
    net = SSD(cfg)

    checkpoint = torch.load(args.weights)
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

    detector = Detect(cfg)
    img_wh = cfg.TEST.INPUT_WH
    ValTransform = BaseTransform(img_wh, bgr_means, (2, 0, 1))
    input_folder = args.images
    thresh = cfg.TEST.CONFIDENCE_THRESH

    '''原始demo'''
    # for item in os.listdir(input_folder)[2:3]:
    #     img_path = os.path.join(input_folder, item)
    #     print(img_path)
    #     img = cv2.imread(img_path)
    #     dets = im_detect(img, net, detector, ValTransform, thresh)
    #     draw_img = draw_rects(img, dets, classes)
    #     out_img_name = "output_" + item
    #     save_path = os.path.join(save_folder, out_img_name)
    #     cv2.imwrite(save_path, img)

    '''采用单图推理'''
    # transform_list = []
    # detection_list = []
    # nms_list = []
    # for _ in range(1000):
    #     for item in os.listdir(input_folder):
    #         img_path = os.path.join(input_folder, item)
    #         # print(img_path)
    #         img = cv2.imread(img_path)
    #         dets, tt, dt, nt = im_detect(img, net, detector, ValTransform, thresh)
    #         draw_img = draw_rects(img, dets, classes)  # 绘制检测框
    #         out_img_name = "output_" + item
    #         save_path = os.path.join(save_folder, out_img_name)
    #         cv2.imwrite(save_path, draw_img)
    #         transform_list.append(tt)
    #         detection_list.append(dt)
    #         nms_list.append(nt)
    # mean_tt = np.mean(transform_list)
    # mean_dt = np.mean(detection_list)
    # mean_nt = np.mean(nms_list)
    # print("1000 inferences mean time: transfer_time: {:.4}, detection_time: {:.4}, nms_time: {:.4}".
    #       format(mean_tt, mean_dt, mean_nt))

    '''采用有batch size的随机张量推理'''
    transform_list = []
    inference_list = []
    detection_list = []
    img_random = torch.randn(batch_size, 3, 512, 512)  # random一个带有batch的tensor伪装成照片
    for _ in range(1000):
        tt, it, dt = im_detect(img_random, net, detector, ValTransform, thresh)
        transform_list.append(tt)
        inference_list.append(it)
        detection_list.append(dt)
    mean_tt = np.mean(transform_list) * 1000  # 转化量纲为毫秒
    mean_it = np.mean(inference_list) * 1000
    mean_dt = np.mean(detection_list) * 1000
    one_pic_it = mean_it / batch_size
    one_pic_dt = mean_dt / batch_size
    print("mean time: "
          "batch transfer_time: {:.3f}ms\n"
          "batch inference_time: {:.3f}ms\n"
          "batch detection_time: {:.3f}ms\n"
          "one image inference_time: {:.3f}ms"
          "one image detection_time: {:.3f}ms".
          format(mean_tt, mean_it, mean_dt, one_pic_it, one_pic_dt))


if __name__ == '__main__':
    st = time.time()
    main()
    print("final time", time.time() - st)
