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
from configs.config import cfg, cfg_from_file, VOC_CLASSES, COCO_CLASSES
from data.checkout import CHECKOUT_CLASSES
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
from torchvision.ops import nms as torch_nms
from multiprocessing import Process, Manager, Pipe, Array
from threading import Thread
from queue import Queue
import json
import math
from collections import defaultdict
import cv2
from functools import partial
<<<<<<< Updated upstream
from itertools import product
=======
>>>>>>> Stashed changes


def arg_parse():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detection')
    parser.add_argument(
        "--video",
        dest='video',
        help="Test video",
        default="test.mp4",
        type=str)
    parser.add_argument(
        '--weights',
        default='new_data6/refine_res50_epoch_200_512.pth',
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
    parser.add_argument(
        '--json',
        default='',
        type=str,
        help='File path to json results')
    parser.add_argument(
        '--class_json',
        default='',
        type=str,
        help='File path to json results')
    parser.add_argument(
        '--json_result',
        default='demo.json',
        type=str,
        help='File path to json results')
    parser.add_argument(
        '--output',
        default='output.avi',
        type=str,
        help='File path to output results')
    args = parser.parse_args()
    return args


def im_detect(img, net, detector, transform, obj_json, thresh=0.01):
    net.eval()
    with torch.no_grad():
        t0 = time.time()
        # w, h = img.shape[1], img.shape[0]
        x = transform(img)[0]
        x = (x - torch.FloatTensor([104., 117., 123.])).transpose(0, 2).transpose(1, 2).unsqueeze(0)
        # x = x.unsqueeze(0)
        t1 = time.time()
        x = x.cuda()
        output = net(x)
        boxes, scores = detector.forward(output)
        t2 = time.time()

        print("transform_t:", round(t1 - t0, 3), "detect_time:",
              round(t2 - t1, 3))

        max_conf, max_id = scores[0].topk(1, 1, True, True)
        pos = max_id > 0

        if len(pos) == 0:
            return np.empty((0, 6))
        boxes = boxes[0][pos.expand_as(boxes[0])].view(-1, 4)
        scores = max_conf[pos].view(-1, 1)
        max_id = max_id[pos].view(-1, 1)
        inds = scores > thresh
        if len(inds) == 0:
            return np.empty((0, 6))
        boxes = boxes[inds.view(-1, 1).expand(len(inds), 4)].view(-1, 4)
        scores = scores[inds].view(-1, 1)
        max_id = max_id[inds].view(-1, 1)
        c_dets = torch.cat((boxes, scores, max_id.float()), 1)
        img_classes = torch.unique(c_dets[:, -1])
        output = None
        flag = False
        for cls in img_classes:
            cls_mask = c_dets[:, -1] == cls
            image_pred_class = c_dets[cls_mask, :]
            # keep = nms(image_pred_class[:, :5], cfg.TEST.NMS_OVERLAP, device=torch.cuda.current_device(), force_cpu=True)
            # keep = keep[:50]
            keep = torch_nms(image_pred_class[:, :4], image_pred_class[:, 4], cfg.TEST.NMS_OVERLAP)
            keep = keep[:50]
            image_pred_class = image_pred_class[keep, :].cpu().numpy()
            if not flag:
                output = image_pred_class
                flag = True
            else:
                output = np.concatenate((output, image_pred_class), axis=0)

        if output is not None:
            output[:, 0:2][output[:, 0:2] < 0] = 0
            output[:, 2:4][output[:, 2:4] > 1] = 1
            # scale = np.array([w, h, w, h])
            # output[:, :4] = output[:, :4] * scale

            # scale = np.array([512, 512, 512, 512])
            scale_w = cfg.DATASETS.ROI[2] - cfg.DATASETS.ROI[0]
            scale_h = cfg.DATASETS.ROI[3] - cfg.DATASETS.ROI[1]
            scale = np.array([scale_w, scale_h, scale_w, scale_h])
            output[:, :4] = output[:, :4] * scale
            # roi_offset = np.array((1100, 700))
            # roi_offset = np.array((350, 850))
            roi_offset = np.array(cfg.DATASETS.ROI[:2])
            output[:, :2] += roi_offset
            output[:, 2:4] += roi_offset

            for o in output:
                tmp_json = {"xmin": int(o[0]), "ymin": int(o[1]), "xmax": int(o[2]), "ymax": int(o[3]),
                            "score": str(o[4]), "label": int(o[5])}
                obj_json.append(tmp_json)

        print(output)

    return output


def decode_job(q, video_name):
    print('decode_job start...')
    video = cv2.VideoCapture(video_name)
    count = 0
    while True:
        _, img = video.read()
        if img is None:
            q.put(None)
            break
        q.put(img)
        count += 1
        print(count)
    print('decode_job end')


def transform_job(img_q, data_q, transform):
    print('transform_job start...')
    while True:
        img = img_q.get(True)
        if img is None:
            data_q.put((None, None))
            break
        x = transform(img)[0]
        data_q.put((img, x))
    print('transform_job end')

<<<<<<< Updated upstream
def cal_obj_distance(o1, o2=None):
    if isinstance(o1, tuple) or isinstance(o1, list):
        o1, o2 = o1
=======
def cal_obj_distance(o1, o2):
>>>>>>> Stashed changes
    center1 = ((o1['xmin'] + o1['xmax']) / 2, (o1['ymin'] + o1['ymax']) / 2)
    center2 = ((o2['xmin'] + o2['xmax']) / 2, (o2['ymin'] + o2['ymax']) / 2)

    distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    return distance

def empty_list():
    return [0, 0]

MONEY = 1
CASHBOX = 2
MOBILE = 3
SCANNER =4

def process_json_new(json_file, class_json_file):
<<<<<<< Updated upstream
    t1 = time.time()
=======
>>>>>>> Stashed changes
    json_result = json.load(open(json_file))
    # class_result = json.load(open(class_json_file))

    # for obj, c in zip(json_result, class_result):
    #     obj['class'] = c

<<<<<<< Updated upstream
    # for obj in json_result:
    #     obj['class'] = 1
=======
    for obj in json_result:
        obj['class'] = 1
>>>>>>> Stashed changes

    money_array = np.zeros(len(json_result), dtype=np.uint8)
    cashbox_array = np.zeros(len(json_result), dtype=np.uint8)
    mobile2mobile_array = np.zeros(len(json_result), dtype=np.uint8)
    scanner2mobile_array = np.zeros(len(json_result), dtype=np.uint8)

    for i, info in enumerate(json_result):
<<<<<<< Updated upstream
        if info['objects'] and any([o['label'] == MONEY and
                                    800 < o['ymin'] < 1000 and
                                    1150 < o['xmin'] < 1480 for o in info['objects']]):
=======
        if info['objects'] and any([o['label'] == MONEY for o in info['objects']]):
>>>>>>> Stashed changes
            money_array[i] = 1
        if info['objects'] and any([o['label'] == CASHBOX for o in info['objects']]):
            cashbox_array[i] = 1

        scanners = list(filter(lambda x: x['label'] == SCANNER, info['objects']))
        mobiles = list(filter(lambda x: x['label'] == MOBILE, info['objects']))
        if scanners and mobiles:
            scanner = scanners[0]
            pfunc = partial(cal_obj_distance, scanner)
            distances = np.array(list(map(pfunc, mobiles)))
            if distances.min() < 150:
                scanner2mobile_array[i] = 1
<<<<<<< Updated upstream
        elif len(mobiles) > 1:
            distances = np.array(list(map(cal_obj_distance, product(mobiles, repeat=2))))
            distances[np.where(distances == 0)] = 1e8
            if distances.min() < 150:
                mobile2mobile_array[i] = 1

    ajust_mask = np.concatenate([np.ones(120, np.uint8), np.zeros(120, np.uint8)])

    cashbox_array = cv2.erode(cashbox_array, np.ones(2, np.uint8))
    cashbox_array = cv2.dilate(cashbox_array, np.ones(150, np.uint8))
    cashbox_array = cv2.erode(cashbox_array, np.ones(250, np.uint8))
    cashbox_array = cv2.dilate(cashbox_array, ajust_mask).squeeze()
=======
        # elif len(mobiles) > 1:

    cashbox_array = cv2.erode(cashbox_array, np.ones(2, np.uint8))
    cashbox_array = cv2.dilate(cashbox_array, np.ones(150, np.uint8))
    cashbox_array = cv2.erode(cashbox_array, np.ones(155, np.uint8)).squeeze()
>>>>>>> Stashed changes

    money_array = np.clip(money_array + cashbox_array, 0, 1)
    money_array = cv2.erode(money_array, np.ones(2, np.uint8))
    money_array = cv2.dilate(money_array, np.ones(150, np.uint8))
<<<<<<< Updated upstream
    money_array = cv2.erode(money_array, np.ones(250, np.uint8))
    money_array = cv2.dilate(money_array, ajust_mask).squeeze()

    scanner2mobile_array = cv2.dilate(scanner2mobile_array, ajust_mask).squeeze()

    mobile2mobile_array = cv2.dilate(mobile2mobile_array, ajust_mask).squeeze()

    result_array = np.clip(money_array + scanner2mobile_array + mobile2mobile_array, 0, 1)
    for info, tag in zip(json_result, result_array):
        info['payment_status'] = int(tag)

    print("T:", time.time() - t1)

    with open("result.json", 'w') as f:
        json.dump(json_result, f, indent=4)
=======
    money_array = cv2.erode(money_array, np.ones(155, np.uint8)).squeeze()

    scanner2mobile_array = cv2.dilate(scanner2mobile_array,
                                      np.concatenate([np.ones(60, np.uint8), np.zeros(60, np.uint8)])).squeeze()
    # scanner2mobile_array = cv2.erode(scanner2mobile_array, np.ones(95, np.uint8)).squeeze()

    result_array = np.clip(money_array + scanner2mobile_array, 0, 1)
    for info, tag in zip(json_result, result_array):
        info['status'] = tag
>>>>>>> Stashed changes

    return json_result


def process_json(json_file, class_json_file):
    json_result = json.load(open(json_file))
    # class_result = json.load(open(class_json_file))

    # for obj, c in zip(json_result, class_result):
    #     obj['class'] = c

    for obj in json_result:
        obj['class'] = 1

    clip_id = 0
    json_result[0]['clip_id'] = clip_id
    for i, info in enumerate(json_result[20:-20]):
        detect_time = 0
        if info['objects'] and any([o['label'] == 1 for o in info['objects']]):
            for j in range(-20, 20):
                if json_result[20 + i + j]['objects'] and \
                        any([o['label'] == 1 for o in json_result[10 + i + j]['objects']]):
                    detect_time += 1
            remove_index = []
            if detect_time < 10:
                for n in range(len(info['objects'])):
                    if info['objects'][n]['label'] == 1:
                        remove_index.append(n)
            for n in remove_index:
                del(info['objects'][n])


    statistic = defaultdict(empty_list)
    status = 0
    cooldown = 30
    for i, info in enumerate(json_result):
        if info['objects']:
            for obj in info['objects']:
                if obj['label'] == 1:
                    for temp_info in json_result[i:i+240]:
                        if any([o['label'] == 2 for o in temp_info['objects']]):
                            cooldown = 240
                        else:
                            cooldown = 90
                    status = 1
                elif obj['label'] == 2:
                    for temp_info in json_result[i:i+240]:
                        if any([o['label'] == 1 for o in temp_info['objects']]):
                            cooldown = 240
                            status = 1
                elif obj['label'] == 3:
                    scanners = list(filter(lambda x: x['label'] == 4, info['objects']))
                    mobiles = list(filter(lambda x: x['label'] == 3, info['objects']))
                    if scanners:
                        distance = cal_obj_distance(obj, scanners[0])
                        if distance < 150:
                            status = 2
                            cooldown = 60
                    elif len(mobiles) > 1:
                        for mobile in mobiles:
                            distance = cal_obj_distance(obj, mobile)
                            if 10 < distance < 150:
                                status = 2
                                cooldown = 60
                                break
                    else:
                        cooldown -= 1
                else:
                    cooldown -= 1
        elif status != 0:
            cooldown -= 1

        if cooldown < 0 and status != 0:
            statistic[info['time'].split(':')[0]][status-1] += 1

        if cooldown < 0:
            status = 0
        info['status'] = status

    with open("result.json", 'w') as f:
        json.dump(json_result, f, indent=4)
    return json_result

def main():
    global args
    args = arg_parse()
    cfg_from_file(args.cfg_file)
    bgr_means = cfg.TRAIN.BGR_MEAN

    if cfg.DATASETS.DATA_TYPE == 'VOC':
        classes = VOC_CLASSES
    elif cfg.DATASETS.DATA_TYPE == 'CHECKOUT':
        classes = CHECKOUT_CLASSES
    else:
        classes = COCO_CLASSES

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
    img_wh = cfg.TEST.INPUT_WH
    ValTransform = BaseTransform(img_wh, bgr_means, (2, 0, 1))
    thresh = cfg.TEST.CONFIDENCE_THRESH

<<<<<<< Updated upstream
    resolution = (500, 500)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(args.output, fourcc, 30.0, resolution, True)

    video = cv2.VideoCapture(args.video)
=======
    resolution = (800, 800)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(args.output, fourcc, 30.0, resolution, True)

    # manager = Manager()
    img_q = Queue(64)
    # data_q = Queue(64)
    decode_process = Thread(target=decode_job, args=(img_q, args.video))
    # transform_process = Thread(target=transform_job, args=(img_q, data_q, ValTransform))
    decode_process.start()
    # transform_process.start()

    video = cv2.VideoCapture(args.video)

>>>>>>> Stashed changes
    if args.json:
        json_result = process_json_new(args.json, args.class_json)

        count = 0
        while True:
            _, img = video.read()
            if img is None:
                break
            obj_json = json_result[count]

            dets = []
            for obj in obj_json['objects']:
                dets.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], float(obj['score']), obj['label']])

            draw_img = draw_rects(img, dets, classes, obj_json.get('status', None), obj_json.get('class', None))
            resized = cv2.resize(draw_img, resolution, interpolation=cv2.INTER_LINEAR)
            writer.write(resized)
            count += 1
        return
<<<<<<< Updated upstream

    img_q = Queue(64)
    # data_q = Queue(64)
    decode_process = Thread(target=decode_job, args=(img_q, args.video))
    # transform_process = Thread(target=transform_job, args=(img_q, data_q, ValTransform))
    decode_process.start()
    # transform_process.start()
=======
>>>>>>> Stashed changes

    count = 0
    json_result = []
    st = time.time()
    while True:
        t1 = time.time()
        # _, img = video.read()
        img = img_q.get(True)
        if img is None:
            break
        # img, x = data_q.get(True)
        # if img is None:
        #     break
        img_json = dict()
        seconds = round(count/30, 3)
        img_json['time'] = "{:0>2}:{:0>2}:{}".format(int(seconds//3600), int((seconds % 3600)//60), round(seconds % 60, 3))
        obj_json = []
        dets = im_detect(img, net, detector, ValTransform, obj_json, thresh)
        img_json['objects'] = obj_json
        json_result.append(img_json)
        count += 1
        draw_img = draw_rects(img, dets, classes, None, None)
        resized = cv2.resize(draw_img, resolution, interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('image', resized)
        # cv2.waitKey(10)
        writer.write(resized)
        print('Time:', time.time() - t1)

    decode_process.join()
    # transform_process.join()
    print("final time", time.time() - st)

    with open(args.json_result, 'w') as fp:
        json.dump(json_result, fp, indent=4)

    # count = 0
    # if True:
    #     img = cv2.imread("22697_0059.jpg")
    #
    #     dets = im_detect(img, net, detector, ValTransform, thresh)
    #     draw_img = draw_rects(img, dets, classes)
    #     cv2.imwrite("test{}.jpg".format(count), img)
    #     count += 1


if __name__ == '__main__':
    main()
