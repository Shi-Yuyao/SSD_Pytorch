from models.model_builder import SSD
import yaml
import cv2
from torchvision.ops import nms as torch_nms
import numpy as np
import torch
import time
import argparse
from data import BaseTransform_demo, BaseTransform, preproc
from configs.config import cfg, cfg_from_file, VOC_CLASSES, COCO_CLASSES
from data.checkout import CHECKOUT_CLASSES
from layers.functions import Detect
import os
import math
from threading import Thread
from queue import Queue


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
        '--output',
        default='output.avi',
        type=str,
        help='File path to output results')
    args = parser.parse_args()
    return args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_color(c, x, max_val):
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                                [1, 1, 0], [1, 0, 0]])
    ratio = float(x) / max_val * 5
    i = int(math.floor(ratio))
    j = int(math.ceil(ratio))
    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
    return int(r * 255)


def draw_rects(img, rects, classes):
    if rects is None:
        return img

    for rect in rects:
        if rect[5] > 0.1:
            left_top = (int(float(rect[0])), int(float(rect[1])))
            right_bottom = (int(float(rect[2])), int(float(rect[3])))
            score = round(rect[4], 2)
            cls_id = int(rect[-1])
            label = "{0}".format(classes[cls_id])
            class_len = len(classes)
            offset = cls_id * 123457 % class_len
            red = get_color(2, offset, class_len)
            green = get_color(1, offset, class_len)
            blue = get_color(0, offset, class_len)
            color = (blue, green, red)
            cv2.rectangle(img, left_top, right_bottom, color, 2)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            right_bottom = left_top[0] + t_size[0] + 100, left_top[1] - t_size[
                1] - 15
            # cv2.rectangle(img, left_top, right_bottom, color, -1)
            # cv2.putText(img,
            #             str(label) + ": " + str(score),
            #             (left_top[0], left_top[1] - t_size[1] + 8),
            #             cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 0], 2)

    return img


def im_detect(img, net, detector, transform, thresh=0.01):
    net.eval()
    with torch.no_grad():
        t0 = time.time()
        w, h = img.shape[1], img.shape[0]
        x = transform(img)[0]
        x = (x - torch.FloatTensor([104., 117., 123.])).transpose(0, 2).transpose(1, 2).unsqueeze(0)
        t1 = time.time()
        x = x.to(device)
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
            scale = np.array([w, h, w, h])
            output[:, :4] = output[:, :4] * scale

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
    net = net.to(device)

    detector = Detect(cfg)
    img_wh = cfg.TEST.INPUT_WH
    ValTransform = BaseTransform_demo(img_wh, bgr_means, (2, 0, 1))
    thresh = cfg.TEST.CONFIDENCE_THRESH

    video = cv2.VideoCapture(args.video)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    resolution = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    writer = cv2.VideoWriter(args.output, fourcc, fps, resolution, True)

    img_q = Queue(64)
    decode_process = Thread(target=decode_job, args=(img_q, args.video))
    decode_process.start()

    count = 0
    st = time.time()
    while True:
        t1 = time.time()
        img = img_q.get(True)
        if img is None:
            break

        dets = im_detect(img, net, detector, ValTransform, thresh)
        count += 1
        draw_img = draw_rects(img, dets, classes)
        cv2.imshow('image', draw_img)
        cv2.waitKey(10)
        writer.write(draw_img)
        print('Time:', time.time() - t1)

    decode_process.join()
    print("final time", time.time() - st)


if __name__ == "__main__":
    main()




