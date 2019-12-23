import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp, decode
from .focal_loss_softmax import FocalLossSoftmax
from .focal_loss_sigmoid import FocalLossSigmoid
from .weight_GIoU_loss import iou_giou_loss

GPU = False
if torch.cuda.is_available():
    GPU = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg):
        super(MultiBoxLoss, self).__init__()
        self.cfg = cfg
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.variance = size_cfg.VARIANCE
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.threshold = cfg.TRAIN.OVERLAP
        self.OHEM = cfg.TRAIN.OHEM
        self.negpos_ratio = cfg.TRAIN.NEG_RATIO
        self.variance = size_cfg.VARIANCE
        if cfg.TRAIN.FOCAL_LOSS:
            if cfg.TRAIN.FOCAL_LOSS_TYPE == 'SOFTMAX':
                self.focaloss = FocalLossSoftmax(
                    self.num_classes, gamma=2, size_average=False)
            else:
                self.focaloss = FocalLossSigmoid()

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            '''对money和scanner这两个小目标的w与h放大1.5倍'''
            truths_new = truths.clone()
            label = labels.cpu().numpy().tolist()
            for i in label:
                if i == 1.0:
                    truths_new[:, 2] = truths_new[:, 2] * 1.5
                    truths_new[:, 3] = truths_new[:, 3] * 1.5
                if i == 4.0:
                    truths_new[:, 2] = truths_new[:, 2] * 1.5
                    truths_new[:, 3] = truths_new[:, 3] * 1.5
            if self.num_classes == 2:
                labels = labels > 0
            defaults = priors.data
            # match(self.threshold, truths, defaults, self.variance, labels,
            #       loc_t, conf_t, idx)
            match(self.threshold, truths_new, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()

        pos = conf_t > 0
        num_pos = pos.sum(1, keepdim=True)

        if self.OHEM:
            # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, self.num_classes)

            loss_hard = log_sum_exp(batch_conf) - batch_conf.gather(
                1, conf_t.view(-1, 1))
            # Hard Negative Mining
            loss_hard[pos.view(-1, 1)] = 0  # filter out pos boxes for now
            loss_hard = loss_hard.view(num, -1)
            _, loss_idx = loss_hard.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            if num_pos.data.sum() > 0:
                num_neg = torch.clamp(
                    self.negpos_ratio * num_pos, max=pos.size(1) - 1, min=self.negpos_ratio)
            else:
                fake_num_pos = torch.ones(num, 1).long() * 2
                num_neg = torch.clamp(
                    self.negpos_ratio * fake_num_pos, max=pos.size(1) - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(
                -1, self.num_classes)
            targets_weighted = conf_t[(pos + neg).gt(0)]

            '''分配交叉熵的权重'''
            class_weight = torch.tensor([1.0, 5.0, 9.95, 2.0, 6.65]).cuda()
            '''使用权重'''
            loss_c = F.cross_entropy(
                conf_p, targets_weighted, weight=class_weight, size_average=False)

        else:
            loss_c = F.cross_entropy(conf_p, conf_t, size_average=False)
        # Localization Loss (Smooth L1/GIoU)
        # Shape: [batch,num_priors,4]
        if num_pos.data.sum() > 0:
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)

            '''采用Smooth L1 Loss'''
            # loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

            '''采用GIoU Loss'''
            priors = priors.unsqueeze(0).expand_as(loc_data)
            priors_pos = priors[pos_idx].view(-1, 4)
            giou_loc_p = decode(loc_p, priors_pos, self.variance) * 512
            giou_loc_t = decode(loc_t, priors_pos, self.variance) * 512
            loss_l = torch.mean(iou_giou_loss(giou_loc_p, giou_loc_t))

            N = (num_pos.data.sum() + num_neg.data.sum()) / 2
        else:
            loss_l = torch.zeros(1)
            N = num_pos.data.sum() + num_neg.data.sum()
            print('Warning, num_pos == 0')

        # loss_l = loss_l  # 适用于计算GIoU的loss
        # loss_l /= float(N)  # 适用于计算Smooth L1的loss
        loss_c /= float(N)
        return loss_l, loss_c
