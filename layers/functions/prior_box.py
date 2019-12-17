import torch
from math import sqrt as sqrt
from itertools import product as product


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.img_wh = size_cfg.IMG_WH
        self.num_priors = len(size_cfg.ASPECT_RATIOS)
        self.feature_maps = size_cfg.FEATURE_MAPS
        self.variance = size_cfg.VARIANCE or [0.1]
        self.min_sizes = size_cfg.MIN_SIZES
        self.use_max_sizes = size_cfg.USE_MAX_SIZE
        if self.use_max_sizes:
            self.max_sizes = size_cfg.MAX_SIZES
        self.steps = size_cfg.STEPS
        self.aspect_ratios = size_cfg.ASPECT_RATIOS
        self.clip = size_cfg.CLIP
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        anchors = []
        for k, f in enumerate(self.feature_maps):
            # 采用shift移动anchor位置,增加anchor的数量,提升anchor的平铺密度
            min_sizes = self.min_sizes
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_k_w = min_size / self.img_wh[0]
                    s_k_h = min_size / self.img_wh[1]
                    if min_size == min_sizes[0]:
                        dense_cx = [x * self.steps[k][0] / self.img_wh[0] for x in
                                    [j + 0, j + 0.25, j + 0.5, j + 0.75]]
                        dense_cy = [y * self.steps[k][1] / self.img_wh[1] for y in
                                    [i + 0, i + 0.25, i + 0.5, i + 0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_k_w, s_k_h]
                            if self.use_max_sizes:
                                s_k_prime_w = sqrt(s_k_w * (self.max_sizes[k] / self.img_wh[0]))
                                s_k_prime_h = sqrt(s_k_h * (self.max_sizes[k] / self.img_wh[1]))
                                anchors += [cx, cy, s_k_prime_w, s_k_prime_h]
                            for ar in self.aspect_ratios[k]:
                                anchors += [cx, cy, s_k_w * sqrt(ar), s_k_h / sqrt(ar)]
                    elif min_size == min_sizes[1]:
                        dense_cx = [x * self.steps[k][0] / self.img_wh[0] for x in
                                    [j + 0, j + 0.25, j + 0.5, j + 0.75]]
                        dense_cy = [y * self.steps[k][1] / self.img_wh[1] for y in
                                    [i + 0, i + 0.25, i + 0.5, i + 0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_k_w, s_k_h]
                            if self.use_max_sizes:
                                s_k_prime_w = sqrt(s_k_w * (self.max_sizes[k] / self.img_wh[0]))
                                s_k_prime_h = sqrt(s_k_h * (self.max_sizes[k] / self.img_wh[1]))
                                anchors += [cx, cy, s_k_prime_w, s_k_prime_h]
                            for ar in self.aspect_ratios[k]:
                                anchors += [cx, cy, s_k_w * sqrt(ar), s_k_h / sqrt(ar)]
                    else:
                        cx = (j + 0.5) * self.steps[k][0] / self.img_wh[0]
                        cy = (i + 0.5) * self.steps[k][1] / self.img_wh[1]
                        anchors += [cx, cy, s_k_w, s_k_h]
                        if self.use_max_sizes:
                            s_k_prime_w = sqrt(s_k_w * (self.max_sizes[k] / self.img_wh[0]))
                            s_k_prime_h = sqrt(s_k_h * (self.max_sizes[k] / self.img_wh[1]))
                            anchors += [cx, cy, s_k_prime_w, s_k_prime_h]
                        for ar in self.aspect_ratios[k]:
                            anchors += [cx, cy, s_k_w * sqrt(ar), s_k_h / sqrt(ar)]

            # 原始配置
            # grid_h, grid_w = f[1], f[0]
            # for i in range(grid_h):
            #     for j in range(grid_w):
            #         # self.steps为下采样率
            #         f_k_h = self.img_wh[1] / self.steps[k][1]
            #         f_k_w = self.img_wh[0] / self.steps[k][0]
            #         # unit center x,y
            #         cx = (j + 0.5) / f_k_w
            #         cy = (i + 0.5) / f_k_h
            #
            #         # aspect_ratio: 1
            #         # rel size: min_size
            #         # 使用min_size的正方形anchor的宽高
            #         s_k_h = self.min_sizes[k] / self.img_wh[1]
            #         s_k_w = self.min_sizes[k] / self.img_wh[0]
            #         mean += [cx, cy, s_k_w, s_k_h]
            #
            #         # aspect_ratio: 1
            #         # rel size: sqrt(s_k * s_(k+1))
            #         # 使用max_size的正方形anchor的宽高
            #         if self.use_max_sizes:
            #             s_k_prime_w = sqrt(
            #                 s_k_w * (self.max_sizes[k] / self.img_wh[0]))
            #             s_k_prime_h = sqrt(
            #                 s_k_h * (self.max_sizes[k] / self.img_wh[1]))
            #             mean += [cx, cy, s_k_prime_w, s_k_prime_h]
            #
            #         # 使用设置比例的anchor的宽高
            #         for ar in self.aspect_ratios[k]:
            #             mean += [cx, cy, s_k_w * sqrt(ar), s_k_h / sqrt(ar)]

        # back to torch land
        # output = torch.Tensor(mean).view(-1, 4)
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        print(output.size())
        return output
