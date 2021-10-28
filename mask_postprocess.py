import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import mask_target
from mmdet.models.builder import HEADS, build_loss
import cv2
import random

def cv2_rotate(img, angle):
    # get image height, width
    (h, w) = img.shape[:2]

    # calculate the center of the image
    center = (w / 2, h / 2)

    scale = 1.0

    # Perform the counter clockwise rotation holding at the center
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (h, w))

    return rotated


def postprocess(labels, scores, boxes, masks, conf_threshold=0.9):
    # 对于每个帧，提取每个检测到的对象的边界框和掩码
    # 掩模的输出大小为NxCxHxW
    # N  - 检测到的框数
    # C  - 类别数量（不包括背景）
    # HxW  - 分割形状
    numDetections = boxes.shape[0]
    print("detection box number is = ", numDetections)
    results = []

    colors = [[0., 255., 0.],
              [0., 0., 255.],
              [255., 0., 0.],
             ]
    # color = self.color[classId%len(colors)]
    color_index = random.randint(0, len(colors) - 1)
    color = colors[color_index]

    for i in range(numDetections):
        box = boxes[i]
        mask = masks[i]
        score = scores[i]
        if score > conf_threshold:
            left = int(box[0])
            top = int(box[1])
            right = int(box[2])
            bottom = int(box[3])
            
            result = {}
            result["score"] = score
            result["classid"] = int(labels[i])
            result["box"] = (left, top, right, bottom)
            result["mask"] = mask[int(labels[i])]  # source result["mask"] = mask 
            results.append(result)

    return results


def vis_res(img_cv2, src_img, results, mask_threshold=0.3):
    for result in results:
        # box
        left, top, right, bottom = result["box"]
        cv2.rectangle(img_cv2,
                      (left, top),
                      (right, bottom),
                      (255, 0, 0), 1)

        # class label
        classid = result["classid"]
        score = result["score"]
        label = '%.2f' % score
        classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        if classes:
            assert (classid < len(classes))
            label = '%s:%s' % (classes[classid], label)
        print("label = ", label)
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, label_size[1])

        cv2.rectangle(
            img_cv2,
            (left, top - round(1.1 * label_size[1])),
            (left + round(1.1 * label_size[0]), top + baseline),
            (255, 255, 255), cv2.FILLED)
        cv2.putText(
            img_cv2,
            label,
            (left, top),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # cv2.imwrite('/home/yhuang/tensorRT_work/convert_trt_quantize/rcnn_out_0.jpg', img_cv2)
        # mask
        class_mask = result["mask"]

        class_mask = cv2.resize(class_mask, (right - left , bottom - top ))
        mask = (class_mask > mask_threshold)
        roi = img_cv2[top: bottom, left: right][mask]

        colors = [[0., 255., 0.],
                  [0., 0., 255.],
                  [255., 0., 0.],
                  [0., 255., 255.],
                  [255., 255., 0.],
                  [255., 0., 255.],
                  [80., 70., 180.],
                  [250., 80., 190.],
                  [245., 145., 50.],
                  [70., 150., 250.],
                  [50., 190., 190.], ]
        # color = self.color[classId%len(colors)]
        color_index = random.randint(0, len(colors) - 1)
        color = colors[color_index]

        img_cv2[top: bottom, left: right][mask] = (
                [0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

        mask = mask.astype(np.uint8)
        contours, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(
            img_cv2[top: bottom, left: right],
            contours,
            -1,
            color,
            1,
            cv2.LINE_8,
            hierachy,
            100)

    # t = 50
    # label = 'Inference time: %.2f ms' % \
    #         (t * 1000.0 / cv2.getTickFrequency())
    # cv2.putText(img_cv2, label, (0, 15),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    src_size = src_img.shape
    img_cv2 = cv2.resize(img_cv2, (src_size[1] , src_size[0] ))  #resize to original image scale for display
    cv2.imwrite('maskrcnn_r50_res.jpg', img_cv2)

    # plt.figure(figsize=(10, 8))
    # plt.imshow(img_cv2[:, :,::-1])
    # plt.axis("off")
    # plt.show()

    rotated = cv2_rotate(src_img, 10)
    cv2.imwrite('maskrcnn_r50_rotated.jpg', rotated)

