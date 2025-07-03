from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from rectpack import newPacker
import matplotlib


def scale_rectangle(rect, n):
    x1, y1, x2, y2 = rect
    
    # 计算矩形的中心点
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # 计算缩放后的半宽度和半高度
    half_width = (x2 - x1) / 2 * n
    half_height = (y2 - y1) / 2 * n
    
    # 计算新的左上角和右下角坐标，并转换为整数
    new_x1 = int(center_x - half_width)
    new_y1 = int(center_y - half_height)
    new_x2 = int(center_x + half_width)
    new_y2 = int(center_y + half_height)
    
    new_rect = (new_x1, new_y1, new_x2, new_y2)
    return new_rect


def scale_to_square(rect, n):
    x1, y1, x2, y2 = rect

    # 计算中心点坐标
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # 计算当前宽度和高度
    width = x2 - x1
    height = y2 - y1

    # 计算新的宽度和高度
    new_width = int(width * n)
    new_height = int(height * n)

    # 找到新的边界，以正方形为准，边长取宽高中的较大值
    side_length = max(new_width, new_height)

    # 计算新的左上角和右下角坐标
    new_x1 = cx - side_length // 2
    new_y1 = cy - side_length // 2
    new_x2 = cx + side_length // 2
    new_y2 = cy + side_length // 2

    # 计算原始矩形框在新正方形框中的相对位置
    rel_x1 = x1 - new_x1
    rel_y1 = y1 - new_y1
    rel_x2 = x2 - new_x1
    rel_y2 = y2 - new_y1

    new_rect = (new_x1, new_y1, new_x2, new_y2)
    relative_rect = (rel_x1, rel_y1, rel_x2, rel_y2)
    return new_rect, relative_rect


def create_mask_from_bbox(bbox, shape):
    mask = Image.new("L", shape, "black")
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle(bbox, fill="white")
    return mask


def get_rays(W, H, fx, fy, cx, cy, c2w_t, center_pixels):  # rot = I
    j, i = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32))
    if center_pixels:
        i = i.copy() + 0.5
        j = j.copy() + 0.5

    directions = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], -1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    rays_o = np.expand_dims(c2w_t, 0).repeat(H * W, 0)

    rays_d = directions  # (H, W, 3)
    rays_d = (rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)).reshape(-1, 3)
    return rays_o, rays_d


def draw_handpose(canvas, all_hand_peaks):
    eps = 0.01
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2),
                         matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas

def draw_facepose(canvas, lmks):
    eps = 0.01
    H, W, C = canvas.shape
    lmks = np.array(lmks)
    for lmk in lmks:
        x, y = lmk
        x = int(x)
        y = int(y)
        if x/W > eps and y/H > eps:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas

def get_bounding_box(image):
    np_image = np.asarray(image)

    # 获取非零像素的坐标
    non_zero_coords = np.argwhere(np_image)

    # 计算外接矩形框
    if non_zero_coords.size > 0:
        top_left = non_zero_coords.min(axis=0)
        bottom_right = non_zero_coords.max(axis=0)
        
        # 返回外接矩形框的左上角和右下角坐标
        return (top_left[1], top_left[0], bottom_right[1], bottom_right[0])
    else:
        # 如果图像全为零，返回None
        return None


def is_overlapping(rect1, rect2):
    """
    检查两个矩形是否重叠
    :param rect1: (x1, y1, x2, y2) 矩形1的坐标
    :param rect2: (x1, y1, x2, y2) 矩形2的坐标
    :return: 布尔值，表示是否重叠
    """
    if rect1[0] >= rect2[0] and rect1[1] >= rect2[1] and rect1[2] <= rect2[2] and rect1[3] <= rect2[3]:
       return True
    if rect2[0] >= rect1[0] and rect2[1] >= rect1[1] and rect2[2] <= rect1[2] and rect2[3] <= rect1[3]:
       return True
    else:
       return False
    #return not (rect1[2] <= rect2[0] or rect1[0] >= rect2[2] or
    #            rect1[3] <= rect2[1] or rect1[1] >= rect2[3])


def calculate_iou(box1, box2):
    """
    计算两个矩形框的交并比（IoU）

    参数:
    box1, box2: 矩形框的坐标，格式为 (x1, y1, x2, y2)
    其中 (x1, y1) 为左上角坐标，(x2, y2) 为右下角坐标

    返回:
    IoU值
    """

    # 确定交集矩形的坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 计算交集矩形的面积
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # 计算两个矩形框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集的面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area

    return iou