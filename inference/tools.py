from __future__ import annotations

from rectpack import newPacker
import cv2
import numpy as np
from diffusers.utils import BaseOutput
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from transformers import PretrainedConfig


def mask_dilate(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    arr = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated)


def mask_gaussian_blur(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    blur = ImageFilter.GaussianBlur(value)
    return image.filter(blur)


def bbox_padding(
        bbox: tuple[int, int, int, int], value: int = 32
) -> tuple[int, int, int, int]:
    arr = np.array(bbox).reshape(2, 2)
    arr[0] -= value
    arr[1] += value
    # arr = np.clip(arr, (0, 0), image_size)
    return tuple(arr.astype(int).flatten())


def bbox_padding_by_ratio(
        bbox: tuple[int, int, int, int], ratio: int = 2
) -> tuple[int, int, int, int]:
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_value = int(max(w, h) * ratio)
    arr = np.array(bbox).reshape(2, 2)

    arr[0] -= pad_value
    arr[1] += pad_value
    # arr = np.clip(arr, (0, 0), image_size)
    return tuple(arr.astype(int).flatten())


def bbox_add_bias(
        bbox: tuple[int, int, int, int], bias: tuple[int, int]
) -> tuple[int, int, int, int]:
    arr = np.array(bbox).reshape(2, 2)
    arr[:, 0] += bias[0]
    arr[:, 1] += bias[1]
    arr = arr.astype(int)
    # arr = np.clip(arr, (0, 0), image_size)
    return tuple(arr.flatten())


def cal_laplacian(image: np.ndarray):
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
    return sharpness


def refine_mask(mask):
    mask_np = np.array(mask)
    _, binary_mask = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    filled_mask_pil = Image.fromarray(filled_mask)
    return filled_mask_pil


def filter_bboxes(bboxes, min_ratio=0.125, max_face_num=6, max_area=0, image=None):
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
    filted_bboxes = []
    for bbox, area in zip(bboxes, areas):
        if area > max(areas) * min_ratio and area < max_area:
            filted_bboxes.append(bbox)

    # -------加入模糊过滤逻辑--------
    sharpnesses = []
    for bbox in filted_bboxes:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        bbox_shrink = (x1 + w // 4, y1 + h // 4, x2 - w // 4, y2 - h // 4)
        cropped_image = image.crop(bbox_shrink)
        cropped_image = cv2.cvtColor(np.asarray(cropped_image), cv2.COLOR_RGB2GRAY)
        sharpness = cal_laplacian(cropped_image)
        sharpnesses.append(sharpness)

    rt_bboxes, rt_sharpnesses = [], []
    for bbox, sharpness in zip(filted_bboxes, sharpnesses):
        if sharpness > 100 and sharpness / max(sharpnesses) > 0.1:
            rt_bboxes.append(bbox)
            rt_sharpnesses.append(sharpness)
    # -----------------------------

    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in rt_bboxes]
    if rt_bboxes:
        rt_bboxes, rt_sharpnesses, areas = zip(
            *sorted(zip(rt_bboxes, rt_sharpnesses, areas), key=lambda x: x[-1], reverse=True))
        return rt_bboxes[:max_face_num], rt_sharpnesses[:max_face_num]
    return rt_bboxes, rt_sharpnesses



def composite(
        init: Image.Image,
        mask: Image.Image,
        gen: Image.Image,
        bbox_padded: tuple[int, int, int, int],
) -> Image.Image:
    img_masked = Image.new("RGBa", init.size)
    img_masked.paste(
        init.convert("RGBA").convert("RGBa"),
        mask=ImageOps.invert(mask),
    )
    img_masked = img_masked.convert("RGBA")
    size = (
        bbox_padded[2] - bbox_padded[0],
        bbox_padded[3] - bbox_padded[1],
    )
    resized = gen.resize(size)

    output = Image.new("RGBA", init.size)
    output.paste(resized, bbox_padded)
    output.alpha_composite(img_masked)
    return output.convert("RGB")


def create_mask_from_bbox(
        bbox: np.ndarray, shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    mask = Image.new("L", shape, "black")
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle(bbox, fill="white")
    return mask


def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str = None, subfolder: str = ""
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    print("*" * 10 + f" Using {model_class} ! " + "*" * 10)

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "BertModel":
        from transformers import BertModel

        return BertModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def bbox_padding_to_square(bbox):
    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    if width > height:
        padding = (width - height) / 2
        new_x1 = x1
        new_x2 = x2
        new_y1 = int(y1 - padding)
        new_y2 = int(y2 + padding)
        if (new_y2 - new_y1) < width:
            new_y2 += 1
    else:
        padding = (height - width) / 2
        new_x1 = int(x1 - padding)
        new_x2 = int(x2 + padding)
        new_y1 = y1
        new_y2 = y2
        if (new_x2 - new_x1) < height:
            new_x2 += 1

    new_bbox = (new_x1, new_y1, new_x2, new_y2)
    return new_bbox


def merge_rectangles(rectangles):
    """
    合并一组矩形框
    :param rectangles: 矩形框列表 [(x1, y1, x2, y2), ...]
    :return: 合并后的矩形框 (x1, y1, x2, y2)
    """
    x1 = min(rect[0] for rect in rectangles)
    y1 = min(rect[1] for rect in rectangles)
    x2 = max(rect[2] for rect in rectangles)
    y2 = max(rect[3] for rect in rectangles)
    return (x1, y1, x2, y2)


def merge_masks(mask_list):
    if not mask_list:
        raise ValueError("mask_list不能为空")

    # 将第一个掩码转换为numpy数组并初始化合并数组
    merged_array = np.array(mask_list[0], dtype=np.uint16)

    # 逐个掩码进行合并，并限制最大灰度值为255
    for mask in mask_list[1:]:
        merged_array += np.array(mask, dtype=np.uint16)
        np.clip(merged_array, 0, 255, out=merged_array)

    # 将合并后的数组转换回Pillow图像
    merged_mask = Image.fromarray(merged_array.astype(np.uint8))

    return merged_mask


def is_overlapping(rect1, rect2):
    """
    检查两个矩形是否重叠
    :param rect1: (x1, y1, x2, y2) 矩形1的坐标
    :param rect2: (x1, y1, x2, y2) 矩形2的坐标
    :return: 布尔值，表示是否重叠
    """
    return not (rect1[2] <= rect2[0] or rect1[0] >= rect2[2] or
                rect1[3] <= rect2[1] or rect1[1] >= rect2[3])


def group_rectangles_ori(rectangles, max_width, max_height):
    """
    对矩形框进行分组，使得每组内的矩形框合并后的宽高不超过阈值
    """
    groups = []

    for outer_rect, mask in rectangles:
        placed = False
        for group in groups:
            # 检查合并后的矩形框是否满足尺寸要求
            merged_group = merge_rectangles([outer_rect] + [or_ for or_, _ in group])
            if (merged_group[2] - merged_group[0] <= max_width and
                    merged_group[3] - merged_group[1] <= max_height):
                group.append((outer_rect, mask))
                placed = True
                break

        if not placed:
            groups.append([(outer_rect, mask)])

    # 合并矩形框和mask
    merged_groups = []
    for group in groups:
        merged_groups.append((
            merge_rectangles([rect for rect, _ in group]),
            merge_masks([mask for _, mask in group])
        ))

    return merged_groups

def group_rectangles(rectangles, max_width, max_height):
    """
    对矩形框进行分组，使得每组内的矩形框合并后的宽高不超过阈值
    """
    groups = []

    for outer_rect, mask, condition, bbox in rectangles:
        placed = False
        for group in groups:
            # 检查合并后的矩形框是否满足尺寸要求
            merged_group = merge_rectangles([outer_rect] + [or_ for or_, _,_,_ in group])
            if (merged_group[2] - merged_group[0] <= max_width and
                    merged_group[3] - merged_group[1] <= max_height):
                group.append((outer_rect, mask, condition, bbox))
                placed = True
                break

        if not placed:
            groups.append([(outer_rect, mask, condition, bbox)])

    # 合并矩形框和mask
    merged_groups = []
    for group in groups:
        merged_groups.append((
            merge_rectangles([rect for rect, _,_,_ in group]),
            merge_masks([mask for _, mask,_,_ in group])
        ))

    return merged_groups, groups


def expand_to_square(bbox, img_width, img_height):
    """
    将矩形框尽量从中心外扩为一个方形，并且保证外扩后不超出图像。

    参数：
    bbox: 一个包含四个值的列表或元组，表示矩形框的左上角和右下角坐标 (x1, y1, x2, y2)
    img_width: 图像的宽度
    img_height: 图像的高度

    返回：
    一个包含四个值的列表，表示外扩后的方形的左上角和右下角坐标 (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox

    # 计算中心点
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 计算矩形框的宽度和高度
    width = x2 - x1
    height = y2 - y1

    # 取宽度和高度的最大值作为方形的边长
    side_length = max(width, height)

    # 计算方形的一半边长
    half_side_length = side_length / 2

    # 计算方形的左上角和右下角坐标
    new_x1 = cx - half_side_length
    new_y1 = cy - half_side_length
    new_x2 = cx + half_side_length
    new_y2 = cy + half_side_length

    # 确保方形的边不会超出图像的边界
    if new_x1 < 0:
        new_x1 = 0
        new_x2 = side_length
    if new_y1 < 0:
        new_y1 = 0
        new_y2 = side_length
    if new_x2 > img_width - 1:
        new_x2 = img_width - 1
        new_x1 = new_x2 - side_length
    if new_y2 > img_height - 1:
        new_y2 = img_height - 1
        new_y1 = new_y2 - side_length

    # 确保坐标不超出边界（再次检查，以防上一步调整带来超出边界的情况）
    new_x1 = max(0, min(new_x1, img_width - 1))
    new_y1 = max(0, min(new_y1, img_height - 1))
    new_x2 = max(0, min(new_x2, img_width - 1))
    new_y2 = max(0, min(new_y2, img_height - 1))

    # 返回调整后的方形的坐标
    return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))


def arrange_in_square(sizes):
    total_area = sum(w * h for w, h in sizes)
    side_length = int(np.ceil(np.sqrt(total_area)))

    while True:
        # 创建一个新的打包器，指定不旋转矩形框
        packer = newPacker(rotation=False)

        # 添加所有图像块到打包器中，使用唯一标识符
        for i, (w, h) in enumerate(sizes):
            packer.add_rect(w, h, rid=i)

        # 添加一个初始尺寸的正方形画布
        packer.add_bin(side_length, side_length, count=1)

        # 执行打包操作
        packer.pack()

        # 检查是否所有矩形框都被打包
        if len(packer) > 0 and len(packer[0]) == len(sizes):
            break

        # 如果未能全部打包，则增大正方形画布的尺寸
        side_length += 1

    # 获取打包后矩形框的位置，使用标识符确保顺序正确
    positions = sorted(packer[0], key=lambda rect: rect.rid)
    positions = [(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height) for rect in positions]

    return side_length, positions


def relative_box(rect, rect_new):
    x_min, y_min, x_max, y_max = rect
    new_x_min, new_y_min, new_x_max, new_y_max = rect_new
    rel_x_min = x_min - new_x_min
    rel_y_min = y_min - new_y_min
    rel_x_max = x_max - new_x_min
    rel_y_max = y_max - new_y_min
    relative_rect = (rel_x_min, rel_y_min, rel_x_max, rel_y_max)
    return relative_rect


def is_contain(rect, rect_new):
    x_min, y_min, x_max, y_max = rect
    new_x_min, new_y_min, new_x_max, new_y_max = rect_new
    if x_min >= new_x_min and y_min >= new_y_min and x_max <= new_x_max and y_max <= new_y_max:
        return True
    else:
        return False


def merged_index(rectangles, rect_merged, merged):
    xmin, ymin, xmax, ymax = rect_merged
    lengths = []
    for idx, rect in enumerate(rectangles):
        if idx in merged:
            side_length = 10000
        else:
            x_min = min(rect[0], xmin)
            y_min = min(rect[1], ymin)
            x_max = max(rect[2], xmax)
            y_max = max(rect[3], ymax)
            width = x_max - x_min
            height = y_max - y_min
            side_length = max(width, height)
        lengths.append(side_length)
    sorted_lengths = sorted(lengths)
    min_length = sorted_lengths[0]
    for idx, length in enumerate(lengths):
        if length == min_length:
            merged_index = idx
    return merged_index, min_length


def calculate_area(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    return (x2 - x1) * (y2 - y1)


def paste_condition(condition_image, pasted_condition_map, bbox_relative):
    condition_map = np.array(condition_image)
    paste_y_min = bbox_relative[1]
    paste_x_min = bbox_relative[0]
    nonzero_y, nonzero_x, _ = (condition_map != 0).nonzero()
    pasted_condition_map[paste_y_min + nonzero_y, paste_x_min + nonzero_x, :] = condition_map[
                                                                                nonzero_y, nonzero_x, :]
    # H = bbox_padded_with_bias[3] - bbox_padded_with_bias[1]
    # W = bbox_padded_with_bias[2] - bbox_padded_with_bias[0]
    # pasted_condition = Image.fromarray(
    #     cv2.cvtColor(np.uint8(pasted_condition_map[0:int(H), 0:int(W), :]), cv2.COLOR_BGR2RGB))
    return pasted_condition_map
