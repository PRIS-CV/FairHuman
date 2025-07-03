from pathlib import Path
import torch
import argparse
import os
import time
import cv2
import numpy as np
import matplotlib
from typing import Tuple, List, Any
from skimage.filters import gaussian
from trimesh.ray.ray_pyembree import RayMeshIntersector
from trimesh import Trimesh
import trimesh
from .hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from .hamer.utils import recursive_to
from .hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from .hamer.utils.renderer import Renderer, cam_crop_to_full
from PIL import Image, ImageDraw, ImageFilter
from .hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from .vitpose_model import ViTPoseModel
from .DWPose.annotator.dwpose import DWposeDetector
from os.path import join, dirname
from diffusers.utils import load_image
from glob import glob
from tqdm import tqdm
from .utils import scale_to_square, scale_rectangle, create_mask_from_bbox, get_rays, draw_handpose, draw_facepose, get_bounding_box, \
    is_overlapping, calculate_iou
from .yolo import YOLODetecotor

COLOR = (1.0, 1.0, 0.9)


def calculate_area(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    return (x2 - x1) * (y2 - y1)


def calculate_IoU(box1, box2):
    """
    computing the IoU of two boxes.
    Args:
        box: [x1, y1, x2, y2],通过左上和右下两个顶点坐标来确定矩形
    Return:
        IoU: IoU of box1 and box2.
    """
    px1 = box1[0]
    py1 = box1[1]
    px2 = box1[2]
    py2 = box1[3]

    gx1 = box2[0]
    gy1 = box2[1]
    gx2 = box2[2]
    gy2 = box2[3]

    parea = calculate_area(box1)  # 计算P的面积
    garea = calculate_area(box2)  # 计算G的面积

    # 求相交矩形的左上和右下顶点坐标(x1, y1, x2, y2)
    x1 = max(px1, gx1)  # 得到左上顶点的横坐标
    y1 = max(py1, gy1)  # 得到左上顶点的纵坐标
    x2 = min(px2, gx2)  # 得到右下顶点的横坐标
    y2 = min(py2, gy2)  # 得到右下顶点的纵坐标

    # 利用max()方法处理两个矩形没有交集的情况,当没有交集时,w或者h取0,比较巧妙的处理方法
    # w = max(0, (x2 - x1))  # 相交矩形的长，这里用w来表示
    # h = max(0, (y1 - y2))  # 相交矩形的宽，这里用h来表示
    # print("相交矩形的长是：{}，宽是：{}".format(w, h))
    # 这里也可以考虑引入if判断
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0

    area = w * h  # G∩P的面积
    if area == parea:
        return -1
    elif area == garea:
        return 2
    # 并集的面积 = 两个矩形面积 - 交集面积
    IoU = area / (parea + garea - area)

    return IoU


def bbox_padding(
        bbox: Tuple[int, int, int, int], value: int = 32
) -> Tuple[Any, ...]:
    arr = np.array(bbox).reshape(2, 2)
    arr[0] -= value
    arr[1] += value
    # arr = np.clip(arr, (0, 0), image_size)
    return tuple(arr.astype(int).flatten())


def cal_laplacian(image: np.ndarray):
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
    return sharpness


def filter_bboxes(bboxes, min_ratio=0.125, max_face_num=6, max_area=0, image=None):
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
    filted_bboxes = []
    for bbox, area in zip(bboxes, areas):
        if max(areas) * min_ratio < area < max_area:
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
        if sharpness > 0 and sharpness / max(sharpnesses) > 0:
            rt_bboxes.append(bbox)
            rt_sharpnesses.append(sharpness)
    # -----------------------------

    # areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in rt_bboxes]
    # if rt_bboxes:
    #     rt_bboxes, rt_sharpnesses, areas = zip(
    #         *sorted(zip(rt_bboxes, rt_sharpnesses, areas), key=lambda x: x[-1], reverse=True))
    #     return rt_bboxes[:max_face_num], rt_sharpnesses[:max_face_num]
    return rt_bboxes, rt_sharpnesses


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


def group_rectangles(rectangles, max_width, max_height):
    """
    对矩形框进行分组，使得每组内的矩形框合并后的宽高不超过阈值
    """
    groups = []

    for outer_rect in rectangles:
        placed = False
        for group in groups:
            # 检查合并后的矩形框是否满足尺寸要求
            merged_group = merge_rectangles([outer_rect] + [or_ for or_ in group])
            if (merged_group[2] - merged_group[0] <= max_width and
                    merged_group[3] - merged_group[1] <= max_height):
                group.append(outer_rect)
                placed = True
                break

        if not placed:
            groups.append([outer_rect])

    return groups


class HamerDetector:
    def __init__(self, model_dir, rescale_factor, device):
        # HaMeR model
        self.model, self.model_cfg = load_hamer(join(model_dir, "hamer/hamer_ckpts/checkpoints/hamer.ckpt"))
        self.model.to(device)
        self.model.eval()

        # keypoint detector
        self.cpm = ViTPoseModel(join(model_dir, "hamer/vitpose_ckpts/vitpose+_huge/wholebody.pth"), device)
        self.dwpose = DWposeDetector()

        # renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

        # yolo detector
        self.yolo = YOLODetecotor(join(model_dir, "yolo/person_yolov8m-seg.pt"), 0.3, device)
        self.rescale_factor = rescale_factor
        self.device = device

    @torch.no_grad()
    def __call__(self, image: Image.Image, bbox_scale_factor, mask_scale_factor, is_cropped):
        # init
        patches = []
        depth_conditions = []
        pose_conditions = []
        mesh_conditions = []
        multi_conditions = []
        masks = []
        bboxes_padded = []
        delete_index = []
        blurred_index = []

        # yolo detect
        yolo_detections, _, confs = self.yolo(image)
        if yolo_detections is None:
            return bboxes_padded, depth_conditions, pose_conditions, multi_conditions, mesh_conditions, masks

        if len(yolo_detections) == 0:
            return bboxes_padded, multi_conditions, depth_conditions, pose_conditions, mesh_conditions, masks

        img_cv2 = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        img = img_cv2.copy()[:, :, ::-1]
        # Detect human keypoints for each person
        vitposes_out,vis = self.cpm.predict_pose_and_visualize(
            img,
            [np.concatenate([yolo_detections, confs[:, None]], axis=1)],0.5,0.3,4,1
        )
       
        bboxes = []
        is_right = []
        sum_valid = []
        mean_valid = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
                sum_valid.append(sum(valid))
                mean_valid.append(np.mean(keyp[:, 2]))

            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)
                sum_valid.append(sum(valid))
                mean_valid.append(np.mean(keyp[:, 2]))

        # print(yolo_detections)
        # import pdb;
        # pdb.set_trace()
        dropped = [False] * len(bboxes)
        for i in range(len(bboxes)):
            for j in range(len(bboxes)):
                if i == j:
                    continue
                if is_overlapping(bboxes[i], bboxes[j]):
                    if calculate_area(bboxes[i]) > calculate_area(bboxes[j]):
                        dropped[j] = True
                    else:
                        dropped[i] = True
                if calculate_iou(bboxes[i], bboxes[j]) > 0.2 and is_right[i] == is_right[j]:
                    if calculate_area(bboxes[i]) >= calculate_area(bboxes[j]):
                        dropped[j] = True
                    else:
                        dropped[i] = True
        bboxes = [x for i, x in enumerate(bboxes) if not dropped[i]]
        is_right = [x for i, x in enumerate(is_right) if not dropped[i]]
        # sum_valid = [x for i, x in enumerate(sum_valid) if not dropped[i]]
        # mean_valid = [x for i, x in enumerate(mean_valid) if not dropped[i]]
        # import pdb;pdb.set_trace()
        if bboxes == []:
            return bboxes_padded, multi_conditions, depth_conditions, pose_conditions, mesh_conditions, masks

        bboxes = np.array(bboxes).astype(int)
        #print(bboxes)
        # do filtering
        #  with size
        filtering = [False] * len(bboxes)
        for i in range(len(bboxes)):
            # print(calculate_area(bboxes[i]))
            if calculate_area(bboxes[i]) <= 400:
                delete_index.append(i)
                filtering[i] = True
        #  with laps
        # rt_bboxes, sharpnesses = filter_bboxes(
        #     bboxes,
        #     min_ratio=0.0,
        #     max_face_num=10,
        #     max_area=image.size[0] * image.size[1],
        #     image=image,
        # )
        # for k, sharpness in enumerate(sharpnesses):
        #     if sharpness < 20 or sharpness > 7000:
        #         blurred_index.append(k)
        #         filtering[k] = True
        #     if len(bboxes) > 4:
        #         if sharpness < 200 or sharpness > 6800:
        #             blurred_index.append(k)
        #             filtering[k] = True
        #print(sharpnesses)

        bboxes = np.array([x for i, x in enumerate(bboxes) if not filtering[i]])
        is_right = np.array([x for i, x in enumerate(is_right) if not filtering[i]])
        # sharpnesses = [x for i, x in enumerate(sharpnesses) if not filtering[i]]

        # extra_filtering = [False] * len(bboxes)
        # extra_blurred_index = []
        # if len(bboxes) > 4:
        #     sharpnesses_sorted = sorted(sharpnesses)
        #     blurred = sharpnesses_sorted[0]
        #     if blurred < 500:
        #         for k, sharpness in enumerate(sharpnesses):
        #             if sharpness < blurred + 100:
        #                 extra_blurred_index.append(k)
        #                 extra_filtering[k] = True
        # bboxes = np.array([x for i, x in enumerate(bboxes) if not extra_filtering[i]])
        # is_right = np.array([x for i, x in enumerate(is_right) if not extra_filtering[i]])
    
        #print(merged_boxes_new)
        #print(merged_is_rights_new)
        # print(bboxes)
        #print(merged_index)
        #print(blurred_index)
        #print(delete_index)
        #print(extra_delete_index)
        #print(is_right)
        # import pdb;
        # pdb.set_trace()
        if len(bboxes)==0:
            return bboxes_padded, multi_conditions, depth_conditions, pose_conditions, mesh_conditions, masks
        multi_condition, depth_condition, pose_condition, mesh_condition = self.inference(
            image,
            bboxes,
            is_right,
            is_cropped
        )
        multi_conditions.append(multi_condition)
        depth_conditions.append(depth_condition)
        pose_conditions.append(pose_condition)
        mesh_conditions.append(mesh_condition)
        global_mask = Image.fromarray(np.zeros((image.size[0], image.size[1]))).convert('L')
        for bbox in bboxes:
            bbox_padded, _ = scale_to_square(bbox, bbox_scale_factor)
            crop_multi_condition = multi_condition.crop(bbox_padded)
            bbox_from_multi = get_bounding_box(crop_multi_condition)
            bbox_for_mask = [
                min(bbox[0], bbox_from_multi[0] + bbox_padded[0]),
                min(bbox[1], bbox_from_multi[1] + bbox_padded[1]),
                max(bbox[2], bbox_from_multi[2] + bbox_padded[0]),
                max(bbox[3], bbox_from_multi[3] + bbox_padded[1]),
            ]
            mask = create_mask_from_bbox(scale_rectangle(bbox_for_mask, mask_scale_factor), image.size)
            nonzero_y, nonzero_x = np.asarray(mask).nonzero()
            ymin = min(nonzero_y)
            ymax = max(nonzero_y)
            xmin = min(nonzero_x)
            xmax = max(nonzero_x)
            crop_mask = mask.crop([xmin, ymin, xmax, ymax])
            global_mask.paste(crop_mask, [xmin, ymin, xmax, ymax])
        # multi_condition.save('multi.png')
        # global_mask.save('mask.png')
        masks.append(global_mask)
        return bboxes_padded, multi_conditions, depth_conditions, pose_conditions, mesh_conditions, masks

    def inference(self, patch: Image.Image, bbox, right, is_cropped):
        img_cv2 = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)
        H, W, C = img_cv2.shape
        if is_cropped:
            dataset = ViTDetDataset(self.model_cfg, img_cv2, np.array([bbox]), np.array([right]),
                                    rescale_factor=self.rescale_factor)
        else:
            dataset = ViTDetDataset(self.model_cfg, img_cv2, np.stack(bbox), np.stack(right),
                                    rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_hand_peaks = []
        all_box_size = []
        all_box_center = []

        multi_condition = None

        padded_multimap = np.zeros((2 * H, 2 * W, 3))
        padded_depthmap = np.zeros((2 * H, 2 * W, 3))
        padded_posemap = np.zeros((2 * H, 2 * W, 3))
        # start = time.time()
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                               scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                        DEFAULT_MEAN[:, None, None] / 255)
                input_patch = input_patch.permute(1, 2, 0).numpy()

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                keyp2d = out['pred_keypoints_2d'][n].detach().cpu().numpy()
                box_size = batch["box_size"][n].detach().cpu().numpy()
                box_center = batch["box_center"][n].detach().cpu().numpy()
                pred_cam = out['pred_cam'][n].detach().cpu().numpy()
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                focal_length = scaled_focal_length.detach().cpu().numpy()
                res = int(box_size)
                camera_t = np.array([-pred_cam[1], -pred_cam[2], -2 * focal_length / (res * pred_cam[0] + 1e-9)])
                faces_new = np.array([[92, 38, 234],
                                      [234, 38, 239],
                                      [38, 122, 239],
                                      [239, 122, 279],
                                      [122, 118, 279],
                                      [279, 118, 215],
                                      [118, 117, 215],
                                      [215, 117, 214],
                                      [117, 119, 214],
                                      [214, 119, 121],
                                      [119, 120, 121],
                                      [121, 120, 78],
                                      [120, 108, 78],
                                      [78, 108, 79]])
                faces = np.concatenate([self.model.mano.faces, faces_new], axis=0)
                mesh = Trimesh(vertices=verts, faces=faces)
                h, w = int(box_size), int(box_size)
                rays_o, rays_d = get_rays(w, h, focal_length, focal_length, w / 2, h / 2, camera_t, True)
                if int(box_size) == 0:
                    continue

                coords = np.array(list(np.ndindex(h, w))).reshape(h, w, -1).transpose(1, 0, 2).reshape(-1, 2)
                intersector = RayMeshIntersector(mesh)
                points, index_ray, _ = intersector.intersects_location(rays_o, rays_d, multiple_hits=False)

                tri_index = intersector.intersects_first(rays_o, rays_d)

                tri_index = tri_index[index_ray]

                assert len(index_ray) == len(tri_index)
                if is_right == 0:
                    discriminator = (np.sum(mesh.face_normals[tri_index] * rays_d[index_ray], axis=-1) >= 0)
                else:
                    discriminator = (np.sum(mesh.face_normals[tri_index] * rays_d[index_ray], axis=-1) <= 0)
                points = points[discriminator]  # ray intersects in interior faces, discard them

                if len(points) == 0:
                    print("no hands detected")
                    continue

                depth = (points + camera_t)[:, -1]
                index_ray = index_ray[discriminator]
                pixel_ray = coords[index_ray]

                minval = np.min(depth)
                maxval = np.max(depth)
                depthmap = np.zeros([h, w])
                depthmap[pixel_ray[:, 0], pixel_ray[:, 1]] = 1.0 - (0.8 * (depth - minval) / (maxval - minval))
                depthmap *= 255

                cropped_depthmap = depthmap
                if cropped_depthmap is None:
                    print("Depth reconstruction failed for image")
                    continue

                resized_cropped_depthmap = cv2.resize(cropped_depthmap, (int(box_size), int(box_size)),
                                                      interpolation=cv2.INTER_LINEAR)
                resized_cropped_depthmap = cv2.cvtColor(np.uint8(resized_cropped_depthmap), cv2.COLOR_GRAY2RGB)
                nonzero_y, nonzero_x, _ = (resized_cropped_depthmap != 0).nonzero()
                if len(nonzero_y) == 0 or len(nonzero_x) == 0:
                    print("Depth reconstruction failed for image")
                    continue

                crop_xc = box_center[0]
                crop_yc = box_center[1]
                crop_y_min = int(crop_yc - box_size / 2)
                crop_x_min = int(crop_xc - box_size / 2)

                padded_multimap[crop_y_min + nonzero_y, crop_x_min + nonzero_x, :] = resized_cropped_depthmap[
                                                                                     nonzero_y, nonzero_x, :]
                padded_depthmap[crop_y_min + nonzero_y, crop_x_min + nonzero_x, :] = resized_cropped_depthmap[
                                                                                     nonzero_y, nonzero_x, :]

                keyp2d = keyp2d + 0.5
                canv = np.zeros(shape=(int(box_size), int(box_size), 3), dtype=np.uint8)
                peaks = []
                peaks.append(keyp2d)
                pose = draw_handpose(canv, peaks)
                pose = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB)

                if is_right == 0:
                    pose = np.flip(pose, 1)
                nonzero_y, nonzero_x, _ = (pose != 0).nonzero()
                crop_xc = box_center[0]
                crop_yc = box_center[1]
                crop_y_min = int(crop_yc - box_size / 2)
                crop_x_min = int(crop_xc - box_size / 2)

                padded_multimap[crop_y_min + nonzero_y, crop_x_min + nonzero_x, :] = pose[nonzero_y, nonzero_x, :]
                padded_posemap[crop_y_min + nonzero_y, crop_x_min + nonzero_x, :] = pose[nonzero_y, nonzero_x, :]
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_hand_peaks.append(keyp2d)
                all_box_size.append(int(box_size))
                all_box_center.append(box_center)
        # time_cost = time.time()- start
        # print(time_cost)
        multi_condition = Image.fromarray(
            cv2.cvtColor(np.uint8(padded_multimap[0:int(H), 0:int(W), :]), cv2.COLOR_BGR2RGB))
        depth_condition = Image.fromarray(
            cv2.cvtColor(np.uint8(padded_depthmap[0:int(H), 0:int(W), :]), cv2.COLOR_BGR2RGB))
        pose_condition = Image.fromarray(
            cv2.cvtColor(np.uint8(padded_posemap[0:int(H), 0:int(W), :]), cv2.COLOR_BGR2RGB))

        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=COLOR,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = self.renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=img_size[n],
                is_right=all_right,
                **misc_args,
            )
        
            mesh_condition = cam_view[:, :, :3] * cam_view[:, :, 3:]
            mesh_condition = Image.fromarray(
                cv2.cvtColor(np.uint8(255 * mesh_condition[:, :, ::-1]), cv2.COLOR_BGR2RGB))
            
        return multi_condition, depth_condition, pose_condition, mesh_condition
