import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
import torch
import string
import csv
import re
import os
from os.path import join
import requests
from PIL import Image,ImageDraw
import io, imageio, base64
from yolo import YOLODetecotor
from preprocessor.Hamer.hamer_detector_multi import HamerDetector
from face_skin import Face_Skin
import argparse
import sys
import time
import warnings
import PIL
import dataclasses
from transformers import CLIPImageProcessor, HfArgumentParser
from diffusers import UNet2DConditionModel, ControlNetModel
from diffusers.utils import load_image,make_image_grid
import numpy as np
import random
from diffusers import AutoPipelineForImage2Image, AutoPipelineForInpainting, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetInpaintPipeline,AutoencoderKL,UniPCMultistepScheduler
from tools import bbox_add_bias, create_mask_from_bbox, group_rectangles, expand_to_square, composite, calculate_area, group_rectangles_ori, mask_dilate,mask_gaussian_blur,refine_mask,bbox_padding,bbox_padding_to_square,arrange_in_square, relative_box

# Extract mask
def create_mask_from_bboxes(
        bboxes, shape):
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
    for bbox in bboxes:
        mask_draw.rectangle(bbox, fill="white")
    return mask


def filter_bboxes(bboxes, min_ratio=0.125, max_face_num=6, max_area=0, image=None):
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
    filted_bboxes = []
    for bbox, area in zip(bboxes, areas):
        if area > max(areas) * min_ratio and area < max_area:
            filted_bboxes.append(bbox)


@dataclasses.dataclass
class InferenceArgs:
    eval_txt_path: str | None = None
    output: str | None = "./inference"
    target_imgs_path: str | None = None
    base_model_path: str | None = "stabilityai/sd_xl_base_1.0"
    controlnet_path: str | None = None
    vae_path: str | None = "madebyollin/sdxl-vae-fp16-fix"
    lora_path: str | None = None
    lora_weight: float=0.0
    detector_path: str | None = "./models"

def main(args: InferenceArgs):
    # Bulid Detector
    model_dir = args.detector_path
    device = "cuda"
    detector1 = YOLODetecotor(os.path.join(model_dir, "yolo/hand_yolov8n.pt"), 0.3, device)
    detector2 = YOLODetecotor(os.path.join(model_dir, "yolo/face_distortion_last.pt"), 0.3, device)
    detector3 = YOLODetecotor(os.path.join(model_dir, "yolo/person_yolov8m-seg.pt"), 0.3, device)
    hamer_detector = HamerDetector(
                model_dir=model_dir,
                rescale_factor=2.0,
                device=device
    )

    segment_enabled = True
    if segment_enabled == True:
        face_skin = Face_Skin(os.path.join(model_dir, "yolo/face_skin.pth"))
        face_skin.model.to(device)

    pretrained_model_name_or_path = args.base_model_path
    vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16)
    controlnet_path = args.controlnet_path
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

    inpaint_pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        controlnet=controlnet,
        torch_dtype=torch.float16)

    # lora
    if args.lora_path is not None and args.lora_weight > 0:
        inpaint_pipeline.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name='text_lora')
        inpaint_pipeline.set_adapters(["text_lora"], adapter_weights=[args.lora_weight])

    inpaint_pipeline.scheduler = UniPCMultistepScheduler.from_config(inpaint_pipeline.scheduler.config)
    inpaint_pipeline.to('cuda')

    prompt_list = open(args.eval_txt_path, "r").readlines()
    prompts = [prompt.strip() for prompt in prompt_list]
    path_list = os.listdir(args.target_imgs_path)
    path_list.sort(key=lambda x:int(x.split('_')[0]))

    start = time.time()
    generator = torch.Generator(device="cuda").manual_seed(0)
    output_path =args.output
    os.makedirs(output_path, exist_ok=True)

    for i, (filename, prompt) in enumerate(zip(path_list,prompts)):  
        image_path =args.target_imgs_path+"/"+filename
        image = load_image(image_path).convert("RGB")
        prompt=prompt+", realistic, high quality, 8k"
        negative_prompt = "fake 3D rendered image, deforemd, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"
        width, height = image.size
        yolo_detections_hand, _, _ = detector1(image)
        yolo_detections_face,_,_ = detector2(image)
        detections_total = []
        if yolo_detections_face is None and yolo_detections_hand is None:
            final_image = image
            final_image.save(output_path+"/"+filename)
            continue

        # 根据生成内容选择重绘类型
        #人物个数
        yolo_detections_human,_,_ = detector3(image)
        if yolo_detections_human is not None and len(yolo_detections_human)>0: 
            num_human = len(yolo_detections_human)
        else:
            num_human = 0
        #手部占比
        areas_hand = []
        if yolo_detections_hand is not None and len(yolo_detections_hand)>0:
            for yolo_detection_hand in yolo_detections_hand:
                areas_hand.append(calculate_area(yolo_detection_hand))
        ratio_hand = sum(areas_hand) / (width*height)  
        # average_ratio_hand = ratio_hand / len(areas_hand) 
        #脸部占比
        areas_face = []
        if yolo_detections_face is not None and len(yolo_detections_face)>0:
            for yolo_detection_face in yolo_detections_face:
                areas_face.append(calculate_area(yolo_detection_face))
        ratio_face = sum(areas_face) / (width*height) 
        # average_ratio_face = ratio_face / len(areas_face) 
        #总占比
        ratio_total = (sum(areas_hand)+sum(areas_face)) / (width*height)
        # average_ratio_total = ratio_total / (len(areas_face)+len(areas_hand))

        #重绘类型
        if ratio_total<=0.01:
            type = "merge_crop"
        elif num_human<=2 and ratio_total>0.1:
            type = "global"
        elif num_human>=5:
            type = "merge_crop"
        else:
            type = "mosaic"
        # type = "global"

        # 全局
        if type =="global":
            bboxes_padded_hand, multi_conditions_wholebody, masks_hand, wholebody_pose= hamer_detector(
                        image, 2.5,
                        1.2,
                        is_cropped=False)
            if bboxes_padded_hand is not None and len(bboxes_padded_hand)>0:
                for bbox_padded_hand in bboxes_padded_hand:
                    # if calculate_area(yolo_detection_hand) > 400:
                    detections_total.append(bbox_padded_hand)
            if yolo_detections_face is not None and len(yolo_detections_face)>0:
                for yolo_detection_face in yolo_detections_face:
                    #过滤大脸
                    if calculate_area(yolo_detection_face)< 0.125*width*height:
                        xmin, ymin, xmax, ymax = yolo_detection_face
                        detections_total.append((int(xmin),int(ymin),int(xmax),int(ymax)))
            if len(detections_total)==0:
                final_image = image
                final_image.save(output_path+"/"+filename)
                continue
            
            bboxes_padded_for_mask  = []
            bbox_scaling_ratio =2.5
            mask_padding = 16
            mask_scaling_ratio = 1.2
            mask_dilation = 8
            mask_blur = 8
            for bbox in detections_total:
                padding_mask = int(0.5 * (mask_scaling_ratio - 1) * max(bbox[2] - bbox[0],
                                                                                                    bbox[3] - bbox[1]))
                bbox_padded_for_mask = bbox_padding(bbox, padding_mask)
                bboxes_padded_for_mask.append(bbox_padded_for_mask)

            mask = create_mask_from_bboxes(bboxes_padded_for_mask,(width,height))
            mask = mask_dilate(mask, mask_dilation)
            # --再一次修复和膨胀--
            mask = refine_mask(mask)
            mask = mask_dilate(mask, mask_dilation)
            # -------------------
            mask = mask_gaussian_blur(mask, mask_blur)
        
            if multi_conditions_wholebody is not None and len(multi_conditions_wholebody)>0:
               control_image = multi_conditions_wholebody[0]
            elif wholebody_pose is not None:
               control_image=Image.fromarray(np.uint8(wholebody_pose))
            else:
                control_image = image
            width,height = control_image.size

            # prompt = "Six dancers rehearse a modern dance under stage lights with elegant and powerful movements., China, professional"
            final_image = inpaint_pipeline(
                prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                image=image,
                mask_image=mask,
                control_image=control_image,
                controlnet_conditioning_scale=0.55,
                strength=0.4,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator
            ).images[0]
            # import pdb;pdb.set_trace()

        
        ###################################################################################################
        # 统一mosaic
        if type == "mosaic":
            bboxes_padded_hand, multi_conditions, masks_hand, wholebody_pose= hamer_detector(
                        image, 2.5,
                        1.2,
                        is_cropped=True)
            if bboxes_padded_hand is not None and len(bboxes_padded_hand)>0:
                for bbox_padded_hand in bboxes_padded_hand:
                    # if calculate_area(yolo_detection_hand) > 400:
                    detections_total.append(bbox_padded_hand)
            if yolo_detections_face is not None and len(yolo_detections_face)>0:
                for yolo_detection_face in yolo_detections_face:
                    #过滤大脸
                    if calculate_area(yolo_detection_face)< 0.125*width*height:
                        xmin, ymin, xmax, ymax = yolo_detection_face
                        detections_total.append((int(xmin),int(ymin),int(xmax),int(ymax)))
                        multi_conditions.append(Image.new("RGB", (int(xmax)-int(xmin), int(ymax)-int(ymin)), "black"))
                        # multi_conditions.append(Image.fromarray(np.uint8(wholebody_pose)).crop((int(xmin),int(ymin),int(xmax),int(ymax))))

            if len(detections_total)==0:
                final_image = image
                final_image.save(output_path+"/"+filename)
                continue

            if len(detections_total)>0:
                bboxes_padded = []
                bboxes_masks_conditions = []
                max_pad_left, max_pad_right, max_pad_top, max_pad_bottom = 0, 0, 0, 0
                bbox_scaling_ratio =2.5
                mask_padding = 16
                mask_scaling_ratio = 1.2
                mask_dilation = 8
                mask_blur = 8
                for bbox in detections_total:
                    if bbox is None:
                        continue
                    if segment_enabled == True:
                        assert bbox_scaling_ratio >= 1.0
                        padding = int(0.5 * (bbox_scaling_ratio - 1) * max(bbox[2] - bbox[0],
                                                                                                    bbox[3] - bbox[1]))
                    else:
                        padding = mask_padding * 2
                    bbox_padded = bbox_padding(
                        bbox, padding
                    )
                    # bbox_padded = bbox_padding_to_square(bbox_padded)
                    bboxes_padded.append(
                        (bbox, bbox_padded)
                    )
                    pad_left = max(0, -bbox_padded[0])
                    pad_right = max(0, bbox_padded[2] - image.size[0])
                    pad_top = max(0, -bbox_padded[1])
                    pad_bottom = max(0, bbox_padded[3] - image.size[1])
                    max_pad_left = max(max_pad_left, pad_left)
                    max_pad_right = max(max_pad_right, pad_right)
                    max_pad_top = max(max_pad_top, pad_top)
                    max_pad_bottom = max(max_pad_bottom, pad_bottom)

                image_size_with_bias = (
                    image.size[0] + max_pad_left + max_pad_right,
                    image.size[1] + max_pad_top + max_pad_bottom,
                )
                pad_image = Image.new("RGB", image_size_with_bias)
                pad_image.paste(image, (max_pad_left, max_pad_top))
                detection = pad_image.copy()

                # breakpoint()
                # bboxes_masks = []
                for k, (bbox_padded, multi_condition) in enumerate(zip(bboxes_padded,multi_conditions)):
                    # 加上上一步得到整图 padding 偏移量
                    bbox_padded_with_bias = bbox_add_bias(
                        bbox_padded[1], (max_pad_left, max_pad_top)
                    )

                    if segment_enabled == True:
                        assert mask_scaling_ratio >= 1.0
                        bbox = bbox_padded[0]
                        padding = int(0.5 * (mask_scaling_ratio - 1) * max(bbox[2] - bbox[0],
                                                                                                    bbox[3] - bbox[1]))
                        bbox_padded_for_mask = bbox_padding(bbox, padding)
                        bbox_padded_for_mask_with_bias = bbox_add_bias(
                            bbox_padded_for_mask, (max_pad_left, max_pad_top)
                        )
                        mask = face_skin(pad_image, [bbox_padded_for_mask_with_bias],
                                                [[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13]])[0].convert("L")
                        mask = refine_mask(mask)
                        mask_area = np.sum(np.array(mask) // 255)
                        x1, y1, x2, y2 = bbox_padded_for_mask_with_bias
                        ratio = mask_area / ((x2 - x1) * (y2 - y1))
                        # 面积太小说明没有分割成功或可能是侧脸，替换为bbox的mask
                        if ratio < 0.1:
                            bbox_with_bias = bbox_add_bias(
                                bbox, (max_pad_left, max_pad_top)
                            )
                            mask = create_mask_from_bbox(
                                bbox_with_bias, image_size_with_bias
                            )
                            mask = mask.convert("L")
                    else:
                        bbox_padded_for_mask_with_bias = bbox_padding(
                            bbox_padded_with_bias,
                            -mask_padding,
                        )

                        mask = create_mask_from_bbox(
                            bbox_padded_for_mask_with_bias, image_size_with_bias
                        )
                        mask = mask.convert("L")

                    mask = mask_dilate(mask, mask_dilation)
                    # --再一次修复和膨胀--
                    mask = refine_mask(mask)
                    mask = mask_dilate(mask, mask_dilation)
                    # -------------------
                    mask = mask_gaussian_blur(mask, mask_blur)

                    bboxes_masks_conditions.append((
                        bbox_padded_with_bias, mask, multi_condition, bbox_padded[0]
                    ))

                groups_total, groups = group_rectangles(bboxes_masks_conditions, 512, 512)

                # merge conditions
                bboxes_padded_with_bias = []
                conditions = []
                bboxes = []
                for k, (bbox_padded_with_bias, mask) in enumerate(groups_total):
                    bboxes_padded_with_bias.append(bbox_padded_with_bias)  

                merge_conditions = []
                for k, (bbox_padded_with_bias, group) in enumerate(zip(bboxes_padded_with_bias, groups)):
                    merge_condition = Image.new("RGB", (bbox_padded_with_bias[2]-bbox_padded_with_bias[0], bbox_padded_with_bias[3]-bbox_padded_with_bias[1]), "black")
                    for _,_,condition,bbox in group:
                        rel_box = relative_box(bbox, bbox_padded_with_bias)
                        merge_condition.paste(condition,rel_box)
                    merge_conditions.append(merge_condition)

    
            #     print(groups_total)
                # import pdb;pdb.set_trace()


            # inpaint total
            if len(groups_total) > 0:
                bboxes_padded_with_bias = []
                masks = []
                for k, (bbox_padded_with_bias, mask) in enumerate(groups_total):
                    bboxes_padded_with_bias.append(bbox_padded_with_bias)
                    masks.append(mask)
                patch_sizes = [(x[2] - x[0], x[3] - x[1]) for x in bboxes_padded_with_bias]
                side_length, positions = arrange_in_square(patch_sizes)
                mosaic_image = Image.new("RGB", (side_length, side_length), "black")
                mosaic_mask = Image.new("L", (side_length, side_length), "black")
                mosaic_control_image = Image.new("RGB", (side_length, side_length), "black")
                for k, (bboxes_pad, mask, merge_condition, position) in enumerate(
                        zip(bboxes_padded_with_bias,
                            masks, merge_conditions,positions)):
                    crop_image = pad_image.crop(bboxes_pad)
                    crop_mask = mask.crop(bboxes_pad)
                    mosaic_image.paste(crop_image, position)
                    mosaic_mask.paste(crop_mask, position)
                    mosaic_control_image.paste(merge_condition,position)
                size = (width, height)
                input_image = mosaic_image.resize((size))
                control_image = mosaic_image.resize((size))
                input_mask = mosaic_mask.resize((size))

                inpainted_image = inpaint_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    mask_image=input_mask.convert("RGB"),
                    control_image=control_image,
                    controlnet_conditioning_scale=0.55,
                    strength=0.4,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator
                ).images[0]
                inpainted_image = inpainted_image.resize(mosaic_image.size)
                for bbox_pad, mask, position in zip(bboxes_padded_with_bias, masks, positions):
                                final_pad_image = composite(
                                    pad_image,
                                    mask,
                                    inpainted_image.crop(position),
                                    bbox_pad,
                                )
                                pad_image = final_pad_image
                final_image = final_pad_image.crop(
                    (
                        max_pad_left,
                        max_pad_top,
                        pad_image.size[0] - max_pad_right,
                        pad_image.size[1] - max_pad_bottom,
                    )
                )
                # import pdb;pdb.set_trace()

        ###################################################################################################
        ## 脸部和手部分开重绘
        if type =="mosaic*2":
            bboxes_padded_hand, multi_conditions, masks_hand, wholebody_pose= hamer_detector(
                        image, 2.0,
                        1.2,
                        is_cropped=True)
        # 脸部
            if yolo_detections_face is not None and len(yolo_detections_face)>0:
                bboxes_padded = []
                bboxes_masks = []
                max_pad_left, max_pad_right, max_pad_top, max_pad_bottom = 0, 0, 0, 0
                bbox_scaling_ratio =2.5
                mask_padding = 16
                mask_scaling_ratio = 1.2
                mask_dilation = 8
                mask_blur = 4
                for bbox in yolo_detections_face:
                    if bbox is None:
                        continue
                    if segment_enabled == True:
                        assert bbox_scaling_ratio >= 1.0
                        padding = int(0.5 * (bbox_scaling_ratio - 1) * max(bbox[2] - bbox[0],
                                                                                                    bbox[3] - bbox[1]))
                    else:
                        padding = mask_padding * 2
                    bbox_padded = bbox_padding(
                        bbox, padding
                    )
                    # bbox_padded = bbox_padding_to_square(bbox_padded)
                    bboxes_padded.append(
                        (bbox, bbox_padded)
                    )
                    pad_left = max(0, -bbox_padded[0])
                    pad_right = max(0, bbox_padded[2] - image.size[0])
                    pad_top = max(0, -bbox_padded[1])
                    pad_bottom = max(0, bbox_padded[3] - image.size[1])
                    max_pad_left = max(max_pad_left, pad_left)
                    max_pad_right = max(max_pad_right, pad_right)
                    max_pad_top = max(max_pad_top, pad_top)
                    max_pad_bottom = max(max_pad_bottom, pad_bottom)

                image_size_with_bias = (
                    image.size[0] + max_pad_left + max_pad_right,
                    image.size[1] + max_pad_top + max_pad_bottom,
                )
                pad_image = Image.new("RGB", image_size_with_bias)
                pad_image.paste(image, (max_pad_left, max_pad_top))
                con_image = Image.new("RGB", image_size_with_bias)
                con_image.paste(Image.fromarray(np.uint8(wholebody_pose)),(max_pad_left, max_pad_top))

                # breakpoint()
                # bboxes_masks = []
                for k, bbox_padded in enumerate(bboxes_padded):
                    # 加上上一步得到整图 padding 偏移量
                    bbox_padded_with_bias = bbox_add_bias(
                        bbox_padded[1], (max_pad_left, max_pad_top)
                    )

                    if segment_enabled == True:
                        assert mask_scaling_ratio >= 1.0
                        bbox = bbox_padded[0]
                        padding = int(0.5 * (mask_scaling_ratio - 1) * max(bbox[2] - bbox[0],
                                                                                                    bbox[3] - bbox[1]))
                        bbox_padded_for_mask = bbox_padding(bbox, padding)
                        bbox_padded_for_mask_with_bias = bbox_add_bias(
                            bbox_padded_for_mask, (max_pad_left, max_pad_top)
                        )
                        mask = face_skin(pad_image, [bbox_padded_for_mask_with_bias],
                                                [[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13]])[0].convert("L")
                        mask = refine_mask(mask)
                        mask_area = np.sum(np.array(mask) // 255)
                        x1, y1, x2, y2 = bbox_padded_for_mask_with_bias
                        ratio = mask_area / ((x2 - x1) * (y2 - y1))
                        # 面积太小说明没有分割成功或可能是侧脸，替换为bbox的mask
                        if ratio < 0.1:
                            bbox_with_bias = bbox_add_bias(
                                bbox, (max_pad_left, max_pad_top)
                            )
                            mask = create_mask_from_bbox(
                                bbox_with_bias, image_size_with_bias
                            )
                            mask = mask.convert("L")
                    else:
                        bbox_padded_for_mask_with_bias = bbox_padding(
                            bbox_padded_with_bias,
                            -mask_padding,
                        )

                        mask = create_mask_from_bbox(
                            bbox_padded_for_mask_with_bias, image_size_with_bias
                        )
                        mask = mask.convert("L")

                    mask = mask_dilate(mask, mask_dilation)
                    # --再一次修复和膨胀--
                    mask = refine_mask(mask)
                    mask = mask_dilate(mask, mask_dilation)
                    # -------------------
                    mask = mask_gaussian_blur(mask, mask_blur)
                    bboxes_masks.append((
                        bbox_padded_with_bias, mask
                    ))

                groups_face = group_rectangles_ori(bboxes_masks, 512, 512)


                # inpaint face
                if len(groups_face) > 0:
                    bboxes_padded_with_bias = []
                    masks = []
                    for k, (bbox_padded_with_bias, mask) in enumerate(groups_face):
                        bboxes_padded_with_bias.append(bbox_padded_with_bias)
                        masks.append(mask)
                    patch_sizes = [(x[2] - x[0], x[3] - x[1]) for x in bboxes_padded_with_bias]
                    side_length, positions = arrange_in_square(patch_sizes)
                    mosaic_image = Image.new("RGB", (side_length, side_length), "black")
                    mosaic_control_image = Image.new("RGB", (side_length, side_length), "black")
                    mosaic_mask = Image.new("L", (side_length, side_length), "black")
                    for k, (bboxes_pad, mask, position) in enumerate(
                            zip(bboxes_padded_with_bias,
                                masks, positions)):
                        crop_image = pad_image.crop(bboxes_pad)
                        crop_mask = mask.crop(bboxes_pad)
                        crop_control = con_image.crop(bboxes_pad)
                        mosaic_image.paste(crop_image, position)
                        mosaic_control_image.paste(crop_control, position)
                        mosaic_mask.paste(crop_mask, position)
                    size = (width, height)
                    input_image = mosaic_image.resize((size))
                    control_image = mosaic_control_image.resize((size))
                    input_mask = mosaic_mask.resize((size))

                    inpainted_image = inpaint_pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=input_image,
                        mask_image=input_mask.convert("RGB"),
                        control_image=control_image,
                        controlnet_conditioning_scale=0.5,
                        strength=0.5,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        generator=generator
                    ).images[0]
                    inpainted_image = inpainted_image.resize(mosaic_image.size)
                    for bbox_pad, mask, position in zip(bboxes_padded_with_bias, masks, positions):
                                    final_pad_image = composite(
                                        pad_image,
                                        mask,
                                        inpainted_image.crop(position),
                                        bbox_pad,
                                    )
                                    pad_image = final_pad_image
                    final_face_image = final_pad_image.crop(
                        (
                            max_pad_left,
                            max_pad_top,
                            pad_image.size[0] - max_pad_right,
                            pad_image.size[1] - max_pad_bottom,
                        )
                    )

            if yolo_detections_face is None or len(yolo_detections_face)==0:
                final_face_image = image.copy()
            
            if bboxes_padded_hand is None or len(bboxes_padded_hand)==0:
                final_image = final_face_image
            
            # inpaint hand   
            if bboxes_padded_hand is not None and len(bboxes_padded_hand)>0:
                mask_dilation = 8
                mask_blur = 4
                bbox_scaling_ratio = 2.8
                mask_scaling_ratio = 1.1
                bboxes_masks_conditions = []
                for k, (bbox, multi_condition) in enumerate(zip(bboxes_padded_hand,multi_conditions)):
                # for bbox in bboxes_padded_hand:
                    padding = int(0.5 * (bbox_scaling_ratio - 1) * max(bbox[2] - bbox[0],bbox[3] - bbox[1]))
                    padded_box = bbox_padding(bbox,padding)
                    bbox_padded_with_bias = bbox_padding_to_square(padded_box)
                    padding_mask = int(0.5 * (mask_scaling_ratio - 1) * max(bbox[2] - bbox[0],bbox[3] - bbox[1]))
                    padded_mask_box = bbox_padding(bbox,padding_mask)
                    mask = create_mask_from_bbox(
                            padded_mask_box, (width,height)
                        )
                    mask = mask.convert("L")
                    mask = mask_dilate(mask, mask_dilation)
                    # --再一次修复和膨胀--
                    mask = refine_mask(mask)
                    mask = mask_dilate(mask, mask_dilation)
                    # -------------------
                    mask = mask_gaussian_blur(mask, mask_blur)
                    bboxes_masks_conditions.append((
                        bbox_padded_with_bias, mask, multi_condition, bbox
                    ))
                groups_hand, groups = group_rectangles(bboxes_masks_conditions, 512, 512)

                # merge conditions
                bboxes_padded_with_bias = []
                for k, (bbox_padded_with_bias, mask) in enumerate(groups_hand):
                    bboxes_padded_with_bias.append(bbox_padded_with_bias)  

                merge_conditions = []
                for k, (bbox_padded_with_bias, group) in enumerate(zip(bboxes_padded_with_bias, groups)):
                    merge_condition = Image.new("RGB", (bbox_padded_with_bias[2]-bbox_padded_with_bias[0], bbox_padded_with_bias[3]-bbox_padded_with_bias[1]), "black")
                    for _,_,condition,bbox in group:
                        rel_box = relative_box(bbox, bbox_padded_with_bias)
                        merge_condition.paste(condition,rel_box)
                    merge_conditions.append(merge_condition)

                bboxes_padded_with_bias = []
                masks = []
                for k, (bbox_padded_with_bias, mask) in enumerate(groups_hand):
                    bboxes_padded_with_bias.append(bbox_padded_with_bias)
                    masks.append(mask)
                patch_sizes = [(x[2] - x[0], x[3] - x[1]) for x in bboxes_padded_with_bias]
                side_length, positions = arrange_in_square(patch_sizes)
                mosaic_image = Image.new("RGB", (side_length, side_length), "black")
                mosaic_control_image = Image.new("RGB", (side_length, side_length), "black")
                mosaic_mask = Image.new("L", (side_length, side_length), "black")
                for k, (bboxes_pad, mask, merge_condition, position) in enumerate(
                        zip(bboxes_padded_with_bias,
                            masks, merge_conditions, positions)):
                    crop_image = final_face_image.crop(bboxes_pad)
                    crop_mask = mask.crop(bboxes_pad)
                    mosaic_image.paste(crop_image, position)
                    mosaic_mask.paste(crop_mask, position)
                    mosaic_control_image.paste(merge_condition,position)
                size = (width, height)
                input_image = mosaic_image.resize((size))
                control_image = mosaic_control_image.resize((size))
                input_mask = mosaic_mask.resize((size))
    
                inpainted_image = inpaint_pipeline(
                    prompt,
                    negative_prompt=negative_prompt,
                    image=input_image,
                    mask_image=input_mask.convert("RGB"),
                    control_image=control_image,
                    controlnet_conditioning_scale=0.5,
                    strength=0.5,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator
                ).images[0]
                inpainted_image = inpainted_image.resize(mosaic_image.size)
                for bbox_pad, mask, position in zip(bboxes_padded_with_bias, masks, positions):
                                final_image = composite(
                                    final_face_image,
                                    mask,
                                    inpainted_image.crop(position),
                                    bbox_pad,
                                )
                                final_face_image = final_image  
                # import pdb;pdb.set_trace()           
            
        ###################################################################################################

        if type == "merge_crop":
            bboxes_padded_hand, multi_conditions, masks_hand, wholebody_pose= hamer_detector(
                        image, 2.5,
                        1.2,
                        is_cropped=True)
            if bboxes_padded_hand is not None and len(bboxes_padded_hand)>0:
                for bbox_padded_hand in bboxes_padded_hand:
                    # if calculate_area(yolo_detection_hand) > 400:
                    detections_total.append(bbox_padded_hand)
            if yolo_detections_face is not None and len(yolo_detections_face)>0:
                for yolo_detection_face in yolo_detections_face:
                    #过滤大脸
                    if calculate_area(yolo_detection_face)< 0.125*width*height:
                        xmin, ymin, xmax, ymax = yolo_detection_face
                        detections_total.append((int(xmin),int(ymin),int(xmax),int(ymax)))
                        multi_conditions.append(Image.new("RGB", (int(xmax)-int(xmin), int(ymax)-int(ymin)), "black"))
                        # multi_conditions.append(Image.fromarray(np.uint8(wholebody_pose)).crop((int(xmin),int(ymin),int(xmax),int(ymax))))

            if len(detections_total)==0:
                final_image = image
                final_image.save(output_path+"/"+filename)
                continue

            if len(detections_total)>0:
                bboxes_padded = []
                bboxes_masks_conditions = []
                max_pad_left, max_pad_right, max_pad_top, max_pad_bottom = 0, 0, 0, 0
                bbox_scaling_ratio =2.5
                mask_padding = 16
                mask_scaling_ratio = 1.2
                mask_dilation = 8
                mask_blur = 8
                for bbox in detections_total:
                    if bbox is None:
                        continue
                    if segment_enabled == True:
                        assert bbox_scaling_ratio >= 1.0
                        padding = int(0.5 * (bbox_scaling_ratio - 1) * max(bbox[2] - bbox[0],
                                                                                                    bbox[3] - bbox[1]))
                    else:
                        padding = mask_padding * 2
                    bbox_padded = bbox_padding(
                        bbox, padding
                    )
                    # bbox_padded = bbox_padding_to_square(bbox_padded)
                    bboxes_padded.append(
                        (bbox, bbox_padded)
                    )
                    pad_left = max(0, -bbox_padded[0])
                    pad_right = max(0, bbox_padded[2] - image.size[0])
                    pad_top = max(0, -bbox_padded[1])
                    pad_bottom = max(0, bbox_padded[3] - image.size[1])
                    max_pad_left = max(max_pad_left, pad_left)
                    max_pad_right = max(max_pad_right, pad_right)
                    max_pad_top = max(max_pad_top, pad_top)
                    max_pad_bottom = max(max_pad_bottom, pad_bottom)

                image_size_with_bias = (
                    image.size[0] + max_pad_left + max_pad_right,
                    image.size[1] + max_pad_top + max_pad_bottom,
                )
                pad_image = Image.new("RGB", image_size_with_bias)
                pad_image.paste(image, (max_pad_left, max_pad_top))
                detection = pad_image.copy()

                # breakpoint()
                # bboxes_masks = []
                for k, (bbox_padded, multi_condition) in enumerate(zip(bboxes_padded,multi_conditions)):
                    # 加上上一步得到整图 padding 偏移量
                    bbox_padded_with_bias = bbox_add_bias(
                        bbox_padded[1], (max_pad_left, max_pad_top)
                    )

                    if segment_enabled == True:
                        assert mask_scaling_ratio >= 1.0
                        bbox = bbox_padded[0]
                        padding = int(0.5 * (mask_scaling_ratio - 1) * max(bbox[2] - bbox[0],
                                                                                                    bbox[3] - bbox[1]))
                        bbox_padded_for_mask = bbox_padding(bbox, padding)
                        bbox_padded_for_mask_with_bias = bbox_add_bias(
                            bbox_padded_for_mask, (max_pad_left, max_pad_top)
                        )
                        mask = face_skin(pad_image, [bbox_padded_for_mask_with_bias],
                                                [[1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13]])[0].convert("L")
                        mask = refine_mask(mask)
                        mask_area = np.sum(np.array(mask) // 255)
                        x1, y1, x2, y2 = bbox_padded_for_mask_with_bias
                        ratio = mask_area / ((x2 - x1) * (y2 - y1))
                        # 面积太小说明没有分割成功或可能是侧脸，替换为bbox的mask
                        if ratio < 0.1:
                            bbox_with_bias = bbox_add_bias(
                                bbox, (max_pad_left, max_pad_top)
                            )
                            mask = create_mask_from_bbox(
                                bbox_with_bias, image_size_with_bias
                            )
                            mask = mask.convert("L")
                    else:
                        bbox_padded_for_mask_with_bias = bbox_padding(
                            bbox_padded_with_bias,
                            -mask_padding,
                        )

                        mask = create_mask_from_bbox(
                            bbox_padded_for_mask_with_bias, image_size_with_bias
                        )
                        mask = mask.convert("L")

                    mask = mask_dilate(mask, mask_dilation)
                    # --再一次修复和膨胀--
                    mask = refine_mask(mask)
                    mask = mask_dilate(mask, mask_dilation)
                    # -------------------
                    mask = mask_gaussian_blur(mask, mask_blur)
                    bboxes_masks_conditions.append((
                        bbox_padded_with_bias, mask, multi_condition, bbox_padded[0]
                    ))

                groups_total, groups = group_rectangles(bboxes_masks_conditions, 512, 512)

                # merge conditions
                bboxes_padded_with_bias = []
                for k, (bbox_padded_with_bias, mask) in enumerate(groups_total):
                    bboxes_padded_with_bias.append(bbox_padded_with_bias)  

                merge_conditions = []
                for k, (bbox_padded_with_bias, group) in enumerate(zip(bboxes_padded_with_bias, groups)):
                    merge_condition = Image.new("RGB", (bbox_padded_with_bias[2]-bbox_padded_with_bias[0], bbox_padded_with_bias[3]-bbox_padded_with_bias[1]), "black")
                    for _,_,condition,bbox in group:
                        rel_box = relative_box(bbox, bbox_padded_with_bias)
                        merge_condition.paste(condition,rel_box)
                    merge_conditions.append(merge_condition)


            for k, ((bbox_padded_with_bias, mask), merge_condition) in enumerate(zip(groups_total, merge_conditions)):
                            bbox_padded_with_bias = expand_to_square(
                                bbox_padded_with_bias,
                                pad_image.size[0],
                                pad_image.size[1]
                            )
                            crop_image = pad_image.crop(bbox_padded_with_bias)
                            mask = mask.convert("RGB")
                            crop_mask = mask.crop(bbox_padded_with_bias)
                            n_prompt = "fake 3D rendered image, deforemd, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"
                            size = (width, height)
                            input_image = crop_image.resize(size)
                            control_image = merge_condition.resize(size)
                            input_mask = crop_mask.resize(size)
                            #import pdb;pdb.set_trace()
                            inpainted_image = inpaint_pipeline(
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                image=input_image,
                                mask_image=input_mask.convert("RGB"),
                                control_image=control_image,
                                controlnet_conditioning_scale=0.55,
                                strength=0.4,
                                num_inference_steps=30,
                                guidance_scale=7.5,
                                generator=generator
                            ).images[0]
                            inpainted_image = inpainted_image.resize(crop_image.size)
                            mask = mask.convert("L")
                            final_pad_image = composite(
                                pad_image,
                                mask,
                                inpainted_image,
                                bbox_padded_with_bias,
                            )
                            pad_image = final_pad_image
            final_image = final_pad_image.crop(
                (
                    max_pad_left,
                    max_pad_top,
                    pad_image.size[0] - max_pad_right,
                    pad_image.size[1] - max_pad_bottom,
                )
            )
            # import pdb;pdb.set_trace()
        final_image.save(output_path+"/"+filename)
        # image_grid = make_image_grid(
        #         [image,image_out],
        #         rows=2,
        #         cols=1,
        #     )
        # image_grid.save(output_path+"/"+filename)
        # import pdb;pdb.set_trace()
    end = time.time()
    print((end-start)/1000)   

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)