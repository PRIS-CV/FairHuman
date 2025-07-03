import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import numpy as np

from diffusers.utils import load_image,make_image_grid
from PIL import Image, ImageDraw
from Hamer.hamer_detector_yolo_global import HamerDetector

model_dir = "./models"
device = "cuda"

hamer_detector = HamerDetector(
            model_dir=model_dir,
            rescale_factor=2.0,
            device=device
)

path_list = os.listdir("/data01/wangyx/train_repo/1000_test/sdxl/gen/lora/gen_imgs_ori")
path_list.sort(key=lambda x:int(x.split('_')[0]))

for i, filename in enumerate(path_list):
    image_path ="/data01/wangyx/train_repo/1000_test/sdxl/gen/lora/gen_imgs_ori/"+filename
    image = load_image(image_path).convert("RGB")
    bboxes_padded_hand, multi_conditions_wholebody, depth_conditions_wholebody, pose_conditions_wholebody, mesh_conditions_wholebody, masks_hand, dwpose = hamer_detector(
                        image, 2.5,
                        1.2,
                        is_cropped=False)
    if dwpose is None:
        control_image= Image.new("RGB", (1024,1024), "black")
    elif len(multi_conditions_wholebody)==0:
         control_image = Image.fromarray(np.uint8(dwpose))
    else:
         control_image = multi_conditions_wholebody[0]
    output_path ="/data01/wangyx/train_repo/1000_test/sdxl/gen/control/ori_conditions"
    os.makedirs(output_path, exist_ok=True) 
    control_image.save(output_path+"/"+"{}.png".format(i))