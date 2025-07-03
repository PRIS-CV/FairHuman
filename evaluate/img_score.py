import torch
from PIL import Image
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import requests
from typing import Union
import huggingface_hub
from hpsv2.utils import root_path, hps_version_map

warnings.filterwarnings("ignore", category=UserWarning)

model_dict = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            '/nvfile-heatstorage/zangxh/intern/wangyx/evaluation/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val

def score(img_path: Union[list, str, Image.Image], prompt: str, cp: str = None, hps_version: str = "v2.0") -> list:

    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    # # check if the checkpoint exists
    # if not os.path.exists(root_path):
    #     os.makedirs(root_path)
    # if cp is None:
    #     cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    
    checkpoint = torch.load(cp, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')
    model = model.to(device)
    model.eval()
    
    
    if isinstance(img_path, list):
        result = []
        for one_img_path in img_path:
            # Load your image and prompt
            with torch.no_grad():
                # Process the image
                if isinstance(one_img_path, str):
                    image = preprocess_val(Image.open(one_img_path)).unsqueeze(0).to(device=device, non_blocking=True)
                elif isinstance(one_img_path, Image.Image):
                    image = preprocess_val(one_img_path).unsqueeze(0).to(device=device, non_blocking=True)
                else:
                    raise TypeError('The type of parameter img_path is illegal.')
                # Process the prompt
                text = tokenizer([prompt]).to(device=device, non_blocking=True)
                # Calculate the HPS
                with torch.cuda.amp.autocast():
                    outputs = model(image, text)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T

                    hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            result.append(hps_score[0])    
        return result
    elif isinstance(img_path, str):
        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            image = preprocess_val(Image.open(img_path)).unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return [hps_score[0]]
    elif isinstance(img_path, Image.Image):
        # Load your image and prompt
        with torch.no_grad():
            # Process the image
            image = preprocess_val(img_path).unsqueeze(0).to(device=device, non_blocking=True)
            # Process the prompt
            text = tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = model(image, text)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return [hps_score[0]]
    else:
        raise TypeError('The type of parameter img_path is illegal.')
        

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/nvfile-heatstorage/zangxh/intern/wangyx/evaluation/HPSv2/HPS_v2.1_compressed.pt', help='Path to the model checkpoint')

    args = parser.parse_args()

    prompt_list = open("/nvfile-heatstorage/zangxh/intern/wangyx/prompt_test.txt", "r").readlines()
    prompts = [prompt.strip() for prompt in prompt_list]

    hps_scores =[]
    for i, prompt in enumerate(prompts): 
            img_paths = []
            for img_path in os.listdir("/nvfile-heatstorage/zangxh/intern/wangyx/train_repo/1000_test/sdxl/gen/control_new/"+str(i)):
                  img_paths.append("/nvfile-heatstorage/zangxh/intern/wangyx/train_repo/1000_test/sdxl/gen/control_new/"+str(i)+"/"+img_path)
            hps_score = score(img_paths,prompt, args.checkpoint)

            mean_hps_score = sum(hps_score)/len(hps_score)
            hps_scores.append(mean_hps_score)
    print("HPSv2 score:", sum(hps_scores)/len(hps_scores))