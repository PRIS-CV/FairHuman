import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import dataclasses
from transformers import HfArgumentParser
from diffusers import StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline, ControlNetModel, UNet2DConditionModel, AutoencoderKL, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image

@dataclasses.dataclass
class InferenceArgs:
    eval_txt_path: str | None = None
    condition_path: str | None = None
    output: str | None = "./inference"
    base_model_path: str | None = "stabilityai/sd_xl_base_1.0"
    controlnet_path: str | None = None
    vae_path: str | None = "madebyollin/sdxl-vae-fp16-fix"
    lora_path: str | None = None
    lora_weight: float=0.0

def main(args: InferenceArgs):
    # initialize the models and pipeline
    vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base_model_path, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
    )

    # lora
    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name='text_lora')
        pipe.set_adapters(["text_lora"], adapter_weights=[args.lora_weight])

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # memory optimization.
    pipe.to("cuda")
    # pipe.enable_model_cpu_offload()

    # inference
    prompt_list = open(args.eval_txt_path, "r").readlines()
    prompts = [prompt.strip() for prompt in prompt_list]
    generator = torch.Generator(device="cuda").manual_seed(0)
    for i, prompt in enumerate(prompts):
            prompt = prompt+", realistic, high quality, 8k"
            negative_prompt = "fake 3D rendered image, deforemd, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"
            # generate image
            # scales = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            scale = 0.5
            control_image = Image.open(args.condition_path+str(i)+".png")
            image = pipe(prompt, 
                            negative_prompt=negative_prompt,
                            controlnet_conditioning_scale=scale,
                            image=control_image,
                            generator=generator, 
                            num_inference_steps=30, 
                            guidance_scale=7.5,
                # control_guidance_start=0.1,
                # control_guidance_end=0.8,
            ).images[0]
            output_path = args.output+str(i)
            os.makedirs(output_path, exist_ok=True)
            image.save(output_path+"/"+"{}_test_.png".format(scale))
            
if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)