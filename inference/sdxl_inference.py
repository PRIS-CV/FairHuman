import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import HfArgumentParser
import dataclasses
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLPipeline, ControlNetModel, UNet2DConditionModel, AutoencoderKL, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image

@dataclasses.dataclass
class InferenceArgs:
    eval_txt_path: str | None = None
    output: str | None = "./inference"
    base_model_path: str | None = "stabilityai/sd_xl_base_1.0"
    vae_path: str | None = "madebyollin/sdxl-vae-fp16-fix"
    lora_path: str | None = None
    lora_weight: float=0.0

def main(args: InferenceArgs):
    # initialize the models and pipeline
    pretrained_model_name_or_path =  args.base_model_path
    vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path, vae=vae, torch_dtype=torch.float16
    )
    # load lora
    if args.lora_path is not None and args.lora_weight > 0:
        pipe.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name='default_lora')
        pipe.set_adapters(["default_lora"], adapter_weights=[args.lora_weight])

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
            prompt = prompt+", realistic, high quality, 8k" # recommand to add
            negative_prompt = "fake 3D rendered image, deforemd, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"
            image = pipe(prompt, 
                        negative_prompt=negative_prompt,
                        generator=generator, 
                        num_inference_steps=30, 
                        guidance_scale=7.5,
            ).images[0]
            output_path = args.output+str(i)
            os.makedirs(output_path, exist_ok=True)
            image.save(output_path+"/"+"{}_fg_test.png".format(args.lora_weight))

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)