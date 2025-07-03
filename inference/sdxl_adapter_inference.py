import os
import random
import dataclasses
from transformers import HfArgumentParser
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteSchedulerTest, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
import torch


@dataclasses.dataclass
class InferenceArgs:
    eval_txt_path: str | None = None
    condition_path: str | None = None
    output: str | None = "./inference"
    base_model_path: str | None = "stabilityai/sd_xl_base_1.0"
    adapter_path: str | None = None
    vae_path: str | None = "madebyollin/sdxl-vae-fp16-fix"

def main(args: InferenceArgs):
    vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float16)
    adapter = T2IAdapter.from_pretrained(args.adapter_path, torch_dtype=torch.float16)
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        args.base_model_path, vae=vae, adapter=adapter, torch_dtype=torch.float16
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = EulerAncestralDiscreteSchedulerTest.from_config(pipe.scheduler.config)
    # remove following line if xformers is not installed or when using Torch 2.0.
    # pipe.enable_xformers_memory_efficient_attention()
    # # memory optimization.
    # pipe.enable_model_cpu_offload()
    pipe.to("cuda")

    prompt_list = open(args.eval_txt_path, "r").readlines()
    prompts = [prompt.strip() for prompt in prompt_list]
    generator = torch.Generator(device="cuda").manual_seed(0)
    # inference
    for i, prompt in enumerate(prompts):
            prompt = prompt+", realistic, high quality, 8k"
            negative_prompt = "fake 3D rendered image, deforemd, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"
            # generate image
            # scales = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
            scale = 0.5
            control_image = Image.open(args.condition_path+str(i)+".png")
            image = pipe(prompt, 
                            negative_prompt=negative_prompt,
                            adapter_conditioning_scale=scale,
                            image=control_image,
                            generator=generator, 
                            num_inference_steps=30, 
                            guidance_scale=7.5,
                # adapter_conditioning_factor=0.1,
            ).images[0]
            output_path = args.output+str(i)
            os.makedirs(output_path, exist_ok=True)
            image.save(output_path+"/"+"{}_test_.png".format(scale))

if __name__ == "__main__":
    parser = HfArgumentParser([InferenceArgs])
    args = parser.parse_args_into_dataclasses()[0]
    main(args)