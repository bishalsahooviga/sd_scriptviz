from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image
import time
import os

#example inputs
prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
sdxl_ckpt="/home/bishal.sahoo/Desktop/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors"
controlnet_ckpt="/home/bishal.sahoo/snap/firefox/common"
controlnet_conditioning_scale = 0.5 
image = "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"




def get_controlnet_images(sdxl_ckpt,prompt,controlnet_ckpt,controlnet_conditioning_scale,image,type):
    image = load_image(image)
   
    if type == 'lineart':
        control_model=="TencentARC/t2i-adapter-lineart-sdxl-1.0"
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        line_art = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        control_image = Image.fromarray(line_art)
    else:
        control_model="diffusers/controlnet-canny-sdxl-1.0"
        # get canny image
        image = cv2.Canny(np.array(image), 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)
        
    controlnet = ControlNetModel.from_pretrained(control_model, torch_dtype=torch.float16,cache_dir=controlnet_ckpt,keep_in_memory=True,load_from_cache=True)
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16,cache_dir=controlnet_ckpt,keep_in_memory=True,load_from_cache=True)
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(sdxl_ckpt, controlnet=controlnet, vae=vae, torch_dtype=torch.float16)
    # pipe.enable_model_cpu_offload()


    
    
    
    # generate image
    image = pipe(prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=control_image).images[0]
    filename = f"{int(time.time())}.jpg"
    image.save(filename)
    image_path = os.path.abspath(filename)
    return image_path