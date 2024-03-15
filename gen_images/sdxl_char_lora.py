from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
import time
import os

sdxl_ckpt="/home/bishal.sahoo/Desktop/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors"
sdxl_refiner_ckpt="/home/bishal.sahoo/Desktop/stable-diffusion-webui/models/Stable-diffusion/sd_xl_refiner_1.0_0.9vae.safetensors"
char_lora="/mnt/x/viga/sheet_data/Models/DOROTHY/BODY/drzy_body.safetensors"
prompt = "A DorothyWoz dive and  floating"

def get_images(sdxl_ckpt,char_lora,prompt):
    pipeline = StableDiffusionXLPipeline.from_single_file(sdxl_ckpt, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    pipeline.load_lora_weights(char_lora)


    
    image = pipeline(prompt=prompt).images[0]
    filename = f"{int(time.time())}.jpg"
    image.save(filename)
    image_path = os.path.abspath(filename)
    return image_path