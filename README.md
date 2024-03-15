# sd_scriptviz

## Dependency 
- Python 3.10.9

## Installation

```
git clone https://github.com/bishalsahooviga/sd_scriptviz.git
cd sd_scriptviz
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```


## Usage

```
from gen_images.sdxl_char_lora import get_images
from gen_image.controlnet import get_controlnet_images
```
## Functions and arguments

### get_images
get_images(sdxl_ckpt,char_lora,prompt)
- sdxl_ckpt : path for sdxl checkpoint
- char_lora : path for character lora
- prompt : prompt for the image
  
### get_controlnet_images
get_controlnet_images(sdxl_ckpt,prompt,controlnet_ckpt,controlnet_conditioning_scale,image,type)
- sdxl_ckpt : path for sdxl checkpoint
- prompt : prompt for the image
- controlnet_ckpt : path for controlnet checkpoint 
- controlnet_conditioning_scale : controlnet conditioning scale
- image : path for the controlnet image
- type : controlnet type

## API

API can be accessed by ``` python app.py ```

### get_images
``` curl -X POST -H "Content-Type: application/json" -d '{"prompt":"your_prompt_here", "sdxl_ckpt":"your_sdxl_ckpt_path_here", "char_lora":"your_char_lora_path_here"}' http://127.0.0.1:5000/generate-image ```

### get_controlnet_images
``` curl -X POST -H "Content-Type: application/json" -d '{"prompt":"your_prompt_here", "sdxl_ckpt":"your_sdxl_ckpt_path_here", "controlnet_ckpt":"your_controlnet_ckpt_path_here", "controlnet_conditioning_scale":your_value_here, "image":"your_image_here", "type":"your_type_here"}' http://127.0.0.1:5000/generate-controlnet-image ```

