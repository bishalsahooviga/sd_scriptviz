from flask import Flask, request, jsonify, render_template
from gen_images.sdxl_char_lora import get_images
from gen_images.controlnet import get_controlnet_images

app = Flask(__name__)



@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt')
    sdxl_ckpt=data.get('sdxl_ckpt')
    char_lora=data.get('char_lora')


    if not prompt:
        return jsonify({'error': 'Prompt is required!'}), 400
    image_path = get_images(sdxl_ckpt=sdxl_ckpt,char_lora=char_lora,prompt=prompt)

    return jsonify({'image_path': image_path})



@app.route('/generate-controlnet-image', methods=['POST'])
def generate_controlnet_image():
    data = request.json
    prompt = data.get('prompt')
    sdxl_ckpt=data.get('sdxl_ckpt')
    controlnet_ckpt=data.get('controlnet_ckpt')
    controlnet_conditioning_scale=float(data.get('controlnet_conditioning_scale'))
    image=data.get('image')
    type=data.get('type')

    if not prompt:
        return jsonify({'error': 'Prompt is required!'}), 400
    image_path = get_controlnet_images(sdxl_ckpt,prompt,controlnet_ckpt,controlnet_conditioning_scale,image)
    
    return jsonify({'image_path': image_path})

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)