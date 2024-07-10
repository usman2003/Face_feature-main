import sys
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from runners.simple_runner import SimpleRunner
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive environments

# Initialize the runner
runner = SimpleRunner(editor_ckpt_pth="pretrained_models/sfe_editor_light.pt")

def base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def handle_job(job):
    try:
        job_input = job["input"]
        
        base64_image = job_input.get("base64_image")
        editing_name = job_input.get("editing_name", "age")
        edited_power = job_input.get("edited_power", -10)
        align = job_input.get("align", True)
        use_mask = job_input.get("use_mask", False)
        mask_trashold = job_input.get("mask_trashold", 0.995)
        show_inversion_result = job_input.get("save_inversion", False)
        show_e4e_approximation = job_input.get("save_e4e", False)

        if base64_image:
            # Process the image using the runner.edit function
            edited_images = runner.edit(
                base64_image=base64_image,
                editing_name=editing_name,
                edited_power=edited_power,
                align=align,
                save_inversion=show_inversion_result,
                use_mask=use_mask,
                mask_trashold=mask_trashold,
                save_e4e=show_e4e_approximation
            )

            response = {
                "edited_image": edited_images["edited_image"]
            }

            if show_inversion_result:
                response["inversion"] = edited_images.get("inversion", "")

            if show_e4e_approximation:
                response["e4e_inversion"] = edited_images.get("e4e_inversion", "")
                response["e4e_edit"] = edited_images.get("e4e_edit", "")

            return response

        else:
            return {"error": "No image provided"}

    except Exception as e:
        return {"error": str(e)}

# Run the serverless function
import runpod
runpod.serverless.start({"handler": handle_job})
