import os
import torch
import base64
import io
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from utils.common_utils import tensor2im, tensor2im_no_tfm
from datasets.transforms import transforms_registry
from runners.inference_runners import FSEInferenceRunner


def base64_to_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image


def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_mask(base64_image, trash=0.995):
    from models.farl.farl import Masker

    orig_img = base64_to_image(base64_image).convert("RGB")
    transform = transforms.ToTensor()
    orig_img_tensor = transform(orig_img)

    orig_img_tensor = (orig_img_tensor.unsqueeze(0) * 255).long().cuda()

    with torch.inference_mode():
        for detector_trash in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
            masker = Masker(trash=detector_trash)
            faces = masker.face_detector(orig_img_tensor)
            if len(faces['image_ids']) != 0:
                break
        if len(faces['image_ids']) == 0:
            raise ValueError("Masker's face detector can't find face on your image :c Maybe you forgot to align it?")
        faces = masker.face_parser(orig_img_tensor, faces)

    background_mask = F.sigmoid(faces['seg']['logits'][:, 0])
    background_mask = background_mask[0].unsqueeze(0)
    background_mask = (background_mask >= trash).cpu()
    to_save = (background_mask[0] * 255).long().numpy()
    mask = Image.fromarray(to_save.astype(np.uint8)).convert("1")

    backfround_tens = orig_img_tensor[0].cpu() / 255 * background_mask.float().repeat(3, 1, 1)
    background = tensor2im_no_tfm(backfround_tens)
    background_base64 = image_to_base64(background)

    face_tens = orig_img_tensor[0].cpu() / 255 * (1 - background_mask.float()).repeat(3, 1, 1)
    face = tensor2im_no_tfm(face_tens)
    face_base64 = image_to_base64(face)

    return image_to_base64(mask), background_base64, face_base64


def run_alignment(base64_image):
    import dlib
    from scripts.align_all_parallel import align_face

    predictor = dlib.shape_predictor("pretrained_models/shape_predictor_68_face_landmarks.dat")
    #image = base64_to_image(base64_image)
    aligned_image = align_face(base64_code=base64_image, predictor=predictor)
    return image_to_base64(aligned_image.convert("RGB"))


class SimpleRunner:
    def __init__(self, editor_ckpt_pth: str, simple_config_pth: str = "configs/simple_inference.yaml"):
        config = OmegaConf.load(simple_config_pth)
        config.model.checkpoint_path = editor_ckpt_pth
        config.methods_args.fse_full = {}

        self.inference_runner = FSEInferenceRunner(config)
        self.inference_runner.setup()
        self.inference_runner.method.eval()
        self.inference_runner.method.decoder = self.inference_runner.method.decoder.float()

    def edit(self, base64_image: str, editing_name: str, edited_power: float, align: bool = False, use_mask: bool = False, mask_trashold=0.995, mask_base64: str = None, save_e4e=False, save_inversion=False):
        if align:
            aligned_image_base64 = run_alignment(base64_image)
            base64_image = aligned_image_base64

        if use_mask and mask_base64 is None:
            print("Preparing mask")
            mask_base64, _, _ = extract_mask(base64_image, trash=mask_trashold)
            print("Done")

        if use_mask and mask_base64 is not None:
            print("Using provided mask")
            mask = base64_to_image(mask_base64).convert("RGB")
            mask = transforms.ToTensor()(mask).unsqueeze(0).to(self.inference_runner.device)
        else:
            mask = None

        orig_img = base64_to_image(base64_image).convert("RGB")
        transform_dict = transforms_registry["face_1024"]().get_transforms()
        orig_img = transform_dict["test"](orig_img).unsqueeze(0).to(self.inference_runner.device)

        inv_images, inversion_results = self.inference_runner._run_on_batch(orig_img)
        edited_image = self.inference_runner._run_editing_on_batch(
            method_res_batch=inversion_results, 
            editing_name=editing_name, 
            editing_degrees=[edited_power],
            mask=mask,
            return_e4e=save_e4e
        )

        edited_images = {
            "edited_image": tensor2im(edited_image[0][0])
        }

        if save_inversion:
            inv_image = tensor2im(inv_images[0])
            edited_images["inversion"] = inv_image

        if save_e4e:
            edited_image, e4e_inv, e4e_edit = edited_image

            e4e_inv_image = tensor2im(e4e_inv[0])
            edited_images["e4e_inversion"] = e4e_inv_image

            e4e_edit_image = tensor2im(e4e_edit[0])
            edited_images["e4e_edit"] = e4e_edit_image

        return edited_images

    def available_editings(self):
        edits_types = []
        for field in dir(self.inference_runner.latent_editor):
            if "directions" in field.split("_"):
                edits_types.append(field)

        print("This code handles the following editing directions for following methods:")
        available_directions = {}
        for edit_type in edits_types:
            print(edit_type + ":")
            edit_type_directions = getattr(self.inference_runner.latent_editor, edit_type, None).keys()
            for direction in edit_type_directions:
                print("\t" + direction)
        print(GLOBAL_DIRECTIONS_DESC)


GLOBAL_DIRECTIONS_DESC = """
You can also use directions from text prompts via StyleClip Global Mapper (https://arxiv.org/abs/2103.17249).
Such directions look as follows: "styleclip_global_{neutral prompt}_{target prompt}_{disentanglement}" where
neutral prompt -- some neutral description of the original image (e.g. "a face")
target prompt -- text that contains the desired edit (e.g. "a smiling face")
disentanglement -- positive number, the more this attribute - the more related attributes will also be changed (e.g. 
for grey hair editing, wrinkle, skin colour and glasses may also be edited)

Example: "styleclip_global_face with hair_face with black hair_0.18"

More information about the purpose of directions and their approximate power range can be found in available_directions.txt.
"""
