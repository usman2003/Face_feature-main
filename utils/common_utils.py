import torch
import random
from torch.nn import functional as F
from PIL import Image


def tensor2im(var):
    var_np = var.cpu().detach().numpy()  # Ensure tensor is detached and in numpy array format
    var_np = np.transpose(var_np, (1, 2, 0))  # Transpose to (H, W, C) format if necessary
    var_np = (var_np * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    
    try:
        img = Image.fromarray(var_np)
        return img
    except Exception as e:
        print(f"Error converting array to PIL Image: {e}")
        return None

def tensor2im_no_tfm(var):
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = var * 255
    return Image.fromarray(var.astype("uint8"))


def printer(obj, tabs=0):
    for (key, value) in obj.items():
        try:
            _ = value.items()
            print(" " * tabs + str(key) + ":")
            printer(value, tabs + 4)
        except:
            print(f" " * tabs + str(key) + " : " + str(value))


def get_keys(d, name, key="state_dict"):
    if key in d:
        d = d[key]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name) + 1] == name + '.'}
    return d_filt


def setup_seed(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
