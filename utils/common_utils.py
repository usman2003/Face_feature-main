import torch
import random
from torch.nn import functional as F
from PIL import Image


def tensor2im(tensor):
    # Ensure tensor is detached and moved to CPU
    array = tensor.cpu().detach().numpy()
    print(f"Initial shape (C, H, W): {array.shape}")

    # Transpose from (C, H, W) to (H, W, C)
    array = np.transpose(array, (1, 2, 0))
    print(f"Transposed shape (H, W, C): {array.shape}")

    # Normalize to range [0, 1]
    array = (array + 1) / 2
    array = np.clip(array, 0, 1)
    print(f"Clipped array: min={array.min()}, max={array.max()}")

    # Convert to range [0, 255]
    array = (array * 255).astype(np.uint8)
    print(f"Final shape and type: {array.shape}, {array.dtype}")

    # Convert to PIL Image
    return Image.fromarray(array)


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
