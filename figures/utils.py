import numpy as np

from flip.flip_api import *
from flip.data import *


def compute_error(image: np.ndarray, ref: np.ndarray, error_type: str):
    if image.shape != ref.shape:
        return np.random.rand(*ref.shape)

    if error_type == 'FLIP':
        # ref, image = check_nans(ref, image)
        flip = compute_ldrflip(HWCtoCHW(ref), HWCtoCHW(image))
        index_map = np.floor(255.0 * flip.squeeze(0))
        error_map = sRGB_to_linear(CHWtoHWC(index2color(
            index_map, get_magma_map()))).astype(np.float32)
        return error_map
    if error_type == 'Error':
        return (image - ref)
    elif error_type == 'Abs Error':
        abserror = np.abs(image - ref)
        original_shape = abserror.shape
        if len(original_shape) == 3:
            abserror = abserror[:, :, 0]

        index_map = np.floor(255.0 * abserror)
        error_map = sRGB_to_linear(CHWtoHWC(index2color(
            index_map, get_magma_map()))).astype(np.float32)
        return error_map
    elif error_type == 'Squared Error':
        return (image - ref)**2
    elif error_type == 'Rel Abs Error':
        return np.abs(image - ref) / (ref + 0.01)
    elif error_type == 'Rel Squared Error':
        return (image - ref)**2 / (ref**2 + 0.01)
    else:
        return image


def sRGB_to_linear(image):
    outSign = np.sign(image)
    image = np.abs(image)
    mask = (image <= 0.04045)
    linear = outSign * image / 12.92
    linear[~mask] = outSign[~mask] * ((image[~mask] + 0.055) / 1.055)**2.4
    return linear
