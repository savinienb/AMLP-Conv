
import numpy as np
import SimpleITK as sitk

from utils.timer import Timer


def sitk_to_np_no_copy(image_sitk):
    return sitk.GetArrayViewFromImage(image_sitk)


def sitk_to_np(image_sitk, type=None):
    if type is None:
        return sitk.GetArrayFromImage(image_sitk)
    else:
        return sitk.GetArrayViewFromImage(image_sitk).astype(type)


def np_to_sitk(image_np, type=None, is_vector=False):
    if type is None:
        return sitk.GetImageFromArray(image_np, is_vector)
    else:
        return sitk.GetImageFromArray(image_np.astype(type), is_vector)


def sitk_list_to_np(image_list_sitk, type=None, axis=0):
    image_list_np = []
    # with Timer('    [sitk_np.py] loop sitk_to_np_no_copy'):
    for image_sitk in image_list_sitk:
        image_list_np.append(sitk_to_np_no_copy(image_sitk))
        # image_list_np.append(sitk_to_np(image_sitk))  # TODO very slow due to copying in 1.20 on turing and Tesla, but not on chris-payer-icg

    # with Timer('    [sitk_np.py] stack'):
    #     np_image = np.stack(image_list_np, axis=axis)  # TODO very slow due to copying in 1.20 on turing and Tesla, but not on chris-payer-icg

    # with Timer('    [sitk_np.py] zeros + for loop'):
    np_image = np.zeros((len(image_list_np),) + image_list_np[0].shape)
    for i, v in enumerate(image_list_np):
        np_image[i] = v
    if axis != 0:
        np_image = np.moveaxis(np_image, 0, axis)

    # with Timer('    [sitk_np.py] astype'):
    if type is not None:
        np_image = np_image.astype(type)
    return np_image
