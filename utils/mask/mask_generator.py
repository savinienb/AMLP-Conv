import numpy as np


def generate_mask(output_size):
    h = output_size[0]
    w = output_size[1]
    center = [float((w - 1) / 2), float((h - 1) / 2)]
    radius = min( h /2, w/ 2)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    mask = np.expand_dims(mask, axis=0)
    return mask.astype(np.float32)