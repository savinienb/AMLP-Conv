
from scipy.ndimage import gaussian_filter
import numpy as np


class FrangiFilter(object):

    def __call__(self, img):

        img_yy = []
        out = []
    #    OUT = np.zeros(img.shape)
        for i in range(5):
            s = 0.5 * (i + 2)
            img_yy.append(gaussian_filter(img, sigma=s, order=[0, 2, 0]))
            I = np.maximum(-img_yy[i], 0)
            I /= np.amax(I)
            out.append(I)
      #      write_np(out[i], 'plate' + str(i) + '.nii')

        OUT = np.amax(out, axis=0)
     #   write_np(OUT, "plate.nii")
        return OUT

