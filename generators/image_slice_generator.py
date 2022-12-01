
import numpy as np
import utils.sitk_np
from generators.transformation_generator_base import TransformationGeneratorBase
from utils.sitk_image import resample
from transformations.spatial.common import create_composite
from transformations.spatial import rotation, translation
import scipy.ndimage as ndimage
import math
import SimpleITK as sitk


#TODO NOTE: This file is based on "image_slice_and_drr_generator.py". As such, there are still some parts that can be removed from this file for it to be clean. However, the time consuming parts (especially DRR Generation) were removed.


class ImageSliceGenerator(TransformationGeneratorBase):
    def __init__(self,
                 dim,
                 output_size,
                 angles,
                 num_drr_views,
                 mask,
                 output_spacing=None,
                 pre_transformation=None,
                 post_transformation=None,
                 post_processing_sitk=None,
                 post_processing_np=None,
                 interpolator='linear',
                 resample_sitk_pixel_type=None,
                 resample_default_pixel_value=None,
                 return_zeros_if_not_found=False,
                 data_format='channels_first'):
        super(ImageSliceGenerator, self).__init__(dim=dim,
                                             pre_transformation=pre_transformation,
                                             post_transformation=post_transformation)
        self.output_size = output_size
        self.angles = angles
        self.num_drr_views = num_drr_views
        self.np_mask = mask
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * dim
        self.interpolator = interpolator
        assert data_format == 'channels_first' or data_format == 'channels_last', 'unsupported data format'
        self.data_format = data_format
        self.post_processing_sitk = post_processing_sitk
        self.post_processing_np = post_processing_np
        self.resample_sitk_pixel_type = resample_sitk_pixel_type
        self.resample_default_pixel_value = resample_default_pixel_value
        self.return_zeros_if_not_found = return_zeros_if_not_found
        #self.np_mask = generate_mask(self.output_size)


    def get_resampled_image_slice(self, image, transformation):
        transformation_list = [transformation]

        composite_transformation = create_composite(self.dim, transformation_list)
        output_image = resample(image,
                                composite_transformation,
                                self.output_size,
                                self.output_spacing,
                                interpolator=self.interpolator,
                                output_pixel_type=self.resample_sitk_pixel_type,
                                default_pixel_value=self.resample_default_pixel_value)

        return output_image


    def get_np_image_list(self, output_image_sitk):
        output_image_list_np = []
        output_image_np = utils.sitk_np.sitk_to_np(output_image_sitk, np.float32)
        pixel_components = output_image_sitk.GetNumberOfComponentsPerPixel()
        if pixel_components > 1:
            for i in range(pixel_components):
                output_image_list_np.append(output_image_np[..., i])
        else:
            output_image_list_np.append(output_image_np)
        return output_image_list_np


    def get_np_image(self, output_image_sitk):
        output_image_list_np = []
        if isinstance(output_image_sitk, list):
            for current_output_image_sitk in output_image_sitk:
                output_image_list_np.extend(self.get_np_image_list(current_output_image_sitk))
        else:
            output_image_list_np = self.get_np_image_list(output_image_sitk)

        if self.data_format == 'channels_first':
            output_image_np = np.stack(output_image_list_np, axis=0)
        elif self.data_format == 'channels_last':
            output_image_np = np.stack(output_image_list_np, axis=self.dim)

        return output_image_np


    def get(self, image, transformation, **kwargs):
        if image is None and self.return_zeros_if_not_found:
            if self.data_format == 'channels_first':
                output_image_np = np.zeros([1] + list(reversed(self.output_size)), np.float32)
            else: # if self.data_format == 'channels_last':
                output_image_np = np.zeros(list(reversed(self.output_size)) + [1], np.float32)
            return output_image_np

        else:

            #GENERATE IMAGE SLICE TARGET
            output_image_sitk = self.get_resampled_image_slice(image, transformation)

            if self.post_processing_sitk is not None:
                output_image_sitk = self.post_processing_sitk(output_image_sitk)

            # convert to np array
            output_image_np = self.get_np_image(output_image_sitk)

            #POST-PROCESSING
            if self.post_processing_np is not None:
                output_image_np = self.post_processing_np(output_image_np)

            output_image_slice_np = output_image_np

            #MASK IMAGE SLICE TARGET WITH CIRCLE FOR LOSS FUNCTION
            if self.np_mask is not None:
                output_image_slice_np = np.multiply(output_image_slice_np, self.np_mask)

            return output_image_slice_np










