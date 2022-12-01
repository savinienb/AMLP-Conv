import os

import numpy as np
import SimpleITK as sitk
import math
from datasets.graph_dataset import GraphDataset

from datasources.cached_image_datasource import CachedImageDataSource
from datasources.image_datasource import ImageDataSource
from datasources.landmark_datasource import LandmarkDataSource
from generators.image_generator import ImageGenerator
from graph.node import LambdaNode
from iterators.id_list_iterator import IdListIterator
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.intensity.sitk.shift_scale_clamp import ShiftScaleClamp as ShiftScaleClampSitk
from transformations.spatial import translation, scale, composite, rotation, landmark, deformation
from utils.np_image import split_label_image
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk
from transformations.intensity.np.smooth import gaussian
from transformations.intensity.np.normalize import normalize_robust
from utils.sitk_image import apply_np_image_function
from skimage.transform import rescale
from transformations.spatial.displacement_field import DisplacementField
from graph.run_graph import RunGraph

def get_random_transformation(dim, image_size, image_spacing, sampling_factor):
    '''
    #Note: used for test-time update!
    :param dim:
    :param image_size:
    :param image_spacing:
    :return:
    '''
    transformation_list = []
    transformation_list.extend([
                                # translation.OutputCenterToOrigin(dim, image_size, image_spacing),
                                # translation.Random(dim, [20, 20, 20]),
                                # rotation.Random(dim, [0.35, 0.35, 0.35]),
                                # scale.RandomUniform(dim, 0.2),
                                # scale.Random(dim, [0.1, 0.1, 0.1]),
                                # translation.OriginToOutputCenter(dim, image_size, image_spacing),
                                # deformation.Output(dim, [8, 8, 8], 15, image_size, image_spacing),
                                deformation.Output(dim, [8, 8, 8], 8, image_size, image_spacing),
    ])
    comp = composite.Composite(dim, transformation_list, name='image')
    # node = LambdaNode(lambda comp: sitk.DisplacementFieldTransform(
    #     sitk.TransformToDisplacementField(comp, sitk.sitkVectorFloat64, size=image_size,
    #                                       outputSpacing=image_spacing)), name='image', kwparents={'comp': comp})
    # transformation_node = LambdaNode(lambda x: transformation_to_displacement_field_np(dim, image_spacing, x, sampling_factor), name='transformation', parents=[node])
    # return transformation_node

    node = DisplacementField(output_size=image_size, output_spacing=image_spacing, sampling_factor=sampling_factor, keep_transformation_size=True, name='image', parents=[comp])
    transformation_node = LambdaNode(lambda x: transformation_to_displacement_field_np(dim, image_spacing, x, sampling_factor), name='transformation', parents=[node])

    # TODO call from tf.py_function
    return RunGraph([transformation_node])
    # return transformation_node
    # return RunGraph([transformation_node]).run()[0]


def transformation_to_displacement_field_np(dim, image_spacing, transformation, sampling_factor, image_pixel_type=np.float32):
    """
    The sitk transformation is converted into a displacement field and returned as a numpy image.
    :param transformation: sitk transformation.
    :return: The displacement field as numpy image.
    """
    x = transformation.GetDisplacementField()
    x = sitk.GetArrayViewFromImage(x)
    x = x.astype(image_pixel_type)
    x = rescale(x, [sampling_factor] * dim, order=1, preserve_range=True, multichannel=True, mode='symmetric', anti_aliasing=False, clip=False)
    for i in range(dim):
        x[:, :, :, i] /= image_spacing[i]
    x = np.flip(x, axis=-1)
    return x


class Dataset(object):
    """
    The dataset that processes files from the MMWHS challenge.
    """
    def __init__(self,
                 image_size,
                 image_spacing,
                 base_folder,
                 cv,
                 modality,
                 setup_folder_to_use=None,  # 'setup'
                 use_only_elastic_deformation_for_transformation=False,
                 cached_datasource=True,
                 cache_maxsize=8192,
                 input_gaussian_sigma=1.0,
                 label_gaussian_sigma=0.0,
                 use_landmarks=True,
                 num_labels=8,
                 generate_consistency_image_and_transformation=False,
                 displacement_field_sampling_factor=2,
                 image_folder=None,
                 setup_folder=None,
                 image_filename_postfix='_image',
                 image_filename_extension='.mha',
                 labels_filename_postfix='_label',
                 labels_filename_extension='.mha',
                 landmarks_file=None,
                 data_format='channels_first',
                 image_pixel_type=np.float32,
                 label_pixel_type=np.uint8,
                 save_debug_images=False):
        """
        Initializer.
        :param image_size: Network input image size.
        :param image_spacing: Network input image spacing.
        :param base_folder: Dataset base folder.
        :param cv: Cross validation index (1, 2, 3). Or 0 if full training/testing.
        :param modality: Either 'ct' or 'mr'.
        :param input_gaussian_sigma: Sigma value for input smoothing.
        :param label_gaussian_sigma: Sigma value for label smoothing.
        :param use_landmarks: If True, center on loaded landmarks, otherwise use image center.
        :param num_labels: The number of output labels.
        :param image_folder: If set, use this folder for loading the images, otherwise use MMWHS default.
        :param setup_folder: If set, use this folder for loading the setup files, otherwise use MMWHS default.
        :param image_filename_postfix: The image filename postfix.
        :param image_filename_extension: The image filename extension.
        :param labels_filename_postfix: The labels filename postfix.
        :param labels_filename_extension: The labels filename extension.
        :param landmarks_file: If set, use this file for loading image landmarks, otherwise us MMWHS default.
        :param data_format: Either 'channels_first' or 'channels_last'. TODO: adapt code for 'channels_last' to work.
        :param save_debug_images: If true, the generated images are saved to the disk.
        """
        self.image_size = image_size
        self.image_spacing = image_spacing
        self.base_folder = base_folder
        self.cv = cv
        self.modality = modality
        self.use_only_elastic_deformation_for_transformation = use_only_elastic_deformation_for_transformation
        self.cached_datasource = cached_datasource
        self.input_gaussian_sigma = input_gaussian_sigma
        self.label_gaussian_sigma = label_gaussian_sigma
        self.use_landmarks = use_landmarks
        self.num_labels = num_labels
        self.generate_consistency_image_and_transformation = generate_consistency_image_and_transformation
        self.displacement_field_sampling_factor = displacement_field_sampling_factor
        self.image_filename_postfix = image_filename_postfix
        self.image_filename_extension = image_filename_extension
        self.labels_filename_postfix = labels_filename_postfix
        self.labels_filename_extension = labels_filename_extension
        self.data_format = data_format
        self.image_pixel_type = image_pixel_type
        self.label_pixel_type = label_pixel_type
        self.save_debug_images = save_debug_images
        self.dim = 3
        self.image_base_folder = image_folder or os.path.join(self.base_folder, modality + '_mha_original')
        self.setup_base_folder = setup_folder or os.path.join(self.base_folder, setup_folder_to_use)
        self.landmarks_file = landmarks_file or os.path.join(self.setup_base_folder, modality + '_com.csv')
        self.cache_maxsize = cache_maxsize

        if modality == 'ct':
            self.postprocessing_random = self.intensity_postprocessing_ct_random
            self.postprocessing = self.intensity_postprocessing_ct
        else:  # if modality == 'mr':
            self.postprocessing_random = self.intensity_postprocessing_mr_random
            self.postprocessing = self.intensity_postprocessing_mr
        if cv is None:
            self.inference_file = os.path.join(self.setup_base_folder, modality + '_cv', 'inference.txt')
        elif cv > 0:
            self.cv_folder = os.path.join(self.setup_base_folder, modality + '_cv', str(cv))
            self.train_file = os.path.join(self.cv_folder, 'train.txt')
            self.unsupervised_file = os.path.join(self.cv_folder, 'unsupervised.txt')
            self.test_file = os.path.join(self.cv_folder, 'test.txt')
        else:
            self.train_file = os.path.join(self.setup_base_folder, modality + '_train_all.txt')
            self.test_file = os.path.join(self.setup_base_folder, modality + '_test_all.txt')

    def datasources(self, iterator, cached, with_label=True):
        """
        Returns the data sources that load data.
        {
        'image:' (Cached)ImageDataSource that loads the image files.
        'landmarks:' LandmarkDataSource that loads the landmark coordinates.
        'mask:' (Cached)ImageDataSource that loads the groundtruth labels.
        }
        :param iterator: The iterator.
        :param cached: If True, use CachedImageDataSource.
        :return: A dict of data sources.
        """
        old_caching = True  # NOTE: True: per process image caching, False: multiprocessing with shared memory
        datasource_dict = {}
        preprocessing = lambda image: gaussian_sitk(image, self.input_gaussian_sigma)
        if old_caching:
            image_datasource_params = dict(root_location=self.image_base_folder,
                                           file_prefix='',
                                           file_suffix=self.image_filename_postfix,
                                           file_ext=self.image_filename_extension,
                                           set_zero_origin=False,
                                           set_identity_spacing=False,
                                           set_identity_direction=False,
                                           sitk_pixel_type=sitk.sitkInt16,
                                           preprocessing=preprocessing,
                                           name='image',
                                           parents=[iterator])
            if cached:
                image_datasource = CachedImageDataSource(cache_maxsize=self.cache_maxsize,
                                                         **image_datasource_params)
            else:
                image_datasource = ImageDataSource(**image_datasource_params)
        else:
            image_data_source = ImageDataSource
            image_datasource = image_data_source(self.image_base_folder, '', self.image_filename_postfix, self.image_filename_extension, set_zero_origin=False, set_identity_spacing=False, set_identity_direction=False, sitk_pixel_type=sitk.sitkInt16, preprocessing=preprocessing, use_caching=True, name='image', parents=[iterator])
        datasource_dict['image'] = image_datasource
        if self.use_landmarks:
            landmark_datasource = LandmarkDataSource(self.landmarks_file, 1, self.dim, name='landmarks', parents=[iterator])
            datasource_dict['landmarks'] = landmark_datasource
        if with_label:
            if old_caching:
                mask_datasource_params = dict(root_location=self.image_base_folder,
                                              file_prefix='',
                                              file_suffix=self.labels_filename_postfix,
                                              file_ext=self.labels_filename_extension,
                                              set_zero_origin=False,
                                              set_identity_spacing=False,
                                              set_identity_direction=False,
                                              sitk_pixel_type=sitk.sitkUInt8,
                                              name='labels',
                                              parents=[iterator])
                if cached:
                    mask_datasource = CachedImageDataSource(cache_maxsize=self.cache_maxsize,
                                                            **mask_datasource_params)
                else:
                    mask_datasource = ImageDataSource(**mask_datasource_params)

            else:
                mask_datasource = image_data_source(self.image_base_folder, '', self.labels_filename_postfix, self.labels_filename_extension, set_zero_origin=False, set_identity_spacing=False, set_identity_direction=False, sitk_pixel_type=sitk.sitkUInt8, use_caching=True, name='labels', parents=[iterator])
            datasource_dict['labels'] = mask_datasource
        return datasource_dict

    def data_generators(self, datasources, transformation, image_post_processing, mask_post_processing, with_label=True, key_suffix=''):
        """
        Returns the data generators that process one input. See datasources() for dict values.
        :param datasources: The datasources dictionary (see self.datasources()).
        :param transformation: The spatial transformation.
        :param image_post_processing: The np postprocessing function for the image data generator.
        :param mask_post_processing: The np postprocessing function fo the mask data generator
        :return: A dict of data generators.
        """
        generator_dict = {}
        image_generator = ImageGenerator(self.dim, self.image_size, self.image_spacing, interpolator='linear', post_processing_np=image_post_processing, np_pixel_type=self.image_pixel_type, data_format=self.data_format, name=f'image{key_suffix}', parents=[datasources['image'], transformation])
        generator_dict[f'data{key_suffix}'] = image_generator
        if with_label:
            mask_image_generator = ImageGenerator(self.dim, self.image_size, self.image_spacing, interpolator='nearest', post_processing_np=mask_post_processing, np_pixel_type=self.label_pixel_type, data_format=self.data_format, name=f'labels{key_suffix}', parents=[datasources['labels'], transformation])
            generator_dict[f'labels{key_suffix}'] = mask_image_generator
        return generator_dict

    # def split_labels(self, image):
    #     """
    #     Splits a groundtruth label image into a stack of one-hot encoded images.
    #     :param image: The groundtruth label image.
    #     :return: The one-hot encoded image.
    #     """
    #     split = split_label_image(np.squeeze(image, 0), list(range(self.num_labels)), np.uint8)
    #     split_smoothed = [gaussian(i, self.label_gaussian_sigma) for i in split]
    #     smoothed = np.stack(split_smoothed, 0)
    #     image_smoothed = np.argmax(smoothed, axis=0)
    #     split = split_label_image(image_smoothed, list(range(self.num_labels)), np.uint8)
    #     return np.stack(split, 0)

    def smooth_labels(self, image):
        """
        Apply label smooth to a groundtruth label image.
        :param image: The groundtruth label image.
        :return: The smoothed label image.
        """
        if self.label_gaussian_sigma == 0.0:
            return image
        else:
            split = split_label_image(np.squeeze(image, 0), list(range(self.num_labels)), np.uint8)
            split_smoothed = [gaussian(i, self.label_gaussian_sigma) for i in split]
            smoothed = np.stack(split_smoothed, 0)
            image_smoothed = np.expand_dims(np.argmax(smoothed, axis=0).astype(np.uint8), axis=0)
            return image_smoothed
        # Note: do not split output to reduce image size that needs to be copied to GPU
        # split = split_label_image(image_smoothed, list(range(self.num_labels)), np.uint8)
        # return np.stack(split, 0)

    def intensity_postprocessing_ct_random(self, image):
        """
        Intensity postprocessing for CT input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               random_shift=0.2,
                               random_scale=0.2,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)  # .astype(self.image_pixel_type)  # removed for datasource augmentation

    def intensity_postprocessing_ct(self, image):
        """
        Intensity postprocessing for CT input.
        :param image: The np input image.
        :return: The processed image.
        """
        return ShiftScaleClamp(shift=0,
                               scale=1 / 2048,
                               clamp_min=-1.0,
                               clamp_max=1.0)(image)  # .astype(self.image_pixel_type)  # removed for datasource augmentation

    def intensity_postprocessing_mr_random(self, image):
        """
        Intensity postprocessing for MR input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image)
        return ShiftScaleClamp(random_shift=0.2,
                               random_scale=0.4,
                               clamp_min=-1.0)(image)  # .astype(self.image_pixel_type)  # removed for datasource augmentation

    def intensity_postprocessing_mr(self, image):
        """
        Intensity postprocessing for MR input.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image)
        return ShiftScaleClamp(clamp_min=-1.0)(image)  # .astype(self.image_pixel_type)  # removed for datasource augmentation


    def spatial_transformation_augmented_consistency(self, datasources, transformation_1_as_identity=False):
        """
        The spatial image transformation with random augmentation. Three transformations are generated:
        (1) Transformation from datasource to generator image 1.
        (2) Transformation from generator image 1 to another generator image 2.
        (3) Both transformation (1) and (2) consecutively, i.e., from datasource to generator image 2.
        :param datasources: datasources dict.
        :return: The three transformations.
        """
        transformation_1_list = []
        kwparents = {'image': datasources['image']}
        if self.use_landmarks:
            transformation_1_list.append(landmark.Center(self.dim, True))
            kwparents['landmarks'] = datasources['landmarks']
        else:
            transformation_1_list.append(translation.InputCenterToOrigin(self.dim))
        if transformation_1_as_identity:
            transformation_1_list.extend([
                                          translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                          ])
        else:
            transformation_1_list.extend([translation.Random(self.dim, [20, 20, 20]),
                                          rotation.Random(self.dim, [0.35, 0.35, 0.35]),
                                          # rotation.Random(self.dim, [math.pi / 4] * 3),
                                          scale.RandomUniform(self.dim, 0.2),
                                          scale.Random(self.dim, [0.1, 0.1, 0.1]),
                                          translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                          deformation.Output(self.dim, [8, 8, 8], 15, self.image_size, self.image_spacing),
                                          ])
        comp_1 = composite.Composite(self.dim, transformation_1_list, name='image', kwparents=kwparents)

        transformation_2_list = []
        transformation_2_list.extend([
                                      translation.OutputCenterToOrigin(self.dim, self.image_size, self.image_spacing),
                                      translation.Random(self.dim, [20, 20, 20]),
                                      rotation.Random(self.dim, [0.35, 0.35, 0.35]),
                                      # rotation.Random(self.dim, [math.pi / 4] * 3),
                                      scale.RandomUniform(self.dim, 0.2),
                                      scale.Random(self.dim, [0.1, 0.1, 0.1]),
                                      translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                      deformation.Output(self.dim, [8, 8, 8], 15, self.image_size, self.image_spacing),
                                      ])
        comp_2 = composite.Composite(self.dim, transformation_2_list, name='image')

        transformation_1_and_2_list = []
        transformation_1_and_2_list.extend(transformation_1_list)
        transformation_1_and_2_list.extend(transformation_2_list)
        comp_1_and_2 = composite.Composite(self.dim, transformation_1_and_2_list, name='image', kwparents=kwparents)
        #node_1_and_2 = LambdaNode(lambda comp_1_and_2: comp_1_and_2, name='image_comp_1_and_2', kwparents={'comp_1_and_2': comp_1_and_2})
        node_1 = DisplacementField(output_size=self.image_size, output_spacing=self.image_spacing, sampling_factor=self.displacement_field_sampling_factor, name='image', parents=[comp_1])
        node_2 = DisplacementField(output_size=self.image_size, output_spacing=self.image_spacing, sampling_factor=self.displacement_field_sampling_factor, keep_transformation_size=True, name='image', parents=[comp_2])
        node_1_and_2 = DisplacementField(output_size=self.image_size, output_spacing=self.image_spacing, sampling_factor=self.displacement_field_sampling_factor, name='image', parents=[comp_1_and_2])

        return node_1, node_2, node_1_and_2

    def spatial_transformation_augmented(self, datasources):
        """
        The spatial image transformation with random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image']}
        if self.use_landmarks:
            transformation_list.append(landmark.Center(self.dim, True))
            kwparents['landmarks'] = datasources['landmarks']
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.extend([translation.Random(self.dim, [20, 20, 20]),
                                    rotation.Random(self.dim, [0.35, 0.35, 0.35]),
                                    # rotation.Random(self.dim, [math.pi/4] * 3),
                                    scale.RandomUniform(self.dim, 0.2),
                                    scale.Random(self.dim, [0.1, 0.1, 0.1]),
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                    deformation.Output(self.dim, [8, 8, 8], 15, self.image_size, self.image_spacing),
                                    ])
        comp = composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)
        return DisplacementField(output_size=self.image_size, output_spacing=self.image_spacing, sampling_factor=self.displacement_field_sampling_factor, name='image', parents=[comp])

    def spatial_transformation(self, datasources):
        """
        The spatial image transformation without random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        kwparents = {'image': datasources['image']}
        if self.use_landmarks:
            transformation_list.append(landmark.Center(self.dim, True))
            kwparents['landmarks'] = datasources['landmarks']
        else:
            transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.append(translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing))
        return composite.Composite(self.dim, transformation_list, name='image', kwparents=kwparents)

    def dataset_train(self):
        """
        Returns the training dataset. Random augmentation is performed.
        :return: The training dataset.
        """
        iterator = IdListIterator(self.train_file, random=True, use_shuffle=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, cached=self.cached_datasource)
        generators = []
        transformations = []
        if self.generate_consistency_image_and_transformation:
            reference_transformation_1, reference_transformation_2, reference_transformation_1_and_2 = self.spatial_transformation_augmented_consistency(sources)
            if self.use_only_elastic_deformation_for_transformation:
                sources['image'] = LambdaNode(lambda x: apply_np_image_function(x, self.postprocessing_random), name='image', parents=[sources['image']])
                generators_1 = self.data_generators(sources, reference_transformation_1, None, self.smooth_labels, key_suffix='')
                generators_2 = self.data_generators(sources, reference_transformation_1_and_2, None, self.smooth_labels, key_suffix='_2')
            else:
                generators_1 = self.data_generators(sources, reference_transformation_1, self.postprocessing_random, self.smooth_labels, key_suffix='')
                generators_2 = self.data_generators(sources, reference_transformation_1_and_2, self.postprocessing_random, self.smooth_labels, key_suffix='_2')
            transformation_node = LambdaNode(lambda x: transformation_to_displacement_field_np(self.dim, self.image_spacing, x, self.displacement_field_sampling_factor, self.image_pixel_type), name='transformation', parents=[reference_transformation_2])
            generators.extend(list(generators_1.values()))
            generators.extend(list(generators_2.values()))
            generators.append(transformation_node)
            transformations.extend([reference_transformation_1, reference_transformation_2, reference_transformation_1_and_2])
        else:
            reference_transformation = self.spatial_transformation_augmented(sources)
            generators_1 = self.data_generators(sources, reference_transformation, self.postprocessing_random, self.smooth_labels, key_suffix='')
            generators.extend(list(generators_1.values()))
        return GraphDataset(data_generators=generators,
                            data_sources=list(sources.values()),
                            transformations=transformations,
                            iterator=iterator,
                            debug_image_folder='debug_train' if self.save_debug_images else None)

        # reference_transformation = self.spatial_transformation_augmented(sources)
        # generators = self.data_generators(sources, reference_transformation, self.postprocessing_random, self.split_labels)
        # return GraphDataset(data_generators=list(generators.values()),
        #                     data_sources=list(sources.values()),
        #                     transformations=[reference_transformation],
        #                     iterator=iterator,
        #                     debug_image_folder='debug_train' if self.save_debug_images else None)

    def dataset_unsupervised(self):
        """
        Returns the unsupervised dataset. Random augmentation is performed.
        :return: The unsupervised dataset.
        """
        iterator = IdListIterator(self.unsupervised_file, random=True, use_shuffle=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, cached=self.cached_datasource, with_label=False)
        generators = []
        transformations = []
        if self.generate_consistency_image_and_transformation:
            reference_transformation_1, reference_transformation_2, reference_transformation_1_and_2 = self.spatial_transformation_augmented_consistency(sources)
            if self.use_only_elastic_deformation_for_transformation:
                sources['image'] = LambdaNode(lambda x: apply_np_image_function(x, self.postprocessing_random), name='image', parents=[sources['image']])
                generators_1 = self.data_generators(sources, reference_transformation_1, None, self.smooth_labels, with_label=False, key_suffix='')
                generators_2 = self.data_generators(sources, reference_transformation_1_and_2, None, self.smooth_labels, with_label=False, key_suffix='_2')
            else:
                generators_1 = self.data_generators(sources, reference_transformation_1, self.postprocessing_random, self.smooth_labels, with_label=False, key_suffix='')
                generators_2 = self.data_generators(sources, reference_transformation_1_and_2, self.postprocessing_random, self.smooth_labels, with_label=False, key_suffix='_2')
            transformation_node = LambdaNode(lambda x: transformation_to_displacement_field_np(self.dim, self.image_spacing, x, self.displacement_field_sampling_factor, self.image_pixel_type), name='transformation', parents=[reference_transformation_2])
            generators.extend(list(generators_1.values()))
            generators.extend(list(generators_2.values()))
            generators.append(transformation_node)
            transformations.extend([reference_transformation_1, reference_transformation_2, reference_transformation_1_and_2])
        else:
            reference_transformation = self.spatial_transformation_augmented(sources)
            generators_1 = self.data_generators(sources, reference_transformation, self.postprocessing_random, self.smooth_labels, with_label=False, key_suffix='')
            generators.extend(list(generators_1.values()))
        return GraphDataset(data_generators=generators,
                            data_sources=list(sources.values()),
                            transformations=transformations,
                            iterator=iterator,
                            debug_image_folder='debug_unsupervised' if self.save_debug_images else None)

    def dataset_val(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        iterator = IdListIterator(self.test_file, random=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, cached=False)
        reference_transformation = self.spatial_transformation(sources)

        if self.use_only_elastic_deformation_for_transformation:
            sources['image'] = LambdaNode(lambda x: apply_np_image_function(x, self.postprocessing), name='image', parents=[sources['image']])
            generators = self.data_generators(sources, reference_transformation, None, self.smooth_labels)
        else:
            generators = self.data_generators(sources, reference_transformation, self.postprocessing, self.smooth_labels)

        if self.cv == 0:
            del sources['labels']
            del generators['labels']

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_val' if self.save_debug_images else None)

    def dataset_train_without_augmentation(self):
        """
        Returns the validation dataset. No random augmentation is performed.
        :return: The validation dataset.
        """
        # iterator = IdListIterator(self.train_file, random=False, keys=['image_id'], name='iterator')
        # sources = self.datasources(iterator, cached=False)
        # reference_transformation = self.spatial_transformation(sources)
        #
        # if self.use_only_elastic_deformation_for_transformation:
        #     sources['image'] = LambdaNode(lambda x: apply_np_image_function(x, self.postprocessing), name='image', parents=[sources['image']])
        #     generators = self.data_generators(sources, reference_transformation, None, self.split_labels)
        # else:
        #     generators = self.data_generators(sources, reference_transformation, self.postprocessing, self.split_labels)
        #
        #
        # return GraphDataset(data_generators=list(generators.values()),
        #                     data_sources=list(sources.values()),
        #                     transformations=[reference_transformation],
        #                     iterator=iterator,
        #                     debug_image_folder='debug_train_without_augmentation' if self.save_debug_images else None)


        iterator = IdListIterator(self.train_file, random=True, use_shuffle=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, cached=self.cached_datasource)
        generators = []
        transformations = []
        if self.generate_consistency_image_and_transformation:
            reference_transformation_1, reference_transformation_2, reference_transformation_1_and_2 = self.spatial_transformation_augmented_consistency(sources, transformation_1_as_identity=True)
            if self.use_only_elastic_deformation_for_transformation:
                sources['image'] = LambdaNode(lambda x: apply_np_image_function(x, self.postprocessing), name='image', parents=[sources['image']])
                generators_1 = self.data_generators(sources, reference_transformation_1, None, self.smooth_labels, key_suffix='')
                generators_2 = self.data_generators(sources, reference_transformation_1_and_2, None, self.smooth_labels, key_suffix='_2')
            else:
                generators_1 = self.data_generators(sources, reference_transformation_1, self.postprocessing, self.smooth_labels, key_suffix='')
                generators_2 = self.data_generators(sources, reference_transformation_1_and_2, self.postprocessing, self.smooth_labels, key_suffix='_2')
            transformation_node = LambdaNode(lambda x: transformation_to_displacement_field_np(self.dim, self.image_spacing, x, self.displacement_field_sampling_factor, self.image_pixel_type), name='transformation', parents=[reference_transformation_2])
            generators.extend(list(generators_1.values()))
            generators.extend(list(generators_2.values()))
            generators.append(transformation_node)
            transformations.extend([reference_transformation_1, reference_transformation_2, reference_transformation_1_and_2])
        else:
            reference_transformation = self.spatial_transformation(sources)
            generators_1 = self.data_generators(sources, reference_transformation, self.postprocessing, self.smooth_labels, key_suffix='')
            generators.extend(list(generators_1.values()))
        return GraphDataset(data_generators=generators,
                            data_sources=list(sources.values()),
                            transformations=transformations,
                            iterator=iterator,
                            debug_image_folder='dataset_train_without_augmentation' if self.save_debug_images else None)


    def dataset_inference(self):
        """
        Returns the inference dataset. No random augmentation is performed.
        :return: The inference dataset.
        """
        # iterator = 'iterator'
        iterator = IdListIterator(self.inference_file, random=False, keys=['image_id'], name='iterator')
        sources = self.datasources(iterator, cached=False, with_label=False)
        reference_transformation = self.spatial_transformation(sources)

        if self.use_only_elastic_deformation_for_transformation:
            sources['image'] = LambdaNode(lambda x: apply_np_image_function(x, self.postprocessing), name='image', parents=[sources['image']])
            generators = self.data_generators(sources, reference_transformation, None, None, with_label=False)
        else:
            generators = self.data_generators(sources, reference_transformation, self.postprocessing, None, with_label=False)

        return GraphDataset(data_generators=list(generators.values()),
                            data_sources=list(sources.values()),
                            transformations=[reference_transformation],
                            iterator=iterator,
                            debug_image_folder='debug_inference' if self.save_debug_images else None)




