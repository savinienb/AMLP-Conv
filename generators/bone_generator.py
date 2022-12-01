
import numpy as np
from utils.sitk_image import resample
from utils.geometry import point_is_left
import utils.sitk_np
from transformations.spatial import translation, flip, center_line_at_y_axis, composite
from generators.transformation_generator_base import TransformationGeneratorBase

class BoneGenerator(TransformationGeneratorBase):
    def __init__(self,
                 dim,
                 output_size,
                 bone_indizes=None,
                 pre_transformation=None,
                 post_transformation=None,
                 post_processing_sitk=None,
                 post_processing_np=None,
                 interpolation='linear',
                 resample_sitk_pixel_type=None,
                 data_format='channels_first'):
        super(BoneGenerator, self).__init__(dim=dim,
                                            pre_transformation=pre_transformation,
                                            post_transformation=post_transformation)
        self.output_size = output_size
        self.bone_indizes = bone_indizes
        self.post_processing_sitk = post_processing_sitk
        self.post_processing_np = post_processing_np
        self.interpolation = interpolation
        self.resample_sitk_pixel_type = resample_sitk_pixel_type
        assert data_format == 'channels_first' or data_format == 'channels_last', 'unsupported data format'
        self.data_format = data_format

    def get_bone_line(self, landmarks, bone_index):
        raise NotImplementedError

    def do_flipping(self, landmarks):
        raise NotImplementedError

    def get_np_image(self, output_image_sitk):
        if isinstance(output_image_sitk, list):
            output_image_list_np = [utils.sitk_np.sitk_to_np(current_output_image_sitk, np.float32)
                                    for current_output_image_sitk in output_image_sitk]
            if self.data_format == 'channels_first':
                output_image_np = np.stack(output_image_list_np, axis=0)
            elif self.data_format == 'channels_last':
                output_image_np = np.stack(output_image_list_np, axis=self.dim)
        else:
            output_image_np = utils.sitk_np.sitk_to_np(output_image_sitk, np.float32)
            if self.data_format == 'channels_first':
                output_image_np = np.expand_dims(output_image_np, axis=0)
            elif self.data_format == 'channels_last':
                output_image_np = np.expand_dims(output_image_np, axis=self.dim)
        return output_image_np

    def get_bone_crop_spacing_and_transformation(self, landmarks, image, bone_index):
        # update transformation with current parameters
        bone_line = self.get_bone_line(landmarks, bone_index)
        input_spacing = np.array(image.GetSpacing())

        # create center line transformations
        line_length = np.linalg.norm((bone_line[1] - bone_line[0]) * input_spacing)
        output_spacing = [line_length / self.output_size[1]] * self.dim
        transformations = [center_line_at_y_axis.CenterLineAtYAxis(self.dim, self.output_size, None)]

        if self.do_flipping(landmarks):
            transformations += [translation.OutputCenterToOrigin(self.dim, self.output_size),
                                flip.Fixed(self.dim, [True] + [False] * (self.dim - 1)),
                                translation.OriginToOutputCenter(self.dim, self.output_size)]

        # add additional transformations if set
        transformation = composite.Composite(self.dim, transformations)
        return output_spacing, transformation.get(image=image, line=bone_line, output_spacing=output_spacing)

    def get_bone_image(self, landmarks, image, bone_index):
        spacing, bone_crop_transformation = self.get_bone_crop_spacing_and_transformation(landmarks, image, bone_index)
        current_transformation = self.get_transformation(bone_crop_transformation, input_size=self.output_size, input_spacing=spacing)
        return resample(image, current_transformation, self.output_size, spacing, interpolator=self.interpolation, output_pixel_type=self.resample_sitk_pixel_type)

    def get_bone_images(self, landmarks, image):
        return [self.get_bone_image(landmarks, image, bone_index) for bone_index in self.bone_indizes]

    def get(self, landmarks, image):
        output_images_sitk = self.get_bone_images(landmarks, image)

        if self.post_processing_sitk is not None:
            output_images_sitk = self.post_processing_sitk(output_images_sitk)

        output_image_np = self.get_np_image(output_images_sitk)

        if self.post_processing_np is not None:
            output_image_np = self.post_processing_np(output_image_np)

        return output_image_np


class HandLBIGenerator(BoneGenerator):
    def do_flipping(self, landmarks):
        return False

    def get_bone_line(self, landmarks, bone_index):
        bone_matrix = [[25, 26, 27],
                       [24, 26, 27],
                       [0, 1],
                       [1, 2],
                       [2, 19],
                       [8, 4],
                       [8, 12],
                       [12, 16],
                       [16, 21],
                       [10, 6],
                       [10, 14],
                       [14, 18],
                       [18, 23]]

        bone_ratio = 1.0
        additional_translate = np.zeros(3, np.float32)
        if bone_index == 0:
            bone_ratio = 1.0/0.7
        elif bone_index == 1:
            bone_ratio = 1.0/0.6
            additional_translate[0] = -8
        elif bone_index in [4, 8, 12]:
            bone_ratio = 1.0
            additional_translate[0] = 0.15
        elif bone_index in [5, 9]:
            bone_ratio = 1.0/3.0
        elif bone_index in [2, 6, 10]:
            bone_ratio = 1.0/2.0
            additional_translate[0] = 0.1
        elif bone_index in [3, 7, 11]:
            bone_ratio = 1.0/1.5
            additional_translate[0] = 0.15

        if bone_index in [0, 1]:
            # for radius and ulna create a horizontal line between three defined points
            # and take a line that goes from the center of this line down in y direction
            p0 = landmarks[bone_matrix[bone_index][0]].coords.copy()  # first point
            p1 = (landmarks[bone_matrix[bone_index][1]].coords + landmarks[bone_matrix[bone_index][2]].coords) / 2.0  # center between other 2 points

            # set y coordinate to be same -> line between p0 and p1 is horizontal
            p1[1] = p0[1]
            dir_length = np.linalg.norm(p1 - p0)

            # start is in the center of line between p0 and p1
            start = ((p1 + p0)) / 2.0 + additional_translate

            # end is start point shifted down in y direction by defined factor
            end = start.copy()
            end[1] += dir_length * bone_ratio
        else:
            # for finger points take line defined by start and end points of bone
            p0 = landmarks[bone_matrix[bone_index][0]].coords.copy()
            p1 = landmarks[bone_matrix[bone_index][1]].coords.copy()
            dir = p1 - p0
            # start is p0
            start = p0 - dir * additional_translate[0]
            # end is start + direction * factor to allow scaling
            end = start + bone_ratio * dir

        return start, end

class ClavicleLBIGenerator(BoneGenerator):
    def do_flipping(self, landmarks):
        return False

    def get_bone_line(self, landmarks, bone_index):
        bone_top = 0.75
        bone_bottom = 0.75
        if bone_index == 0:
            p0 = landmarks[0].coords.copy()
            p1 = landmarks[1].coords.copy()
            dir = p1 - p0
            dir_length = np.linalg.norm(dir)
            c = (p0 + p1) * 0.5
            normal = np.array([dir[1], -dir[0], 0])
            unit_normal = normal / np.linalg.norm(normal)
        elif bone_index == 1:
            p0 = landmarks[3].coords.copy()
            p1 = landmarks[2].coords.copy()
            dir = p1 - p0
            dir_length = np.linalg.norm(dir)
            c = (p0 + p1) * 0.5
            normal = np.array([-dir[1], dir[0], 0])
            unit_normal = normal / np.linalg.norm(normal)

        start = c + bone_top * dir_length * unit_normal
        end = c - bone_bottom * dir_length * unit_normal

        return start, end

class TeethLBIGenerator(BoneGenerator):
    def do_flipping(self, landmarks):
        return False

    def get_bone_line(self, landmarks, bone_index):
        bone_length = 0.75

        p0 = landmarks[0].coords.copy()
        p1 = landmarks[1].coords.copy()

        if bone_index == 0:
            p0 = landmarks[0].coords.copy()
            p1 = landmarks[2].coords.copy()
        else:
            p0 = landmarks[1].coords.copy()
            p1 = landmarks[3].coords.copy()

        dir = p0 - p1
        length = np.linalg.norm(dir)

        vec_top = np.zeros([3])
        if bone_index == 0:
            vec_top[0] = dir[0]
            vec_top[1] = -dir[1]
        else:
            vec_top[0] = -dir[0]
            vec_top[1] = dir[1]
        vec_top /= np.linalg.norm(vec_top)

        start = p0 + vec_top * bone_length * length
        end = p0 - vec_top * bone_length * length

        return start, end


class ChallengeGenerator(BoneGenerator):
    def is_left_hand(self, landmarks):
        wrist_index = 0
        little_finger_index = 6
        thumb_index = 10
        line = (landmarks[wrist_index].coords, landmarks[thumb_index].coords)
        p = landmarks[little_finger_index].coords
        return point_is_left(line, p)

    def do_flipping(self, landmarks):
        return self.is_left_hand(landmarks)


class HandChallengeGenerator(ChallengeGenerator):
    def get_bone_line(self, landmarks, bone_index):
        start_index = 0
        mean_end_indizes = [7, 8, 9]
        bone_ratio = 1.2
        additional_translation = np.array([0.1, -0.1])
        if self.do_flipping(landmarks):
            additional_translation[0] *= -1

        p0 = landmarks[start_index].coords.copy()
        p1 = np.mean(np.array([landmarks[i].coords for i in mean_end_indizes]), axis=0)

        dir = p1 - p0
        orthogonal = np.array([-dir[1], dir[0]])
        # start is p0
        start = p0 + dir * additional_translation[1] + orthogonal * additional_translation[0]
        # end is start + direction * factor to allow scaling
        end = start + bone_ratio * dir

        return start, end


class WristChallengeGenerator(ChallengeGenerator):
    def get_bone_line(self, landmarks, bone_index):
        start_index = 0
        mean_end_indizes = [1, 2, 3, 4, 5]
        bone_ratio = 0.65
        additional_translation = np.array([0.2, -0.05])
        if self.do_flipping(landmarks):
            additional_translation[0] *= -1

        p0 = landmarks[start_index].coords.copy()
        p1 = np.mean(np.array([landmarks[i].coords for i in mean_end_indizes]), axis=0)

        dir = p1 - p0
        orthogonal = np.array([-dir[1], dir[0]])
        # start is p0
        start = p0 + dir * additional_translation[1] + orthogonal * additional_translation[0]
        # end is start + direction * factor to allow scaling
        end = start + bone_ratio * dir

        return start, end


class FingersChallengeGenerator(ChallengeGenerator):
    def get_bone_line(self, landmarks, bone_index):
        bone_matrix = [[1, 6],
                       [2, 7],
                       [3, 8],
                       [4, 9],
                       [5, 10]]
        bone_ratio = 1.2
        additional_translation = np.array([0.0, -0.1])
        if self.do_flipping(landmarks):
            additional_translation[0] *= -1

        # for finger points take line defined by start and end points of bone
        p0 = landmarks[bone_matrix[bone_index][0]].coords.copy()
        p1 = landmarks[bone_matrix[bone_index][1]].coords.copy()

        dir = p1 - p0
        orthogonal = np.array([-dir[1], dir[0]])
        # start is p0
        start = p0 + dir * additional_translation[1] + orthogonal * additional_translation[0]
        # end is start + direction * factor to allow scaling
        end = start + bone_ratio * dir

        return start, end

class HandXrayGenerator(BoneGenerator):
    def do_flipping(self, landmarks):
        return False

    def get_bone_line(self, landmarks, bone_index):
        mean_end_indizes = [3, 4]
        mean_start_indizes = [24, 28, 32]
        bone_ratio = 1.2
        additional_translation = np.array([0.1, -0.1])

        p0 = np.mean(np.array([landmarks[i].coords for i in mean_start_indizes]), axis=0)
        p1 = np.mean(np.array([landmarks[i].coords for i in mean_end_indizes]), axis=0)

        dir = p1 - p0
        orthogonal = np.array([-dir[1], dir[0]])
        # start is p0
        start = p0 + dir * additional_translation[1]
        # end is start + direction * factor to allow scaling
        end = start + bone_ratio * dir

        return start, end

class FingersXrayGenerator(BoneGenerator):
    def do_flipping(self, landmarks):
        return False

    def get_bone_line(self, landmarks, bone_index):
        bone_matrix = [[4, 3],
                       [0, 2],
                       [17, 18],
                       [18, 19],
                       [19, 20],
                       [25, 15],
                       [25, 26],
                       [26, 27],
                       [27, 28],
                       [33, 13],
                       [33, 34],
                       [34, 35],
                       [35, 36]]

        bone_ratio = 1.0
        additional_translate = np.zeros(2, np.float32)
        if bone_index == 0:
            bone_ratio = 1.6
        elif bone_index == 1:
            bone_ratio = 1.6
        elif bone_index in [4, 8, 12]:
            bone_ratio = 1.0
            additional_translate[0] = 0.15
        elif bone_index in [5, 9]:
            bone_ratio = 1.0 / 3.0
        elif bone_index in [2, 6, 10]:
            bone_ratio = 1.0 / 2.0
            additional_translate[0] = 0.1
        elif bone_index in [3, 7, 11]:
            bone_ratio = 1.0 / 1.5
            additional_translate[0] = 0.15



        if bone_index in [0, 1]:
            # for radius and ulna create a horizontal line between three defined points
            # and take a line that goes from the center of this line down in y direction
            p0 = landmarks[bone_matrix[bone_index][0]].coords.copy()  # first point
            p1 = landmarks[bone_matrix[bone_index][1]].coords.copy()

            # set y coordinate to be same -> line between p0 and p1 is horizontal
            dir_length = np.linalg.norm(p1 - p0)

            # start is in the center of line between p0 and p1
            start = ((p1 + p0)) / 2.0
            start[0] -= additional_translate[0] * dir_length

            # end is start point shifted down in y direction by defined factor
            end = start.copy()
            end[1] += dir_length * bone_ratio / 2
            start[1] -= dir_length * bone_ratio / 2
        else:
            # for finger points take line defined by start and end points of bone
            p0 = landmarks[bone_matrix[bone_index][0]].coords.copy()
            p1 = landmarks[bone_matrix[bone_index][1]].coords.copy()
            dir = p1 - p0
            # start is p0
            start = p0 - dir * additional_translate[0]
            # end is start + direction * factor to allow scaling
            end = start + bone_ratio * dir
        return start, end

class WristXrayGenerator(BoneGenerator):
    def do_flipping(self, landmarks):
        return False

    def get_bone_line(self, landmarks, bone_index):
        mean_end_indizes = [0, 2, 3, 4]
        mean_start_indizes = [14, 15, 16]

        bone_ratio = 0.2

        p0 = np.mean(np.array([landmarks[i].coords for i in mean_start_indizes]), axis=0)
        p1 = np.mean(np.array([landmarks[i].coords for i in mean_end_indizes]), axis=0)

        # set y coordinate to be same -> line between p0 and p1 is horizontal
        dir_length = np.linalg.norm(p1 - p0)
        dir = p1 - p0

        end = p1 + bone_ratio * dir
        start = end - (1 + 2 * bone_ratio) * dir

        return start, end
