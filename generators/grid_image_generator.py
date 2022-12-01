
import SimpleITK as sitk
from generators.image_generator import ImageGenerator


class GridImageGenerator(ImageGenerator):
    """
    Generator a grid image that can be used for visualizing a transformation.
    """
    def __init__(self,
                 dim,
                 sigma,
                 grid_spacing,
                 grid_offset,
                 *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param sigma: sigma parameter for sitk.GridSource()
        :param grid_spacing: grid_spacing parameter for sitk.GridSource()
        :param grid_offset: grid_offset parameter for sitk.GridSource()
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(ImageGenerator, self).__init__(dim=dim,
                                             *args, **kwargs)
        self.sigma = sigma
        self.grid_spacing = grid_spacing
        self.grid_offset = grid_offset

    def get(self, image, transformation, **kwargs):
        """
        Uses the sitk image and transformation to generate a resampled np array.
        :param image: The sitk image.
        :param transformation: The sitk transformation.
        :param kwargs: Not used.
        :return: The resampled np array.
        """
        #
        grid_image = sitk.GridSource(outputPixelType=sitk.sitkUInt16,
                                     size=image.GetSize(),
                                     spacing=image.GetSpacing(),
                                     sigma=self.sigma,
                                     gridSpacing=self.grid_spacing,
                                     gridOffset=self.grid_offset)
        return super(GridImageGenerator, self).get(grid_image, transformation, **kwargs)
