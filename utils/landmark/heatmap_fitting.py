
import numpy as np
from utils.np_image import find_maximum_in_image
import scipy.optimize as opt


def gauss2d(data, means, sigmas, rotation, scale):
    """
    Evaluates the Gaussian functions with the given parameters for the data array.
    :param data: The coordinates on which the Gaussian function should be evaluated.
    :param means: The coordinates of the Gaussian mean.
    :param sigmas: The sigma parameters of the Gaussian.
    :param rotation: The rotation parameter of the Gaussian.
    :param scale: The multiplicative scale factor of the Gaussian.
    :return: The heatmap image as a np array.
    """
    dim = len(means)

    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    det_covariances = np.prod(sigmas)
    sigmas_inv_eye = np.diag(1.0 / sigmas)
    inv_covariances = rotation_matrix.T @ sigmas_inv_eye @ rotation_matrix

    scale /= np.sqrt(np.power(2 * np.pi, dim) * det_covariances)

    grid_stacked = np.stack(data, axis=dim)

    x_minus_mu = grid_stacked - means
    exp_factor = np.sum(np.sum(np.expand_dims(x_minus_mu, -1) * inv_covariances * np.expand_dims(x_minus_mu, -2), axis=-1), axis=-1)
    heatmap = scale * np.exp(-0.5 * exp_factor)

    return heatmap


def normalize_rotation(rotation, max_rotation=np.pi):
    """
    Normalize rotation such that it lies between 0 and max_rotation.
    :param rotation: The rotation to normalize
    :param max_rotation: The max_rotation.
    :return: Normalized rotation.
    """
    return rotation % max_rotation


def get_covariance(sigma, rotation):
    """
    Calclate the covariance matrix for the given sigmas and rotation.
    :param sigma: The Gaussian sigmas.
    :param rotation: The rotation.
    :return: The covariance matrix as np array.
    """
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])
    return rotation_matrix.T @ np.array([[sigma[0], 0], [0, sigma[1]]]) @ rotation_matrix


def get_rotations(eigenvectors):
    """
    Return rotation values for the given eigenvectors. Used in get_sigma_rotation.
    :param eigenvectors: The calculated eigenvectors of the covariances.
    :return: Rotation values for each entry of the the eigenvectors.
    """
    rotation_00 = np.arccos(eigenvectors[0, 0])
    rotation_10 = np.arcsin(-eigenvectors[1, 0])
    rotation_01 = np.arcsin(eigenvectors[0, 1])
    rotation_11 = np.arccos(eigenvectors[1, 1])
    # print(rotation_00 * 180 / np.pi, rotation_01 * 180 / np.pi, rotation_10 * 180 / np.pi, rotation_11 * 180 / np.pi)
    if rotation_00 < 0:
        rotation_00 += np.pi
    if rotation_10 < 0:
        rotation_10 += np.pi
    if rotation_01 < 0:
        rotation_01 += np.pi
    if rotation_11 < 0:
        rotation_11 += np.pi
    return rotation_00, rotation_01, rotation_10, rotation_11


def get_sigma_rotation(covariances):
    """
    Calculates the sigmas and rotation value for the given covariances.
    :param covariances: The Gaussian covariances.
    :return: sigmas, rotation
    """
    eigenvalues, eigenvectors = np.linalg.eig(covariances)
    rotations = get_rotations(eigenvectors)
    rotation = np.mean(rotations)
    if not np.allclose(rotation, rotations, rtol=1e-2, atol=1e-2):
        rotations = get_rotations(eigenvectors.T)
        rotation = np.mean(rotations)
        # if not np.allclose(rotation, rotations, rtol=1e-2, atol=1e-2):
        #     print('error')
    sigma = eigenvalues
    calculated_covariance = get_covariance(sigma, rotation)
    if not np.allclose(covariances, calculated_covariance, rtol=1e-2, atol=1e-2):
        rotation = -rotation
    # calculated_covariance = get_covariance(sigma, rotation)
    # if not np.allclose(covariances, calculated_covariance, rtol=1e-2, atol=1e-2):
    #     print(covariances)
    #     print(calculated_covariance)
    #     print('error second')
    if sigma[0] < sigma[1]:
        sigma = np.flip(sigma, axis=-1)
        rotation += np.pi / 2
    sigma = np.sqrt(sigma)
    rotation = normalize_rotation(rotation)
    return sigma, rotation


def get_mean_cov_points(points):
    """
    Calculates the mean and covariances for given points.
    :param points: The point coordinates.
    :return: mean, cov
    """
    dim = points.shape[-1]
    mean = np.mean(points, axis=0)

    # calculate coords minus mean for covariance calculation
    cov = np.zeros([dim, dim], np.float32)
    for i in range(dim):
        for j in range(dim):
            # calculate covariance for all combinations of dimensions
            cov[i, j] = np.mean((points[..., i] - mean[i]) * (points[..., j] - mean[j]))

    return mean, cov


def fit_gaussian_curve(f, initial_guess, bounds, heatmap, region_size=None):
    """
    Fits a Gaussian function to the given heatmap with a robust fitting function (scipy curve_fit).
    The function f is evaluated and fitted to the heatmap. The initially guessed coordinate is calculated as the maximum in the image,
    while the remaining corresponding initial_guess parameters are appended and must be given.
    :param f: The function that will be evaluated.
    :param initial_guess: The initial parameters tuple (except the initial coordinates).
    :param bounds: The bounds tuple of tuple (except the coordinates).
    :param heatmap: The heatmap as np array.
    :param region_size: The region size to crop around the landmark maximum location. If None, the whole image will be used (slower and less robust).
    :return: The optimal parameters. If curve_fit did not succeed, the initial guess is returned.
    """
    if region_size is None:
        region_size = heatmap.shape
    region_size = np.array(region_size)
    image_size = heatmap.shape
    max_value, max_coord = find_maximum_in_image(heatmap)
    region_start = np.maximum(0, max_coord - region_size / 2).astype(int)
    region_end = np.minimum(image_size, max_coord + region_size / 2).astype(int)

    data = np.stack(np.meshgrid(*[np.arange(start, end) for start, end in zip(region_start, region_end)], indexing='ij'), axis=0)

    initial_guess = (max_coord[1], max_coord[0]) + initial_guess
    bounds = (region_start[1], region_start[0]) + bounds[0], (region_end[1], region_end[0]) + bounds[1]
    cropped_heatmap = heatmap[tuple([slice(s, e) for s, e in zip(region_start, region_end)])]

    try:
        popt, pcov = opt.curve_fit(f, data, cropped_heatmap.reshape(-1), p0=initial_guess, bounds=bounds)
    except:
        popt = initial_guess

    return popt


def fit_gaussian_curve_all_parameters(heatmap, sigma, rotation, scale, region_size=None):
    """
    Fits an arbitrary Gaussian function to the given heatmap.
    :param heatmap: The heatmap as np array.
    :param sigma: The initial sigma parameters.
    :param rotation: The initial rotation parameter.
    :param scale: The initial scale parameter.
    :param region_size: The region size that should be cropped around the heatmap maximum.
    :return: mean, sigma, rotation of the fitted Gaussian.
    """
    def f(data, mean_x, mean_y, sigma_x, sigma_y, rotation, scale):
        return gauss2d(data, np.array([mean_y, mean_x]), np.array([sigma_y, sigma_x]), rotation, scale).ravel()
    initial_guess = (sigma[0], sigma[1], rotation, scale)
    bounds = ((0.1, 0.1, -10 * np.pi, 0.0), (20.0, 20.0, 10 * np.pi, 1000.0))
    mean_x, mean_y, sigma_x, sigma_y, rotation, scale = fit_gaussian_curve(f, initial_guess, bounds, heatmap, region_size)
    mean = (mean_x, mean_y)
    # first sigma entry must be larger than second one, if not -> reorder
    if sigma_x > sigma_y:
        sigma = sigma_x, sigma_y
        rotation = normalize_rotation(rotation)
    else:
        sigma = sigma_y, sigma_x
        rotation = normalize_rotation(rotation + np.pi / 2)
    return mean, sigma, rotation


def fit_gaussian_curve_sigma_rotation_scale_fixed(heatmap, sigma, rotation, scale, region_size=None):
    """
    Fits a Gaussian function with fixed sigma, rotation and scale to the given heatmap.
    :param heatmap: The heatmap as np array.
    :param sigma: The initial sigma parameters.
    :param rotation: The initial rotation parameter.
    :param scale: The initial scale parameter.
    :param region_size: The region size that should be cropped around the heatmap maximum.
    :return: mean, sigma, rotation of the fitted Gaussian.
    """
    def f(data, mean_x, mean_y):
        return gauss2d(data, np.array([mean_y, mean_x]), np.array([sigma[1], sigma[0]]), rotation, scale).ravel()
    initial_guess = tuple()
    bounds = (tuple(), tuple())
    mean_x, mean_y = fit_gaussian_curve(f, initial_guess, bounds, heatmap, region_size)
    return (mean_x, mean_y), sigma, rotation


def fit_gaussian_curve_single_sigma(heatmap, sigma, rotation, scale, region_size=None):
    """
    Fits a Gaussian function with a single isotropic sigma value.
    :param heatmap: The heatmap as np array.
    :param sigma: The initial sigma parameters.
    :param rotation: The initial rotation parameter.
    :param scale: The initial scale parameter.
    :param region_size: The region size that should be cropped around the heatmap maximum.
    :return: mean, sigma, rotation of the fitted Gaussian.
    """
    def f(data, mean_x, mean_y, single_sigma):
        return gauss2d(data, np.array([mean_y, mean_x]), np.array([single_sigma, single_sigma]), rotation, scale).ravel()
    initial_guess = (sigma[0],)
    bounds = ((0.1,), (20.0,))
    mean_x, mean_y, single_sigma = fit_gaussian_curve(f, initial_guess, bounds, heatmap, region_size)
    mean = (mean_x, mean_y)
    sigma = (single_sigma, single_sigma)
    return mean, sigma, rotation


def fit_gaussian_mean_sigma_rotation(heatmap, region_size=None, min_value=0.0, max_min_value_factor=0.25, *args, **kwargs):
    """
    Fit a Gaussian function to a heatmap by calculating the mean and covariances directly from the image data.
    Not robust against responses far away from the heatmap maximum.
    :param heatmap: The heatmap as np array.
    :param region_size: The region size that should be cropped around the heatmap maximum.
    :param min_value: Values smaller than min_value are not used for heatmap mean and covariance calculation.
    :param max_min_value_factor: Values smaller than max_heatmap_value * max_min_value_factor are not used for heatmap mean and covariance calculation.
    :param args: Not used.
    :param kwargs: Not used.
    :return: mean, sigma, rotation
    """
    mean, cov = fit_gaussian_mean_cov(heatmap, region_size, min_value, max_min_value_factor)
    mean = np.flip(mean, axis=0)
    sigma, rotation = get_sigma_rotation(cov)
    rotation = normalize_rotation(rotation + np.pi / 2)
    return mean, sigma, rotation


def fit_gaussian_mean_cov(heatmap, region_size=None, min_value=0.0, max_min_value_factor=0.25):
    """
    Fit a Gaussian function to a heatmap by calculating the mean and covariances directly from the image data.
    Not robust against responses far away from the heatmap maximum.
    :param heatmap: The heatmap as np array.
    :param region_size: The region size that should be cropped around the heatmap maximum.
    :param min_value: Values smaller than min_value are not used for heatmap mean and covariance calculation.
    :param max_min_value_factor: Values smaller than max_heatmap_value * max_min_value_factor are not used for heatmap mean and covariance calculation.
    :return: mean, covariances
    """
    # image parameters of heatmap
    if region_size is None:
        region_size = heatmap.shape
    region_size = np.array(region_size)
    dim = heatmap.ndim
    image_size = heatmap.shape
    if region_size is None:
        region_size = image_size

    # search for maximum in image for gaussian fitting
    max_value, max_coord = find_maximum_in_image(heatmap)

    # define region to consider for calculating heatmap parameters
    region_start = (max_coord - region_size / 2).astype(int)
    region_end = (max_coord + region_size / 2).astype(int)
    region_start = np.maximum(0, region_start).astype(int)
    region_end = np.minimum(image_size, region_end).astype(int)

    # crop heatmap according to region
    cropped_heatmap = heatmap[tuple([slice(s, e) for s, e in zip(region_start, region_end)])]

    # set values < max_value * max_min_value_factor to 0.0
    cropped_heatmap[cropped_heatmap < max_value * max_min_value_factor] = 0.0
    # set values < min_value to 0.0
    cropped_heatmap[cropped_heatmap < min_value] = 0.0
    # normalize heatmap to sum up to one
    heatmap_min = np.min(cropped_heatmap)
    heatmap_sum = np.sum(cropped_heatmap)
    cropped_normalized_heatmap = (cropped_heatmap - heatmap_min) / heatmap_sum

    # set coordinate grid inside region
    coords = np.stack(np.meshgrid(*[np.arange(start, end) for start, end in zip(region_start, region_end)], indexing='ij'), axis=0)
    # multiply coordinates with heatmap values to get weighted coordinates
    normalized_heatmap_coords = coords * cropped_normalized_heatmap

    # mean is the sum over all weighted coordinates for each dimension
    heatmap_mean = normalized_heatmap_coords
    for i in range(dim):
        heatmap_mean = np.sum(heatmap_mean, axis=-1)

    # calculate coords minus mean for covariance calculation
    coords_minus_mean = coords - np.reshape(heatmap_mean, [dim] + [1] * dim)
    heatmap_cov = np.zeros([dim, dim], np.float32)
    for i in range(dim):
        for j in range(dim):
            # calculate covariance for all combinations of dimensions
            heatmap_cov[i, j] = np.sum(coords_minus_mean[i, ...] * coords_minus_mean[j, ...] * cropped_normalized_heatmap)

    return heatmap_mean, heatmap_cov
