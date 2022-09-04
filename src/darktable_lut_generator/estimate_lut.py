"""
    Darktable LUT Generator: Generate .cube lookup tables from out-of-camera photos
    Copyright (C) 2021  Björn Sonnenschein

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import shutil

import colour
from scipy.optimize import linprog
from scipy import sparse
# import tensorflow as tf

from colour import LUT3D

from scipy.interpolate import RegularGridInterpolator
from plotly.subplots import make_subplots
import numpy as np
import cv2
import scipy.optimize
from sklearn.linear_model import Lasso
import logging
from tqdm import tqdm
import tempfile
import os
import subprocess
from plotly import graph_objects as go
from importlib.resources import path
from scipy.spatial import KDTree
from scipy.optimize import lsq_linear
from scipy import ndimage
import time

INTERPOLATORS = {
    'trilinear': colour.algebra.table_interpolation_trilinear,
    'tetrahedral': colour.algebra.table_interpolation_tetrahedral,
}


# FIXME: Something is wron with the export from darktable via command line:
#   for peter's dataset, consider the pattern images:
#   the developed raw outputted via the command line by this script differs
#   significantly from the output of darktable with my configuration if the same style is applied.
#   and yes, I took care that the history stack was the same before applying the style
#   and history handling was set to append in both cases.
#   but this is not really reproducible. sometimes, the images are fine.
#   I have no idea what's going on.

# FIXME: Regarding blue problems: look at the aligned pattern image. for some, the dark blues are black. why?
#   Buffer overflow while conversion?

# TODO: some boundary colors are off although enough samples are present.
#   would be nice to optimize with proper spatial regularization w.r.t. the lut colors
#   (maybe grmf prior)

# TODO: especially at extreme color valures, there are still outlier estimates
#   where colors are really off.
#   how does DT's lut 3D module transform into the application color space?
#   how are out of gamut colors handled?
#   is there a problem when exporting the sample images regarding the rendering intent,
#   so that out-of-gamut values mapping is not bijective?

# def estimate_transform_features(im1_grey, im2_grey):
#     import numpy as np
#     import cv2 as cv
#     MIN_MATCH_COUNT = 10
#     img1 = cv.imread('box.png', 0)  # queryImage
#     img2 = cv.imread('box_in_scene.png', 0)  # trainImage
#     # Initiate SIFT detector
#     sift = cv.SIFT_create()
#     # find the keypoints and descriptors with SIFT
#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)
#     flann = cv.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1, des2, k=2)
#     # store all the good matches as per Lowe's ratio test.
#     good = []
#     for m, n in matches:
#         if m.distance < 0.7 * n.distance:
#             good.append(m)
#
#     H, _ = cv.findHomography( features0.matched_pts, \
#         features1.matched_pts, cv.RANSAC, 5.0)

def align_images_ecc(im1, im2, edge_detection=False, translation_only=False, dir_out_info=None, name_1=None,
                     name_2=None):
    """Align image 1 to image 2.
    From https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/"""
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(
        im1,
        cv2.COLOR_BGR2GRAY
    )
    im2_gray = cv2.cvtColor(
        im2,
        cv2.COLOR_BGR2GRAY
    )

    # min max scaling
    im1_gray = ((im1_gray - np.min(im1_gray)) / (np.max(im1_gray) - np.min(im1_gray))).astype(np.float32)
    im2_gray = ((im2_gray - np.min(im2_gray)) / (np.max(im2_gray) - np.min(im2_gray))).astype(np.float32)

    if edge_detection:
        # im1_gray = cv2.Sobel(src=im1_gray, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=5)
        # TODO: better noise reduction than gauss
        im1_gray_edge = cv2.Sobel(src=cv2.GaussianBlur(im1_gray, (3, 3), 0), ddepth=cv2.CV_32F, dx=3, dy=3)
        # im1_gray_edge = cv2.Canny((im1_gray*255).astype(np.uint8), threshold1=0, threshold2=50).astype(np.float32) / 255.
        im1_gray_edge = (
                (im1_gray_edge - np.min(im1_gray_edge)) / (np.max(im1_gray_edge) - np.min(im1_gray_edge))).astype(
            np.float32)
        # im1_gray = cv2.Canny(im1_gray, 100, 100)
        # im2_gray = cv2.Sobel(src=im2_gray, ddepth=cv2.CV_32F, dx=1, dy=1, ksize=5)
        # im2_gray_edge = cv2.Laplacian(src=im2_gray, ddepth=cv2.CV_32F, ksize=1)
        im2_gray_edge = cv2.Sobel(src=cv2.GaussianBlur(im2_gray, (3, 3), 0), ddepth=cv2.CV_32F, dx=3, dy=3)
        im2_gray_edge = (
                (im2_gray_edge - np.min(im2_gray_edge)) / (np.max(im2_gray_edge) - np.min(im2_gray_edge))).astype(
            np.float32)

        # im2_gray = cv2.Canny(im2_gray, 100, 100)

    if dir_out_info is not None and name_1 is not None and name_2 is not None:
        path_dir_info_export = os.path.join(dir_out_info, 'alignment')
        max_ = get_max_value(np.zeros((1, 1), dtype=np.uint8))
        if not os.path.exists(path_dir_info_export):
            os.makedirs(path_dir_info_export)
        cv2.imwrite(os.path.join(path_dir_info_export, f'{name_1}_grayscale.png'), im1_gray * max_)
        cv2.imwrite(os.path.join(path_dir_info_export, f'{name_2}_grayscale.png'), im2_gray * max_)
        if edge_detection:
            cv2.imwrite(os.path.join(path_dir_info_export, f'{name_1}_edges.png'), im1_gray_edge * max_)
            cv2.imwrite(os.path.join(path_dir_info_export, f'{name_2}_edges.png'), im2_gray_edge * max_)

    # Define the motion model
    warp_mode = cv2.MOTION_TRANSLATION if translation_only else cv2.MOTION_AFFINE

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-8

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im2_gray, im1_gray, warp_matrix, warp_mode, criteria)
    if edge_detection:
        (cc, warp_matrix) = cv2.findTransformECC(im2_gray_edge, im1_gray_edge, warp_matrix, warp_mode, criteria)

    mask_ones = np.full_like(im1[..., 0], get_max_value(im1))

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im1_aligned = cv2.warpPerspective(im1, warp_matrix, (im2.shape[1], im2.shape[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        mask = cv2.warpPerspective(mask_ones, warp_matrix, (im2.shape[1], im2.shape[0]),
                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im1_aligned = cv2.warpAffine(im1, warp_matrix, (im2.shape[1], im2.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        mask = cv2.warpAffine(mask_ones, warp_matrix, (im2.shape[1], im2.shape[0]),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return im1_aligned, mask


def get_max_value(image: np.ndarray):
    if image.dtype == np.uint8:
        return 2 ** 8 - 1
    elif image.dtype == np.uint16:
        return 2 ** 16 - 1
    else:
        raise NotImplementedError


def get_aligned_image_pair(path_reference, path_raw, do_alignment, translation_only, interpolation, dir_out_info=None,
                           lut_alignment=None):
    reference = cv2.cvtColor(cv2.imread(path_reference, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    raw = cv2.cvtColor(cv2.imread(path_raw, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

    if reference.dtype != raw.dtype:
        raise ValueError(f'Images have different bit depth: {reference.dtype} != {raw.dtype}')
    if reference.dtype not in [np.uint8, np.uint16]:
        raise ValueError(f'Unsupported image dtype: {reference.dtype}')

    if do_alignment:
        # align the images
        if lut_alignment is None:
            raw_use = raw
        else:
            print('Applying estimated LUT to alignment raw image')
            raw_use = apply_lut_colour(raw, lut_alignment, interpolation)
        print(f'aligning image {path_reference}')
        reference_aligned, mask = align_images_ecc(
            reference,
            raw_use,
            translation_only=translation_only,
            dir_out_info=dir_out_info,
            name_1=os.path.basename(path_reference),
            name_2=os.path.basename(path_raw)
        )
        raw_aligned = raw
        print('Finished alignment')

    else:
        diff_size = np.asarray(raw.shape[:2]) - np.asarray(reference.shape[:2])
        crop_one_side = diff_size / 2
        crops = np.stack([np.floor(crop_one_side), np.ceil(crop_one_side)], axis=1).astype(int)

        def crop_dimension(raw, reference, crops, axis):
            if axis == 1:
                axes = [1, 0, 2]
                raw, reference = np.transpose(raw, axes), np.transpose(reference, axes)
            if crops[0] < 0:
                reference = reference[-crops[0]:]
            elif crops[0] > 0:
                raw = raw[crops[0]:]

            if crops[1] < 0:
                reference = reference[:crops[1]]
            elif crops[1] > 0:
                raw = raw[:-crops[1]]

            if axis == 1:
                axes = [1, 0, 2]
                raw, reference = np.transpose(raw, axes), np.transpose(reference, axes)

            return raw, reference

        raw_aligned, reference_aligned = crop_dimension(raw, reference, crops[0], 0)
        raw_aligned, reference_aligned = crop_dimension(raw_aligned, reference_aligned, crops[1], 1)

        mask = np.full_like(reference_aligned[..., 0], get_max_value(reference_aligned))

    mask_result = mask == get_max_value(reference)
    if dir_out_info is not None:
        path_dir_info_export = os.path.join(dir_out_info, 'alignment')
        if not os.path.exists(path_dir_info_export):
            os.makedirs(path_dir_info_export)
        mix = 0.5 * reference_aligned + 0.5 * raw_aligned
        cv2.imwrite(os.path.join(path_dir_info_export, f'{os.path.basename(path_reference)}_aligned_mix.png'),
                    cv2.cvtColor(mix.astype(raw_aligned.dtype), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(path_dir_info_export, f'{os.path.basename(path_reference)}_aligned_raw.png'),
                    cv2.cvtColor(raw_aligned, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(path_dir_info_export, f'{os.path.basename(path_reference)}_aligned_image.png'),
                    cv2.cvtColor(reference_aligned, cv2.COLOR_RGB2BGR))

    return reference_aligned, raw_aligned, mask_result


def estimate_lut(filepaths_images: [[str, str]], size, n_pixels_sample, is_grayscale, dir_out_info,
                 make_interpolated_red, make_unchanged_red, interpolate_unreliable, do_alignment,
                 align_translation_only,
                 sample_uniform,
                 interpolate_only_missing_data, interpolation, lut_alignment=None) -> np.ndarray:
    """
    :param filepaths_images: paths of image pairs: [reference, vanilla raw development]
    :return:
    """
    logging.info('Opening and aligning images')
    pixels_raws = []
    pixels_references = []
    for path_reference, path_raw in tqdm(filepaths_images):
        try:
            pixels_reference, pixels_raw, max_value = get_pixels_sample_image_pair(
                path_reference,
                path_raw,
                int(n_pixels_sample / len(filepaths_images)) if n_pixels_sample is not None else None,
                dir_out_info,
                do_alignment,
                sample_uniform,
                align_translation_only,
                lut_alignment,
                interpolation
            )
        except Exception as e:
            print(f'Image Alignment failed for images {os.path.basename(path_reference)}, {os.path.basename(path_raw)}.'
                  f'Skipping image.: {e}')
            continue

        pixels_raws.append(pixels_raw)
        pixels_references.append(pixels_reference)

    pixels_raws = np.concatenate(pixels_raws, axis=0)
    pixels_references = np.concatenate(pixels_references, axis=0)

    lut_result_normed = perform_estimation(pixels_references, pixels_raws, size, is_grayscale, interpolation,
                                           dir_out_info,
                                           make_interpolated_red, make_unchanged_red, interpolate_unreliable,
                                           interpolate_only_missing_data, lut_alignment)

    return lut_result_normed


def sample_uniform_from_histogram(histogram, edges, pixels, indices_pixels, n_samples):
    indices_bins_r = np.digitize(pixels[..., 0], edges[0]) - 1
    indices_bins_g = np.digitize(pixels[..., 1], edges[1]) - 1
    indices_bins_b = np.digitize(pixels[..., 2], edges[2]) - 1

    probability_densities_samples = histogram[indices_bins_r, indices_bins_g, indices_bins_b]
    weigths_samples = 1. / probability_densities_samples
    probabilities_samples = weigths_samples / np.sum(weigths_samples)
    indices_sampled = np.random.choice(indices_pixels, n_samples, p=probabilities_samples)

    return indices_sampled


def sample_indices_pixels(pixels, n_samples, uniform=False, size_batch_uniform=100000):
    if n_samples is None:
        return np.arange(0, pixels.shape[0])
    if uniform:
        # Generate sample that is approx. uniformly distributed w.r.t. pixel color values
        #   to enhance generalization of fitted lut coefficients and hence reduce needed sample size.
        #   Use histogram to estimate PDF and weight with the inverse
        n_bins = 10
        bins = np.stack([
            np.linspace(np.min(pixels[..., 0]), np.max(pixels[..., 0]) + 1e-10, n_bins),
            np.linspace(np.min(pixels[..., 1]), np.max(pixels[..., 1]) + 1e-10, n_bins),
            np.linspace(np.min(pixels[..., 2]), np.max(pixels[..., 2]) + 1e-10, n_bins),
        ],
            axis=0
        )
        histogram, edges = np.histogramdd(pixels, density=True,
                                          bins=bins)

        if size_batch_uniform is None:
            indices_pixels = np.arange(0, pixels.shape[0])
            return sample_uniform_from_histogram(histogram, edges, pixels, indices_pixels, n_samples)
        else:
            # Build the dataset consecutively from batches in order to circumvent
            #   numerical issues for very large images and very common pixel colors
            indices_list = []
            indices_pixels = np.arange(0, pixels.shape[0])
            n_samples_iteration = int(size_batch_uniform / 100.)
            for i in range(int(np.ceil(n_samples / n_samples_iteration))):
                indices_pixels_batch = np.random.choice(indices_pixels, size_batch_uniform, p=None)
                indices_list.append(sample_uniform_from_histogram(
                    histogram,
                    edges,
                    pixels[indices_pixels_batch],
                    indices_pixels_batch,
                    n_samples_iteration
                ))

            return np.concatenate(indices_list, axis=0)[:n_samples]
    else:
        indices = np.arange(0, pixels.shape[0])
        indices_sampled = np.random.choice(indices, n_samples, p=None)

    return indices_sampled


def get_pixels_sample_image_pair(path_reference, path_raw, n_samples, dir_out_info, do_alignment, sample_uniform,
                                 align_translation_only, lut_alignment, interpolation, dtype=np.float64):
    reference, raw, mask = get_aligned_image_pair(path_reference, path_raw, do_alignment, align_translation_only,
                                                  interpolation, dir_out_info, lut_alignment)
    max_value = get_max_value(reference)

    pixels_reference = np.reshape(
        reference,
        (
            reference.shape[0] * reference.shape[1],
            reference.shape[-1]
        )
    )[np.reshape(mask, mask.shape[0] * mask.shape[1])]
    pixels_raw = np.reshape(
        raw,
        (
            raw.shape[0] * raw.shape[1],
            raw.shape[-1]
        )
    )[np.reshape(mask, mask.shape[0] * mask.shape[1])]

    indices_sample = sample_indices_pixels(pixels_raw, n_samples, uniform=sample_uniform)
    result_raw = pixels_raw[indices_sample].astype(dtype) / max_value
    result_reference = pixels_reference[indices_sample].astype(dtype) / max_value

    return result_reference, result_raw, max_value


# def make_weights_distances_lut_entries_channels(pixels, size):
#     """ Get trilinear interpolation weights for LUT entries for each piel coordinate of the lut for one color axis.
#     """
#     coordinates = np.linspace(0, 1, size)
#     step_size = 1. / (size - 1)
#
#     # differences_channels is [... (pixels), channel, lut coordinate]
#     differences_channels = (
#             pixels[..., np.newaxis]
#             - np.expand_dims(np.stack([coordinates] * 3, axis=0), [i for i in range(pixels.ndim - 2)])
#     )
#     differences_channels_relative_grid_steps = differences_channels / step_size
#     weights_distances_channels = np.maximum(1. - np.abs(differences_channels_relative_grid_steps), 0.)
#
#     return weights_distances_channels

def make_weights_distances_lut_entries_channels(pixels, size):
    """ Get trilinear interpolation weights for LUT entries for each pixel coordinate of the lut for one color axis.
    """
    coordinates = np.linspace(0, 1, size)
    step_size = 1. / (size - 1)

    # differences_channels is [... (pixels), channel, lut coordinate]
    differences_channels = (
            pixels[..., np.newaxis]
            - np.expand_dims(np.stack([coordinates] * 3, axis=0), [i for i in range(pixels.ndim - 2)])
    )
    differences_channels_relative_grid_steps = differences_channels / step_size
    weights_distances_channels = np.maximum(1. - np.abs(differences_channels_relative_grid_steps), 0.)

    return weights_distances_channels


# def apply_lut(image, lut):
#     size = lut.shape[0]
#
#     max_value = get_max_value(image)
#     image_normed = image.astype(np.float64) / max_value
#     weights_distances_channels = make_weights_distances_lut_entries_channels(image_normed, size)
#
#     result = np.zeros_like(image_normed)
#     # TODO: speed up while still balancing memory usage
#     # result = apply_lut_pixel(lut, weights_distances_channels)
#     # traverse slices instead of interpolating whole image for memory usage limitation
#     for idx_y in range(image_normed.shape[0]):
#         result[idx_y] = apply_lut_pixel(
#             lut,
#             weights_distances_channels[idx_y]
#         )
#
#     result *= max_value
#
#     return result.astype(image.dtype)

def apply_lut_colour(image, lut, interpolation):
    size = lut.shape[0]
    lut_3d = LUT3D(table=lut, size=size)

    max_value = get_max_value(image)
    image_normed = image.astype(np.float64) / max_value

    result = lut_3d.apply(image_normed, interpolator=INTERPOLATORS[interpolation])
    result *= max_value

    return result.astype(image.dtype)


#
# def apply_lut_scipy(image, lut):
#     size = lut.shape[0]
#     coordinates = np.linspace(0, 1, size)
#     result = np.zeros_like(image, dtype=np.float64)
#
#     max_value = get_max_value(image)
#     image_normed = image.astype(np.float64) / max_value
#
#     for idx_channel in range(lut.shape[-1]):
#         interpolator = RegularGridInterpolator(
#             (coordinates, coordinates, coordinates),
#             lut[..., idx_channel]
#         )
#         pixels = np.reshape(image_normed, (image_normed[..., 0].size, image_normed.shape[-1]))
#         pixels_transformed = interpolator(
#             pixels
#         )
#         result[..., idx_channel] = np.reshape(pixels_transformed, image_normed.shape[:-1])
#
#     result *= max_value
#
#     return result.astype(image.dtype)

def apply_lut_pixel(lut, weights_distances_channels_pixel):
    # result = np.zeros(weights_distances_channels_pixel.shape[:-1], np.float)

    weights_entries_lut = (
            weights_distances_channels_pixel[..., 0, :, np.newaxis, np.newaxis]
            * weights_distances_channels_pixel[..., 1, np.newaxis, :, np.newaxis]
            * weights_distances_channels_pixel[..., 2, np.newaxis, np.newaxis, :]
    )

    result = np.sum(weights_entries_lut[..., np.newaxis] * lut, axis=(-2, -3, -4))

    # for idx_channel in range(3):
    #     result[..., idx_channel] = np.sum(weights_entries_lut * lut[..., idx_channel], axis=(-1, -2, -3))

    return result


def make_design_matrix(pixels_references, pixels_raws, size, interpolation):
    if interpolation not in INTERPOLATORS:
        raise ValueError(f'Interpolation {interpolation} not supported.')
    # feature matrix with order of permutation: r, g, b
    print('generating design matrix')
    design_matrix = np.zeros((pixels_references.shape[0], size * size * size), pixels_references.dtype)
    # design_matrix_new = np.zeros((pixels_references.shape[0], size * size * size), pixels_references.dtype)

    weights_distances_channels = make_weights_distances_lut_entries_channels(pixels_raws, size)

    lut = LUT3D(table=np.zeros((size, size, size, 3), dtype=pixels_raws.dtype), size=size)

    idx_design_matrix = 0
    for idx_r in tqdm(range(size)):
        for idx_g in range(size):
            for idx_b in range(size):
                if interpolation == 'linear':
                    # for each pixel, get the distance to the current lut grid point.
                    # from this, the weight of this point is calculated.
                    weights_entry_lut = (
                            weights_distances_channels[..., 0, idx_r]
                            * weights_distances_channels[..., 1, idx_g]
                            * weights_distances_channels[..., 2, idx_b]
                    )
                    design_matrix[..., idx_design_matrix] = weights_entry_lut
                else:
                    lut.table[idx_r, idx_g, idx_b] = 1.
                    design_matrix[..., idx_design_matrix] = lut.apply(
                        pixels_raws,
                        interpolator=INTERPOLATORS[interpolation]
                    )[..., 0]
                    lut.table[idx_r, idx_g, idx_b] = 0.
                idx_design_matrix += 1

    return design_matrix


def calc_is_trustful_estimate(design_matrix, size):
    # TODO: use OLS parameter estimator std error.
    #   Corresponding statistical assumptions are not met,
    #   but should suffice in practice.

    """TODO:
        calc std error.
        entries are unreliable if OLS std error is relatively large whereas lasso estimate is 0,
        meaning that it indicates that coefficient would most probably be non-zero  via OLS but is zero
        in lasso.
        Alternative: make OLS and LASSO and drop all coefficients that are zero in lasso but not in OLS.
    """
    sums_design_matrix = np.sum(np.abs(design_matrix), axis=0)
    has_enough_data = sums_design_matrix > design_matrix.shape[0] / size ** 3 / 10  # TODO: more sophisticated threshold
    has_no_data = sums_design_matrix < 1.

    return has_enough_data, has_no_data


def interpolate_best_missing_lut_entry(lut, indices_sufficient_data, indices_missing_data, make_interpolated_red):
    lut_result = np.copy(lut)

    n_neighbors_missing_data = []
    indices_direct_neighbors_missing_entries = []
    interpolator = KDTree(indices_sufficient_data)

    for idx_missing in indices_missing_data:
        distances, indices_nearest = interpolator.query(
            idx_missing,
            # distance_upper_bound=1.,
            k=8,
        )
        indices_direct_neighbors_ = indices_nearest[distances == 1.]
        n_direct_neighbors = indices_direct_neighbors_.shape[0]
        n_neighbors_missing_data.append(n_direct_neighbors)
        indices_direct_neighbors_missing_entries.append(indices_direct_neighbors_)

    idx_index_missing_most_direct_neighbors = np.argmax(n_neighbors_missing_data)
    index_missing_most_direct_neighbors = indices_missing_data[idx_index_missing_most_direct_neighbors]

    indices_missing_result = np.asarray([
        index_missing
        for idx, index_missing in enumerate(indices_missing_data)
        if idx != idx_index_missing_most_direct_neighbors
    ])
    indices_sufficient_data_result = np.concatenate([
        index_missing_most_direct_neighbors[np.newaxis, ...],
        indices_sufficient_data
    ])

    # indices of the direct neighbor of the missing lut entry that shall be interpolated.
    indices_direct_neighbors = indices_sufficient_data[
        indices_direct_neighbors_missing_entries[
            idx_index_missing_most_direct_neighbors
        ]
    ]
    direct_neighbors = np.asarray([lut[i[0], i[1], i[2]] for i in indices_direct_neighbors])
    # lut_result[index_missing_most_direct_neighbors] = np.mean(direct_neighbors, axis=0)
    lut_result[
        index_missing_most_direct_neighbors[0],
        index_missing_most_direct_neighbors[1],
        index_missing_most_direct_neighbors[2],
    ] = np.mean(direct_neighbors, axis=0) if not make_interpolated_red else np.asarray([1, 0, 0])
    # ] = np.asarray([1, 0, 0])

    return lut_result, indices_sufficient_data_result, indices_missing_result


def interpolate_unreliable_lut_entries(design_matrix, lut, only_without_data, make_interpolated_red):
    indices_lut = make_meshgrid_cube_coordinates(lut.shape[0]).reshape([lut.shape[0] ** 3, 3])
    has_enough_data, has_no_data = calc_is_trustful_estimate(design_matrix, lut.shape[0])

    indices_invalid = has_no_data if only_without_data else np.logical_not(has_enough_data)
    # indices_invalid = has_no_data

    indices_missing_data = np.argwhere(indices_invalid.reshape(lut.shape[:3]))
    indices_sufficient_data = indices_lut[np.logical_not(indices_invalid)]

    result = lut
    while indices_missing_data.shape[0]:
        result, indices_sufficient_data, indices_missing_data = interpolate_best_missing_lut_entry(
            result,
            indices_sufficient_data,
            indices_missing_data,
            make_interpolated_red
        )

    return result


def save_info_fitting(lut, design_matrix, dir_out_info, residuals_channels, pixels_references, pixels_raws):
    # Make 3d cube plot where outline is coordinate of lut node and inner color is mapped color

    identity = make_lut_identity_normed(lut.shape[0])
    coords = identity.reshape(lut.shape[0] ** 3, 3)

    lut_rounded = np.round(lut, 2)

    colors_mapped = [f'rgb({x[0] * 255},{x[1] * 255},{x[2] * 255})' for x in
                     lut_rounded.reshape((lut_rounded.shape[0] ** 3, 3))]
    colors_coordinates = [f'rgb({x[0] * 255},{x[1] * 255},{x[2] * 255})' for x in coords]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=coords[..., 0],
            y=coords[..., 1],
            z=coords[..., 2],
            mode='markers',
            marker=dict(
                line=dict(
                    width=5,
                    color=colors_mapped
                ),
                color=colors_coordinates
            ),
        ),
    )
    # fig.show()

    fig.write_html(os.path.join(dir_out_info, 'lut.html'))

    has_enough_data, has_no_data = calc_is_trustful_estimate(design_matrix, lut.shape[0])
    colors_valid = []
    for has_enough_data_, has_no_data_ in zip(has_enough_data, has_no_data):
        colors_valid.append(
            'rgb(0,255,0)' if has_enough_data_ else 'rgb(255,0,0)' if has_no_data_ else 'rgb(255,255,0)'
        )
    # colors_valid = ['rgb(0,255,0)' if x else 'rgb(255,0,0)' for x in has_enough_data]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=coords[..., 0],
            y=coords[..., 1],
            z=coords[..., 2],
            mode='markers',
            marker=dict(
                line=dict(
                    width=3,
                    color=colors_valid
                ),
                color=colors_coordinates
            ),
        ),
    )
    # fig.show()

    fig.write_html(os.path.join(dir_out_info, 'lut_no_datapoints.html'))

    # Residuals and datapoints
    for idx_channel, residuals_channel in enumerate(residuals_channels):
        fig = make_subplots(2, 2,
                            specs=[
                                [
                                    {'type': 'scene'},
                                    {'type': 'scene'},
                                ],
                                [
                                    {'type': 'scene'},
                                    {'type': 'xy'},
                                ]
                            ]
                            )

        fig.add_trace(
            go.Scatter3d(
                x=pixels_raws[:, 0],
                y=pixels_raws[:, 1],
                z=residuals_channel,
                mode='markers',
                marker={'size': 1}
            ),
            col=1,
            row=1
        )

        fig.add_trace(
            go.Scatter3d(
                x=pixels_raws[:, 0],
                y=pixels_raws[:, 2],
                z=residuals_channel,
                mode='markers',
                marker={'size': 1}
            ),
            col=2,
            row=1
        )
        fig.add_trace(
            go.Scatter3d(
                x=pixels_raws[:, 1],
                y=pixels_raws[:, 2],
                z=residuals_channel,
                mode='markers',
                marker={'size': 1}
            ),
            col=1,
            row=2
        )

        fig.add_trace(
            go.Histogram(
                x=residuals_channel
            ),
            col=2,
            row=2
        )

        # Update xaxis properties
        fig.update_xaxes(title_text="channel 0", row=1, col=1)
        fig.update_xaxes(title_text="channel 0", row=1, col=2)
        fig.update_xaxes(title_text="channel 1", row=2, col=1)
        fig.update_xaxes(title_text="residual", row=2, col=2)

        # Update yaxis properties
        fig.update_yaxes(title_text="channel 1", row=1, col=1)
        fig.update_yaxes(title_text="channel 2", row=1, col=2)
        fig.update_yaxes(title_text="channel 2", row=2, col=1)
        fig.update_yaxes(title_text="count", row=2, col=2)

        fig.write_html(os.path.join(dir_out_info, f'residuals_channel_{idx_channel}.html'))

        ##### Datapoints
        fig = make_subplots(2, 2,
                            specs=[
                                [
                                    {'type': 'scene'},
                                    {'type': 'scene'},
                                ],
                                [
                                    {'type': 'scene'},
                                    {'type': 'xy'},
                                ]
                            ]
                            )

        fig.add_trace(
            go.Scatter3d(
                x=pixels_raws[:, 0],
                y=pixels_raws[:, 1],
                z=pixels_references[:, idx_channel],
                mode='markers',
                marker={'size': 1}
            ),
            col=1,
            row=1
        )

        fig.add_trace(
            go.Scatter3d(
                x=pixels_raws[:, 0],
                y=pixels_raws[:, 2],
                z=pixels_references[:, idx_channel],
                mode='markers',
                marker={'size': 1}
            ),
            col=2,
            row=1
        )
        fig.add_trace(
            go.Scatter3d(
                x=pixels_raws[:, 1],
                y=pixels_raws[:, 2],
                z=pixels_references[:, idx_channel],
                mode='markers',
                marker={'size': 1}
            ),
            col=1,
            row=2
        )

        fig.add_trace(
            go.Histogram(
                x=pixels_raws[:, idx_channel]
            ),
            col=2,
            row=2
        )

        # Update xaxis properties
        fig.update_xaxes(title_text="raw channel 0", row=1, col=1)
        fig.update_xaxes(title_text="raw channel 0", row=1, col=2)
        fig.update_xaxes(title_text="raw channel 1", row=2, col=1)
        fig.update_xaxes(title_text=f"RAW channel {idx_channel}", row=2, col=2)

        # Update yaxis properties
        fig.update_yaxes(title_text="raw channel 1", row=1, col=1)
        fig.update_yaxes(title_text="raw channel 2", row=1, col=2)
        fig.update_yaxes(title_text="raw channel 2", row=2, col=1)
        fig.update_yaxes(title_text="count", row=2, col=2)

        fig.write_html(os.path.join(dir_out_info, f'datapoints_channel_{idx_channel}.html'))


#
def constrained_quantile_regression(design_matrix, y, bounds_lower, bounds_upper, quantile=0.5):
    # Simple linear programming implementation of constrained quantile regression
    #   adapted from h
    #   ttps://stats.stackexchange.com/questions/384909/formulating-quantile-regression-as-linear-programming-problem

    K = design_matrix.shape[1]
    N = design_matrix.shape[0]

    # equality constraints - left hand side

    A1 = design_matrix  # intercepts & data points - positive weights
    A2 = design_matrix * - 1  # intercept & data points - negative weights
    A3 = np.identity(N, dtype=design_matrix.dtype)  # error - positive
    A4 = np.identity(N, dtype=design_matrix.dtype) * -1  # error - negative

    A_eq = np.concatenate((A1, A2, A3, A4), axis=1)  # all the equality constraints

    # equality constraints - right hand side
    b_eq = y

    # goal function - intercept & data points have 0 weights
    # positive error has tau weight, negative error has 1-tau weight
    c = np.concatenate((np.repeat(0, 2 * K), quantile * np.repeat(1, N), (1 - quantile) * np.repeat(1, N))).astype(
        design_matrix.dtype)

    # all variables must be greater than zero
    # adding inequality constraints - left hand side
    n = A_eq.shape[-1]
    A_ub = np.full((n, n), 0., dtype=design_matrix.dtype)
    A_ub[::n + 1] = -1.0

    # adding inequality constraints - right hand side (all zeros)
    b_ub = np.full((n, 1), 0., dtype=design_matrix.dtype)

    # add parameter bounda
    print('Inserting bounds into constraint arrays')
    zeros = np.zeros((1, n), dtype=design_matrix.dtype)
    bounds_left = []
    bounds_right = []
    for idx_parameter in range(K):
        bounds_left_upper_param = zeros.copy()
        bounds_left_upper_param[0, [idx_parameter, idx_parameter + K]] = np.asarray([1, -1], dtype=design_matrix.dtype)
        bounds_right_upper_param = np.full((1, 1), bounds_upper[idx_parameter], dtype=design_matrix.dtype)

        bounds_left.append(bounds_left_upper_param)
        bounds_right.append(bounds_right_upper_param)
        # A_ub = np.concatenate([A_ub, bounds_left_upper_param], axis=0)
        # b_ub = np.concatenate([b_ub, bounds_right_upper_param], axis=0)

        bounds_left_lower_param = zeros.copy()
        bounds_left_lower_param[0, [idx_parameter, idx_parameter + K]] = np.asarray([-1, 1], dtype=design_matrix.dtype)
        bounds_right_lower_param = np.full((1, 1), -bounds_lower[idx_parameter], dtype=design_matrix.dtype)

        bounds_left.append(bounds_left_lower_param)
        bounds_right.append(bounds_right_lower_param)

        # A_ub = np.concatenate([A_ub, bounds_left_lower_param], axis=0)
        # b_ub = np.concatenate([b_ub, bounds_right_lower_param], axis=0)

    A_ub = np.concatenate([A_ub, *bounds_left], axis=0)
    b_ub = np.concatenate([b_ub, *bounds_right], axis=0)

    print('Making sparse matrices')

    # c = scipy.sparse.csc_array(c)
    A_ub = scipy.sparse.csc_array(A_ub)
    # b_ub = scipy.sparse.csc_array(b_ub)
    A_eq = scipy.sparse.csc_array(A_eq)
    # b_eq = scipy.sparse.csc_array(b_eq)

    print('Starting fit')
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        method='highs-ds'
    )

    x = res.x

    # both negative and positive components get values above zero, this gets fixed here
    coefficients = x[:K] - x[K:2 * K]

    return coefficients

def fit_channel_smoothness_penalty(design_matrix, differences_references_raw_channel, idx_channel, size):
    print(f'Fitting channel {idx_channel}')
    stds = np.std(design_matrix, axis=0)
    stds[stds == 0] = 1.
    identity = make_lut_identity_normed(size)

    design_matrix_scaled = design_matrix / stds[np.newaxis, ...]

    bounds_lower = (-1 * identity[..., idx_channel].reshape([size ** 3]))
    bounds_lower_scaled = bounds_lower * stds
    bounds_upper = (1. - identity[..., idx_channel]).reshape([size ** 3])
    bounds_upper_scaled = bounds_upper * stds

    bounds_list = [(bounds_lower_scaled[idx], bounds_upper_scaled[idx]) for idx in range(size ** 3)]

    def loss(coeffs):
        regularization_strength = 1e-5
        estimate = np.matmul(design_matrix_scaled, coeffs)
        mse = np.mean((differences_references_raw_channel - estimate) ** 2)
        coeffs_rescaled = coeffs / stds
        array_changes = coeffs_rescaled.reshape((size, size, size))
        grad_magnitude = ndimage.generic_gradient_magnitude(array_changes, ndimage.sobel)
        penalty = np.mean(grad_magnitude ** 2)

        result = mse + penalty * regularization_strength


        return result

    print('Fitting OLS start parameters')
    params_start = lsq_linear(design_matrix_scaled, differences_references_raw_channel,
                              (bounds_lower_scaled, bounds_upper_scaled)).x

    print('Fitting regularized least squares')
    result = scipy.optimize.minimize(
        loss,
        params_start,
        method='Powell',
        bounds=bounds_list,
        callback=lambda x_: print(loss(x_))
        # tol=1e-2
    )

    coeffs_rescaled = result.x / stds

    return coeffs_rescaled


def fit_channel_constrained_abs_dev(design_matrix, differences_references_raw_channel, idx_channel, size):
    stds = np.std(design_matrix, axis=0)
    stds[stds == 0] = 1.
    identity = make_lut_identity_normed(size, dtype=design_matrix.dtype)

    design_matrix_scaled = design_matrix / stds[np.newaxis, ...]

    bounds_lower = (-1 * identity[..., idx_channel].reshape([size ** 3]))
    bounds_lower_scaled = bounds_lower * stds
    bounds_upper = (1. - identity[..., idx_channel]).reshape([size ** 3])
    bounds_upper_scaled = bounds_upper * stds

    print('Calculating least absolute deviation solution')

    coeffs = constrained_quantile_regression(
        design_matrix_scaled,
        differences_references_raw_channel,
        bounds_lower_scaled,
        bounds_upper_scaled
    )

    # coeffs = regressor.coef_
    coeffs_rescaled = coeffs / stds

    return coeffs_rescaled


def fit_channel_constrained(design_matrix, differences_references_raw_channel, idx_channel, size):
    stds = np.std(design_matrix, axis=0)
    stds[stds == 0] = 1.
    identity = make_lut_identity_normed(size, dtype=design_matrix.dtype)

    # design_matrix_scaled = design_matrix / stds[np.newaxis, ...]

    bounds_lower = (-1 * identity[..., idx_channel].reshape([size ** 3]))
    # bounds_lower_scaled = bounds_lower * stds
    bounds_upper = (1. - identity[..., idx_channel]).reshape([size ** 3])
    # bounds_upper_scaled = bounds_upper * stds

    # regression = LinearRegression(fit_intercept=False)
    t1 = time.time()
    # result_opt = lsq_linear(design_matrix_scaled, differences_references_raw_channel,
    #                         (bounds_lower_scaled, bounds_upper_scaled)
    #                         )
    result_opt = lsq_linear(
        design_matrix,
        differences_references_raw_channel,
        (bounds_lower, bounds_upper)
    )
    # coeffs_rescaled = result_opt.x / stds
    coeffs_rescaled = result_opt.x
    t2 = time.time()
    print(f'Fitted in {t2 - t1} seconds.')

    return coeffs_rescaled


def fit_channel_lasso(design_matrix, differences_references_raw_channel, idx_channel, size):
    stds = np.std(design_matrix, axis=0)
    stds[stds == 0] = 1.

    design_matrix_scaled = design_matrix / stds[np.newaxis, ...]
    regression = Lasso(
        alpha=1e-6,
        fit_intercept=False,
        tol=1e-4,
        selection='random'
    )
    # regression = LinearRegression(fit_intercept=False)
    regression.fit(design_matrix_scaled, differences_references_raw_channel)
    coeffs_rescaled = regression.coef_ / stds

    return coeffs_rescaled


def perform_estimation(pixels_references, pixels_raws, size, is_grayscale, interpolation, dir_out_info=None,
                       make_interpolated_red=False, make_unchanged_red=False, interpolate_unreliable=True,
                       interpolate_only_missing_data=False, lut_start=None):
    design_matrix = make_design_matrix(pixels_references, pixels_raws, size, interpolation)

    print('fitting lookup table coefficients')

    result = make_lut_identity_normed(size)
    differences_references_raw = pixels_references - pixels_raws
    rmse_pre_channnels = []
    rmse_past_channels = []
    changes = np.zeros_like(result)

    stds = np.std(design_matrix, axis=0)
    stds[stds == 0] = 1.

    residuals_channels = []

    for idx_channel in range(3):
        rmse_pre_channnels.append(np.sqrt(np.mean(differences_references_raw[..., idx_channel] ** 2)))
        print(f'estimating channel {idx_channel}')

        # coefficients = fit_channel_constrained_abs_dev(
        coefficients = fit_channel_constrained(
            # coefficients = fit_channel_tf(
            design_matrix,
            differences_references_raw[..., idx_channel],
            idx_channel,
            size,
            # lut_start
        )

        residuals_channels.append(
            differences_references_raw[..., idx_channel]
            - np.matmul(design_matrix, coefficients)
        )

        rmse_past_channels.append(
            np.sqrt(np.mean(
                residuals_channels[-1] ** 2
            ))
        )
        lut_difference_channel = np.reshape(coefficients, [size, size, size])

        # todo: refactor to use changes array after loop to fill result
        if is_grayscale:
            lut_all_channels = result[..., 0] + lut_difference_channel
            result[..., 0] = lut_all_channels
            result[..., 1] = lut_all_channels
            result[..., 2] = lut_all_channels
            changes[..., 0] = lut_difference_channel
            changes[..., 1] = lut_difference_channel
            changes[..., 2] = lut_difference_channel
            break
        else:
            changes[..., idx_channel] = lut_difference_channel
            result[..., idx_channel] += lut_difference_channel

    if interpolate_unreliable:
        result = interpolate_unreliable_lut_entries(design_matrix, result, interpolate_only_missing_data,
                                                    make_interpolated_red)

    result = np.clip(result, a_min=0., a_max=1.)

    if make_unchanged_red:
        result[np.sqrt(np.sum(changes ** 2, axis=-1)) < 0.001] = np.asarray([1., 0., 0.])

    if dir_out_info is not None:
        save_info_fitting(result, design_matrix, dir_out_info, residuals_channels, pixels_references, pixels_raws)

    print(f'channels rmse without lut: {rmse_pre_channnels}')
    print(f'channels rmse with fitted lut: {rmse_past_channels}')

    return result


def make_meshgrid_cube_coordinates(size):
    return np.stack(
        np.meshgrid(
            *([
                  np.arange(0, size)[np.newaxis, ...],
              ] * 3),
            indexing='ij'
        ),
        axis=-1
    )


def make_lut_identity_normed(size, dtype=np.float32):
    # identity with [r,g,b, channel]
    result = np.stack(
        np.meshgrid(
            *([
                  np.linspace(0, 1., size)[np.newaxis, ...],
              ] * 3),
            indexing='ij'
        ),
        axis=-1
    ).astype(dtype)

    return result

def get_name_style(path_style):
    with open(path_style) as f:
        str_style = f.read()

    return str_style.split('<name>')[1].split('</name>')[0]


def main(dir_images, file_out, size=9, n_pixels_sample=100000, is_grayscale=False, resize=0,
         path_dt_exec=None,
         path_style_image_user=None, path_style_raw_user=None, path_dir_intermediate=None, dir_out_info=None,
         make_interpolated_red=False, make_unchanged_red=False, interpolate_unreliable=True,
         use_lens_correction=True, legacy_color=False, n_passes_alignment=1,
         align_translation_only=False,
         sample_uniform=False, interpolate_only_missing_data=False, interpolation='trilinear',
         paths_dirs_files_config_use=None, title_lut=None, comment_lut=None):
    extensions_raw = ['raw', 'raf', 'dng', 'nef', 'cr3', 'arw', 'cr2', 'cr3', 'orf', 'rw2']
    extensions_image = ['jpg', 'jpeg', 'tiff', 'tif', 'png']

    pairs_images = []
    for filename_raw in os.listdir(dir_images):
        path_raw = os.path.join(dir_images, filename_raw)
        base_raw, extension_raw = os.path.splitext(filename_raw)
        if extension_raw[1:].lower() in extensions_raw:
            for filename_image in os.listdir(dir_images):
                path_image = os.path.join(dir_images, filename_image)
                base_image, extension_image = os.path.splitext(filename_image)
                if extension_image[1:].lower() in extensions_image and base_image == base_raw:
                    pairs_images.append((path_image, path_raw))

    # use darktable to generate images
    with tempfile.TemporaryDirectory() as path_dir_temp:
        if path_dir_intermediate is not None:
            path_dir_temp = path_dir_intermediate
        filepaths_images_converted = []

        path_dir_images_temp = os.path.join(path_dir_temp, 'images')
        os.mkdir(path_dir_images_temp)

        path_dir_conf_temp = os.path.join(path_dir_temp, 'conf')
        os.mkdir(path_dir_conf_temp)
        path_styles_temp = os.path.join(path_dir_temp, 'styles')
        os.mkdir(path_styles_temp)
        print(path_dir_conf_temp)

        # if supplied, fill conf dir with user data
        if paths_dirs_files_config_use is not None:
            paths_config = paths_dirs_files_config_use.split(',')
            for path_ in paths_config:
                path_ = os.path.normpath(path_)
                if os.path.isfile(path_):
                    shutil.copyfile(path_, path_dir_conf_temp)
                else:
                    shutil.copytree(path_, os.path.join(path_dir_conf_temp, os.path.basename(path_)))

        with path('darktable_lut_generator.styles', 'image.dtstyle') as path_style_image_default:
            path_style_image = path_style_image_user if path_style_image_user is not None else path_style_image_default
            path_style_image_temp = os.path.join(path_styles_temp, 'image.dtstyle')
            shutil.copyfile(path_style_image, path_style_image_temp)
        with path(
                'darktable_lut_generator.styles',
                'raw_lens_correction.dtstyle'
        ) as path_style_raw_default:
            path_style_raw = path_style_raw_user if path_style_raw_user is not None else path_style_raw_default
            path_style_raw_temp = os.path.join(path_styles_temp, 'raw.dtstyle')
            shutil.copyfile(path_style_raw, path_style_raw_temp)

        args_common = [
            '--width',
            str(resize),
            '--height',
            str(resize),
            '--icc-type',
            'ADOBERGB',
            # '--icc-intent',
            # 'ABSOLUTE_COLORIMETRIC',
            '--style-overwrite',
            # TODO: activating leads to the color calibration module not rendered on export despite it being active in darkroom.
            '--core',
            '--configdir',
            path_dir_conf_temp,
            '--library',
            ':memory:',
            '--conf',
            f'plugins/darkroom/chromatic-adaptation={"legacy" if legacy_color else "modern"}',
            '--conf',
            'plugins/darkroom/sharpen/auto_apply=FALSE',
            '--conf',
            'plugins/darkroom/workflow=none',
            '--conf',
            'opencl=FALSE'
        ]

        for path_image, path_raw in pairs_images:
            path_out_image = os.path.join(path_dir_temp, os.path.basename(path_image) + '.png')
            path_out_raw = os.path.join(path_dir_temp, os.path.basename(path_raw) + '.png')
            print(f'converting image {os.path.basename(path_image)}')

            # Copy the images so that no accompanying .xmp files are present
            #   because for some reason, color shifts etc. occur when developing the raw from the local files
            #   even with style-overwrite flag.
            path_in_image = os.path.join(path_dir_images_temp, os.path.basename(path_image))
            path_in_raw = os.path.join(path_dir_images_temp, os.path.basename(path_raw))
            shutil.copyfile(path_image, path_in_image)
            shutil.copyfile(path_raw, path_in_raw)

            args = [
                'darktable-cli' if path_dt_exec is None else path_dt_exec,
                path_in_image,
                path_out_image,
                *args_common,
            ] if path_style_image_user is None else [
                'darktable-cli' if path_dt_exec is None else path_dt_exec,
                path_in_image,
                path_out_image,
                # '--style-overwrite',
                '--style',
                get_name_style(path_style_image_temp),
                *args_common,
                "--luacmd",
                f"local dt = require \"darktable\"; dt.styles.import(\"{path_style_image_temp}\")"
            ]
            print(' '.join(args))
            subprocess.call(
                args,
                timeout=1e10
            )
            print(f'converting raw {os.path.basename(path_raw)}')

            args = [
                'darktable-cli' if path_dt_exec is None else path_dt_exec,
                path_in_raw,
                path_out_raw,
                # '--style',
                # get_name_style(path_style_raw_temp),
                *args_common,
                # "--luacmd",
                # f"local dt = require \"darktable\"; dt.styles.import(\"{path_style_raw_temp}\")"
            ] if path_style_raw_user is None and not use_lens_correction else [
                'darktable-cli' if path_dt_exec is None else path_dt_exec,
                path_in_raw,
                path_out_raw,
                # '--style-overwrite',
                '--style',
                get_name_style(path_style_raw_temp),
                *args_common,
                "--luacmd",
                f"local dt = require \"darktable\"; dt.styles.import(\"{path_style_raw_temp}\")"
            ]
            print(' '.join(args))
            subprocess.call(
                args,
                timeout=1e10
            )

            filepaths_images_converted.append((path_out_image, path_out_raw))

        if dir_out_info:
            path_dir_info_export = os.path.join(dir_out_info, 'export_darktable')
            if not os.path.exists(path_dir_info_export):
                os.makedirs(path_dir_info_export)

            for path_image, path_raw in filepaths_images_converted:
                shutil.copyfile(path_image, os.path.join(path_dir_info_export, os.path.basename(path_image)))
                shutil.copyfile(path_raw, os.path.join(path_dir_info_export, os.path.basename(path_raw)))

        print('Finished converting. Generating LUT.')
        # a halc clut is a cube with level**2 entries on each dimension
        lut_alignment = None

        if n_passes_alignment > 1:
            for idx_pass in range(n_passes_alignment - 1):
                print(
                    f'Estimating approximate first-pass LUT for alignment: Pass {idx_pass + 1} of {n_passes_alignment - 1}')
                lut_alignment = estimate_lut(filepaths_images_converted, size, n_pixels_sample, is_grayscale, None,
                                             False, False, interpolate_unreliable, lut_alignment is not None,
                                             align_translation_only, sample_uniform, interpolate_only_missing_data,
                                             interpolation, lut_alignment)

        result = estimate_lut(filepaths_images_converted, size, n_pixels_sample, is_grayscale, dir_out_info,
                              make_interpolated_red, make_unchanged_red, interpolate_unreliable, n_passes_alignment > 0,
                              align_translation_only, sample_uniform, interpolate_only_missing_data, interpolation,
                              lut_alignment)

        print(f'Writing result to {file_out}')
        write_cube(result, file_out, title_lut, comment_lut)

        if dir_out_info is not None:
            print('Exporting transformed images')
            path_dir_info_image = os.path.join(dir_out_info, 'reference_and_transformed')
            if not os.path.exists(path_dir_info_image):
                os.mkdir(path_dir_info_image)
            for path_reference, path_raw in tqdm(filepaths_images_converted):
                raw = cv2.cvtColor(cv2.imread(path_raw, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

                raw_transformed = apply_lut_colour(raw, result, interpolation)
                cv2.imwrite(
                    os.path.join(path_dir_info_image, os.path.basename(path_raw)),
                    cv2.cvtColor(raw_transformed, cv2.COLOR_RGB2BGR)
                )
                cv2.imwrite(
                    os.path.join(path_dir_info_image, os.path.basename(path_reference)),
                    cv2.imread(path_reference, cv2.IMREAD_UNCHANGED)
                )

    return result


def write_cube(lut: np.ndarray, path_output, title, comment):
    size = lut.shape[0] ** 3
    lut_flattened = np.reshape(np.swapaxes(lut, 0, 2), (size, 3))

    s = '{:.10f}'

    with open(path_output, 'w') as f:
        f.write('# Generated by darktable_lut_creator: https://github.com/wilecoyote2015/darktabe_lut_generator\n')
        if comment:
            f.write(f'# {comment}\n')
        f.write(f'TITLE "{title if title is not None else os.path.splitext(os.path.basename(path_output))[0]}"\n')
        f.write(f'LUT_3D_SIZE {lut.shape[0]}\n')
        f.write('\n')
        for idx in range(lut_flattened.shape[0]):
            f.write(
                f'{s.format(lut_flattened[idx][0])} {s.format(lut_flattened[idx][1])} {s.format(lut_flattened[idx][2])}\n')
