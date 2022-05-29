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

import numpy as np
import cv2
from sklearn.linear_model import Lasso
import logging
from tqdm import tqdm
import tempfile
import os
import subprocess
from importlib.resources import path

def align_images_ecc(im1, im2):
    """Align image 1 to image 2.
    From https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/"""
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    if im1_gray.dtype == np.uint16:
        im1_gray = (im1_gray / 2 ** 8).astype(np.uint8)
        im2_gray = (im2_gray / 2 ** 8).astype(np.uint8)

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-5

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im2_gray, im1_gray, warp_matrix, warp_mode, criteria)

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


def get_aligned_image_pair(path_reference, path_raw):
    reference = cv2.cvtColor(cv2.imread(path_reference, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    raw = cv2.cvtColor(cv2.imread(path_raw, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

    if reference.dtype != raw.dtype:
        raise ValueError(f'Images have different bit depth: {reference.dtype} != {raw.dtype}')
    if reference.dtype not in [np.uint8, np.uint16]:
        raise ValueError(f'Unsupported image dtype: {reference.dtype}')

    # align the images
    print(f'aligning image {path_reference}')
    reference_aligned, mask = align_images_ecc(
        reference,
        raw
    )
    print('Finished alignment')
    mask_result = mask == get_max_value(reference)
    # mix = 0.5 * reference_aligned + 0.5 * np.asarray(raw)
    # cv2.imwrite(os.path.join('..', 'output', os.path.basename(path_reference)), mix)

    return reference_aligned, raw, mask_result


def estimate_lut(filepaths_images: [[str, str]], size, n_pixels_sample, is_grayscale) -> np.ndarray:
    """
    :param filepaths_images: paths of image pairs: [reference, vanilla raw development]
    :return:
    """
    logging.info('Opening and aligning images')
    pixels_raws = []
    pixels_references = []
    for path_reference, path_raw in tqdm(filepaths_images):
        pixels_reference, pixels_raw, max_value = get_pixels_sample_image_pair(
            path_reference,
            path_raw,
            int(n_pixels_sample / len(filepaths_images)) if n_pixels_sample is not None else None
        )
        pixels_raws.append(pixels_raw)
        pixels_references.append(pixels_reference)

    pixels_raws = np.concatenate(pixels_raws, axis=0)
    pixels_references = np.concatenate(pixels_references, axis=0)

    lut_result_normed = perform_estimation_linear_regression(pixels_references, pixels_raws, size, is_grayscale)

    return lut_result_normed


def sample_uniform_from_histogram(histogram, edges, pixels, indices_pixels, n_samples):
    indices_bins_r = np.digitize(pixels[..., 0], edges[0]) - 1
    indices_bins_g = np.digitize(pixels[..., 1], edges[1]) - 1
    indices_bins_b = np.digitize(pixels[..., 2], edges[2]) - 1

    probability_densities_samples = histogram[indices_bins_r, indices_bins_g, indices_bins_b]
    weigths_samples = 1. / probability_densities_samples
    probabilities_samples = weigths_samples / np.sum(weigths_samples)
    indices_sampled = np.random.choice(indices_pixels, n_samples, p=probabilities_samples, replace=False)

    return indices_sampled


def sample_indices_pixels(pixels, n_samples, uniform=True, size_batch_uniform=100000):
    if n_samples is None:
        return np.arange(0, pixels.shape[0])
    if uniform:
        # Generate sample that is approx. uniformly distributed w.r.t. pixel color values
        #   to enhance generalization of fitted lut coefficients and hence reduce needed sample size.
        #   Use histogram to estimate PDF and weight with the inverse
        bins = np.stack([
            np.linspace(np.min(pixels[..., 0]), np.max(pixels[..., 0]) + 1e-10, 10),
            np.linspace(np.min(pixels[..., 1]), np.max(pixels[..., 1]) + 1e-10, 10),
            np.linspace(np.min(pixels[..., 2]), np.max(pixels[..., 2]) + 1e-10, 10),
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


def get_pixels_sample_image_pair(path_reference, path_raw, n_samples):
    reference, raw, mask = get_aligned_image_pair(path_reference, path_raw)
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

    indices_sample = sample_indices_pixels(pixels_raw, n_samples)
    result_raw = pixels_raw[indices_sample].astype(np.float64) / max_value
    result_reference = pixels_reference[indices_sample].astype(np.float64) / max_value

    return result_reference, result_raw, max_value


def make_design_matrix(pixels_references, pixels_raws, size):
    coordinates = np.linspace(0, 1, size)
    step_size = 1. / (size - 1)

    # feature matrix with order of permutation: r, g, b
    print('generating design matrix')
    design_matrix = np.zeros((pixels_references.shape[0], size * size * size), np.float32)

    # [pixel, channel, channel_value]
    differences_channels = pixels_raws[..., np.newaxis] - np.stack([coordinates] * 3, axis=0)[np.newaxis, ...]
    differences_channels_relative_grid_steps = differences_channels / step_size
    weights_distances_channels = np.maximum(1. - np.abs(differences_channels_relative_grid_steps), 0.)

    idx_design_matrix = 0
    for idx_r in tqdm(range(size)):
        for idx_g in range(size):
            for idx_b in range(size):
                # for each pixel, get the distance to the current lut grid point
                # from this, the weight of this point is calculated.
                weights_entry_lut = (
                        weights_distances_channels[..., 0, idx_r]
                        * weights_distances_channels[..., 1, idx_g]
                        * weights_distances_channels[..., 2, idx_b]
                )
                design_matrix[..., idx_design_matrix] = weights_entry_lut
                idx_design_matrix += 1

    return design_matrix


def perform_estimation_linear_regression(pixels_references, pixels_raws, size, is_grayscale):
    design_matrix = make_design_matrix(pixels_references, pixels_raws, size)

    print('fitting lookup table coefficients')

    result = make_lut_identity_normed(size)
    differences_references_raw = pixels_references - pixels_raws
    rmse_pre_channnels = []
    rmse_past_channels = []
    for idx_channel in range(3):
        rmse_pre_channnels.append(np.sqrt(np.mean(differences_references_raw[..., idx_channel] ** 2)))
        regression = Lasso(alpha=1e-6, fit_intercept=False)
        regression.fit(design_matrix, differences_references_raw[..., idx_channel])
        rmse_past_channels.append(
            np.sqrt(np.mean(
                (
                        differences_references_raw[..., idx_channel]
                        - np.matmul(design_matrix, regression.coef_)) ** 2
            ))
        )
        lut_difference_channel = np.reshape(regression.coef_, [size, size, size])

        if is_grayscale:
            # FIXME: Does not work correctly. Not understood?
            lut_all_channels = result[..., 0] + lut_difference_channel
            result[..., 0] = lut_all_channels
            result[..., 1] = lut_all_channels
            result[..., 2] = lut_all_channels
            break
        else:
            result[..., idx_channel] += lut_difference_channel

    result = np.clip(result, a_min=0., a_max=1.)

    print(f'channels rmse without lut: {rmse_pre_channnels}')
    print(f'channels rmse with fitted lut: {rmse_past_channels}')

    return result

def make_lut_identity_normed(size):
    # identity with [r,g,b, channel]
    result = np.stack(
        np.meshgrid(
            *([
                  np.linspace(0, 1., size)[np.newaxis, ...],
              ] * 3),
            indexing='ij'
        ),
        axis=-1
    )

    return result


def main(dir_images, file_out, level=3, n_pixels_sample=100000, is_grayscale=False, resize=0, path_dt_exec=None,
         path_xmp_image=None, path_xmp_raw=None):
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
        filepaths_images_converted = []

        for path_image, path_raw in pairs_images:
            path_out_image = os.path.join(path_dir_temp, os.path.basename(path_image) + '.png')
            path_out_raw = os.path.join(path_dir_temp, os.path.basename(path_raw) + '.png')
            print(f'converting image {os.path.basename(path_image)}')
            # TODO: make temporary config and memory db work
            with tempfile.TemporaryDirectory() as path_dir_conf_temp:
                with path('darktable_lut_generator.styles', 'image.xmp') as path_xmp:
                    subprocess.call(
                        [
                            'darktable-cli' if path_dt_exec is None else path_dt_exec,
                            path_image,
                            str(path_xmp.resolve()) if path_xmp_image is None else path_xmp_image,
                            path_out_image,
                            '--width',
                            str(resize),
                            '--height',
                            str(resize),
                            '--core',
                            '--configdir',
                            path_dir_conf_temp,
                            '--library',
                            ':memory:'
                        ]
                    )
                print(f'converting raw {os.path.basename(path_raw)}')

                with path('darktable_lut_generator.styles', 'raw.xmp') as path_xmp:
                    subprocess.call(
                        [
                            'darktable-cli' if path_dt_exec is None else path_dt_exec,
                            path_raw,
                            str(path_xmp.resolve()) if path_xmp_raw is None else path_xmp_raw,
                            path_out_raw,
                            '--width',
                            str(resize),
                            '--height',
                            str(resize),
                            '--core',
                            '--configdir',
                            path_dir_conf_temp,
                            '--library',
                            ':memory:'
                        ]
                    )

            filepaths_images_converted.append((path_out_image, path_out_raw))

        print('Finished converting. Generating LUT.')
        # a halc clut is a cube with level**2 entries on each dimension
        result = estimate_lut(filepaths_images_converted, level ** 2, n_pixels_sample, is_grayscale)

        write_cube(result, file_out)

        return result


def write_cube(lut: np.ndarray, path_output):
    size = lut.shape[0] ** 3
    lut_flattened = np.reshape(np.swapaxes(lut, 0, 2), (size, 3))

    s = '{:.10f}'

    with open(path_output, 'w') as f:
        f.write('TITLE "Generated by darktable_lut_creator\n')
        f.write(f'LUT_3D_SIZE {lut.shape[0]}\n')
        f.write('\n')
        for idx in range(lut_flattened.shape[0]):
            f.write(
                f'{s.format(lut_flattened[idx][0])} {s.format(lut_flattened[idx][1])} {s.format(lut_flattened[idx][2])}\n')

def write_hald(lut: np.ndarray, path_output, bit_depth=8):
    raise NotImplementedError('Something with target color space is wrong.')
    if bit_depth not in ([8, 16]):
        raise ValueError(f'Bit depth must be 8 or 16. is: {bit_depth}')
    # size of quadratic haldclut is level**3 and cube size is level**2
    image_size = int(np.power(lut.shape[0], 1.5))

    if bit_depth == 16:
        lut_write = np.round(lut * (2 ** 16 - 1)).astype(np.uint16)
    elif bit_depth == 8:
        lut_write = np.round(lut * (2 ** 8 - 1)).astype(np.uint8)

    result = np.reshape(lut_write, (image_size, image_size, 3))

    cv2.imwrite(
        path_output,
        result
    )
