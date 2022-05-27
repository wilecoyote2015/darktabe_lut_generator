import numpy as np
import cv2
from PIL import ImageFilter, Image
from sklearn.linear_model import Lasso
import logging
from tqdm import tqdm
import tempfile
import os
import subprocess


# TODO: proper 16 bit handling with correspondin export from darktable.
#   stop using pillow. it is buggy and has very limited support for 16 bit handling.
#   load directly via cv2

# TODO: use constrained optimization

def align_images_ecc(im1, im2):
    """Align image 1 to image 2.
    From https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/"""
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

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

    mask_ones = np.full_like(im1[..., 0], 255)

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


def get_aligned_image_pair(path_reference, path_raw):
    reference = Image.open(path_reference)
    raw = Image.open(path_raw)
    # reference = cv2.imread(path_reference, cv2.COLOR_BGR2RGB)
    # raw = cv2.imread(path_raw, cv2.COLOR_BGR2RGB)

    # align the images
    print(f'aligning image {path_reference}')
    reference_aligned, mask = align_images_ecc(
        np.asarray(reference),
        np.asarray(raw)
    )
    print('Finished alignment')

    mask_result = mask == 255

    return reference_aligned, np.asarray(raw), mask_result


def estimate_lut(filepaths_images: [[str, str]], size, n_pixels_sample) -> np.ndarray:
    """
    :param filepaths_images: paths of image pairs: [reference, vanilla raw development]
    :return:
    """
    logging.info('Opening and aligning images')
    pixels_raws = []
    pixels_references = []
    for path_reference, path_raw in tqdm(filepaths_images):
        pixels_reference, pixels_raw = get_pixels_sample_image_pair(
            path_reference,
            path_raw,
            int(n_pixels_sample / len(filepaths_images))
        )
        pixels_raws.append(pixels_raw)
        pixels_references.append(pixels_reference)

    pixels_raws = np.concatenate(pixels_raws, axis=0)
    pixels_references = np.concatenate(pixels_references, axis=0)

    lut_result = perform_estimation_linear_regression(pixels_references, pixels_raws, size)

    return lut_result


def sample_indices_pixels(pixels, n_samples, uniform=False):
    if uniform:
        histogram, edges = np.histogramdd(pixels, density=False,
                                          bins=np.tile(np.linspace(0, 256, 10)[np.newaxis, ...], [3, 1]))

        indices_bins_r = np.digitize(pixels[..., 0], edges[0]) - 1
        indices_bins_g = np.digitize(pixels[..., 1], edges[1]) - 1
        indices_bins_b = np.digitize(pixels[..., 2], edges[2]) - 1

        counts_samples = histogram[indices_bins_r, indices_bins_g, indices_bins_b]
        counts_samples_relative = counts_samples / np.sum(counts_samples)
        # densities_samples_normalized = densities_samples / np.sum(densities_samples)

        weights_samples = 1. / counts_samples_relative
        probabilities_samples = weights_samples / np.sum(weights_samples)
    else:
        probabilities_samples = None

    indices = np.arange(0, pixels.shape[0])
    indices_sampled = np.random.choice(indices, n_samples, p=probabilities_samples)

    return indices_sampled


def get_pixels_sample_image_pair(path_reference, path_raw, n_samples):
    reference, raw, mask = get_aligned_image_pair(path_reference, path_raw)

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
    result_raw = pixels_raw[indices_sample].astype(np.float64)
    result_reference = pixels_reference[indices_sample].astype(np.float64)

    return result_reference, result_raw


def perform_estimation_linear_regression(pixels_references, pixels_raws, size):
    coordinates = np.linspace(0, 255, size)
    step_size = 255. / (size - 1)

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

    print('fitting lookup table coefficients')

    result = make_lut_identity(size)
    differences_references_raw = pixels_references - pixels_raws
    for idx_channel in range(3):
        regression = Lasso(alpha=1e-4, fit_intercept=False)
        regression.fit(design_matrix, differences_references_raw[..., idx_channel])

        lut_difference_channel = np.reshape(regression.coef_, [size, size, size])
        result[..., idx_channel] += lut_difference_channel

    result = np.clip(result, a_min=0., a_max=255)

    lut = get_lut_pillow(result)

    w_h_img_evaluation = int(np.floor(np.sqrt(pixels_references.shape[0])))
    size_img_evaluation = w_h_img_evaluation ** 2

    img_evaluation_raw = Image.fromarray(
        pixels_raws[:size_img_evaluation].reshape([w_h_img_evaluation, w_h_img_evaluation, 3]).astype(np.uint8)
    )
    img_evaluation_reference = Image.fromarray(
        pixels_references[:size_img_evaluation].reshape([w_h_img_evaluation, w_h_img_evaluation, 3]).astype(np.uint8)
    )

    transformed = img_evaluation_raw.filter(lut)

    # img_evaluation_reference.show()
    # transformed.show()

    rmse_pre = np.sqrt(np.mean((np.asarray(img_evaluation_raw, dtype=np.float32) - np.asarray(img_evaluation_reference,
                                                                                              dtype=np.float32)) ** 2))
    rmse_past = np.sqrt(np.mean(
        (np.asarray(transformed, dtype=np.float32) - np.asarray(img_evaluation_reference, dtype=np.float32)) ** 2))

    print(f'rmse without lut: {rmse_pre}')
    print(f'rmse with fitted lut: {rmse_past}')

    return result


def get_lut_pillow(
        # [r,g,b,channel(RGB)]
        lut_np
):
    lut_bgr = np.swapaxes(lut_np, 0, 2)
    return ImageFilter.Color3DLUT(size=lut_bgr.shape[0], table=lut_bgr.flatten() / 255.)


def make_lut_identity(size):
    # identity with [r,g,b, channel]
    result = np.stack(
        np.meshgrid(
            *([
                  np.linspace(0, 255, size)[np.newaxis, ...],
              ] * 3),
            indexing='ij'
        ),
        axis=-1
    )

    return result


def run_process(args):
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip())


def main(dir_images, file_out, level=3, n_pixels_sample=1000):
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

        # command_base = 'darktable-cli {path_in} {path_xmp} {path_out} --core --configdir ://temp --library :memory:'

        filepaths_images_converted = []

        for path_image, path_raw in pairs_images:
            path_out_image = os.path.join(path_dir_temp, os.path.basename(path_image) + '.png')
            path_out_raw = os.path.join(path_dir_temp, os.path.basename(path_raw) + '.png')
            print(f'converting image {os.path.basename(path_image)}')
            run_process(
                [
                    'darktable-cli',
                    path_image,
                    'styles/image.xmp',
                    path_out_image,
                    # '--core',
                    # '--configdir \'://temp\'',
                    # '--library \':memory:\''
                ]
            )
            print(f'converting raw {os.path.basename(path_raw)}')

            run_process(
                [
                    'darktable-cli',
                    path_raw,
                    'styles/raw.xmp',
                    path_out_raw,
                ]
            )

            filepaths_images_converted.append((path_out_image, path_out_raw))

        print('Finished converting. Generating LUT.')
        # a halc clut is a cube with level**2 entries on each dimension
        result = estimate_lut(filepaths_images_converted, level ** 2, n_pixels_sample)

        write_hald(result, file_out, bit_depth=16)

        return result


def write_hald(lut: np.ndarray, path_output, bit_depth=8):
    if bit_depth not in ([8, 16]):
        raise ValueError(f'Bit depth must be 8 or 16. is: {bit_depth}')
    # size of quadratic haldclut is level**3 and cube size is level**2
    image_size = int(np.power(lut.shape[0], 1.5))

    # lut is float with 0-255. convert to uint16 for saving in 16 bit

    if bit_depth == 16:
        lut = (lut * (2 ** 16 / 2 ** 8)).astype(np.uint16)

    # result = np.ndarray(
    #     (image_size, image_size, 3),
    #     np.uint16
    # )

    result = np.reshape(lut, (image_size, image_size, 3))

    cv2.imwrite(
        path_output,
        # cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        result
    )

    # result_pil = Image.fromarray(result, mode='I;16')
    # result_pil.save(path_output)


if __name__ == '__main__':
    paths = [
        (
            '/home/bjoern/Pictures/2022_05_19_hochzeit/fuji/darktable_exported_lut/reference.png',
            '/home/bjoern/Pictures/2022_05_19_hochzeit/fuji/darktable_exported_lut/raw.png',
        )
    ]

    path_samples = '/home/bjoern/PycharmProjects/darktabe_hald_generator/samples'

    file_out = '/home/bjoern/Pictures/hald-clut/HaldCLUT/test.png'

    identity = make_lut_identity(16)
    # write_hald(identity, file_out)

    lut = main(path_samples, file_out, level=4, n_pixels_sample=100000)

    # lut = estimate_lut(paths, size=9, n_pixels_sample=100000)

    write_hald(lut, 'hald.png', 16)
file_out
