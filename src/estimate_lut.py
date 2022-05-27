import numpy as np
import cv2
from PIL import ImageFilter, Image
from scipy import optimize, interpolate, stats
from sklearn.linear_model import LinearRegression
import logging
from tqdm import tqdm


def align_images(im1, im2):
    """From https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    Align image 1 to image 2
    """
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    mask_ones = np.full_like(im1[..., 0], 255)
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    mask = cv2.warpPerspective(mask_ones, h, (width, height))

    return im1Reg, mask


def apply_lut(images: [Image], lut: np.ndarray) -> [np.ndarray]:
    """ Apply LUT transformation

    :param images:  [[width, height, channel ]]
    :param lut: 4 dimensions: first three are raw r,g,b coordinates.
        last has 3 entries for r,g,b target coordinates
    :return:
    """
    r = np.linspace(0, 255, lut.shape[0])
    g = np.linspace(0, 255, lut.shape[1])
    b = np.linspace(0, 255, lut.shape[2])

    print('Making interpolator')
    interpolator = interpolate.RegularGridInterpolator(
        (r, g, b),
        lut
    )

    images_transformed = []
    for image in images:
        print('transforming image')
        image_array = np.asarray(image)
        # pixels = np.reshape(image_array, (image_array.shape[0] * image_array.shape[1], image_array.shape[-1]))
        pixels_transformed = interpolator(image_array)

        images_transformed.append(
            pixels_transformed
        )
        print('transformed image')

    return images_transformed

    # raise NotImplementedError


def estimate_lut(filepaths_images: [[str, str]], size=8) -> np.ndarray:
    """

    :param filepaths_images: paths of image pairs: [reference, vanilla raw development]
    :return:
    """
    references = []
    raws = []
    masks = []

    # open the images
    logging.info('Opening and aligning images')
    for path_reference, path_raw in tqdm(filepaths_images):
        reference = Image.open(path_reference)
        raw = Image.open(path_raw)
        # reference = cv2.imread(path_reference, cv2.COLOR_BGR2RGB)
        # raw = cv2.imread(path_raw, cv2.COLOR_BGR2RGB)

        # align the images
        print(f'aligning image {path_reference}')
        reference_aligned, mask = align_images(
            np.asarray(reference),
            np.asarray(raw)
        )
        print('Finished alignment')

        # blur the images a bit to reduce effects of spatial processing of in camera image (sharpening etc.)
        # TODO: how are nans in borders handled?
        size_blur = (5, 5)
        reference_aligned_blurred = cv2.blur(reference_aligned, size_blur)
        raw_blurred = cv2.blur(np.asarray(raw), size_blur)
        mask_blurred = cv2.blur(mask, size_blur)

        references.append(Image.fromarray(reference_aligned_blurred, mode='RGB'))
        raws.append(Image.fromarray(raw_blurred, mode='RGB'))
        masks.append(mask_blurred == 255)

    # references[0].show()
    # lut estimation
    # lut_start = ImageFilter.Color3DLUT.generate(
    #     size,
    #     lambda r, g, b: (r, g, b)
    # )

    perform_estimation_linear_regression(references, raws, masks, size)

    return params_lut_optimized


def perfom_estimation_local_mean(references: [Image], raws: [Image], masks: [np.ndarray], size):
    coordinates = np.linspace(0, 255, size)
    centers = np.stack(np.meshgrid(coordinates, coordinates, coordinates, indexing='ij'), axis=-1)

    references_arrays = [np.asarray(img) for img in references]
    raws_arrays = [np.asarray(img) for img in raws]

    max_distance = 255. / (size - 1)

    result = np.zeros(
        (size, size, size, 3),
        np.float
    )

    luts_unnormalized_images = [
        np.zeros(
            (size, size, size, 3),
            np.float
        ) for image in raws
    ]
    sums_weights = np.zeros(
        (size, size, size),
        np.float
    )
    for idx_image, (raw, reference, mask) in enumerate(zip(raws_arrays, references_arrays, masks)):
        invalid_region = np.logical_not(mask)
        for idx_r in tqdm(range(size)):
            r = coordinates[idx_r]
            diff_r_squared = (raw[..., 0] - r) ** 2
            for idx_g in range(size):
                g = coordinates[idx_g]
                diff_g_squared = (raw[..., 1] - g) ** 2
                for idx_b in range(size):
                    b = coordinates[idx_b]
                    diff_b_squared = (raw[..., 2] - b) ** 2

                    distances = np.sqrt(diff_g_squared + diff_b_squared + diff_r_squared)
                    weights = np.maximum(1. - (distances / max_distance), 0.)
                    weights[invalid_region] = 0.

                    luts_unnormalized_images[idx_image][idx_r, idx_g, idx_b] += np.sum(
                        reference * weights[..., np.newaxis],
                        axis=(0, 1)
                    )
                    sums_weights[idx_r, idx_g, idx_b] += np.sum(weights)

    return result


def sample_indices_pixels(pixels, n_samples=100000, uniform=False):
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


def perform_estimation_linear_regression(references: [Image], raws: [Image], masks: [np.ndarray], size):
    references_arrays = [np.asarray(img) for img in references]
    raws_arrays = [np.asarray(img) for img in raws]

    coordinates = np.linspace(0, 255, size)

    step_size = 255. / (size - 1)

    # TODO: reshaping correct?
    pixels_references = np.concatenate([
        np.reshape(img, (img.shape[0] * img.shape[1], img.shape[-1]))[np.reshape(mask, mask.shape[0] * mask.shape[1])]
        for img, mask in zip(references_arrays, masks)
    ],
        axis=0
    )
    pixels_raws = np.concatenate([
        np.reshape(img, (img.shape[0] * img.shape[1], img.shape[-1]))[np.reshape(mask, mask.shape[0] * mask.shape[1])]
        for img, mask in zip(raws_arrays, masks)
    ],
        axis=0
    )

    print('sampling pixels')
    indices_sample = sample_indices_pixels(pixels_raws)
    pixels_raws = pixels_raws[indices_sample]
    pixels_references = pixels_references[indices_sample]

    # feature matrix with order of permutation: r, g, b
    print('generating design matrix')
    design_matrix = np.zeros((pixels_references.shape[0], size * size * size), np.float32)

    # [pixel, channel, channel_value]
    differences_channels = pixels_raws[..., np.newaxis] - np.stack([coordinates] * 3, axis=0)[np.newaxis, ...]
    differences_channels_relative_grid_steps = differences_channels / step_size
    weights_distances_channels = np.maximum(1. - np.abs(differences_channels_relative_grid_steps), 0.)
    # differences_colors_squared = differences_colors**2

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
    result = np.zeros(
        (size, size, size, 3),
        np.float
    )
    for idx_channel in range(3):
        regression = LinearRegression(fit_intercept=False, positive=True)
        regression.fit(design_matrix, pixels_references[..., idx_channel])

        lut_channel = np.reshape(regression.coef_, [size, size, size])
        lut_channel = np.swapaxes(lut_channel, 0, 2)
        result[..., idx_channel] = lut_channel

    result = np.clip(result, a_min=0., a_max=255)

    lut = ImageFilter.Color3DLUT(size=size, table=result.flatten() / 255.)

    filtered = raws[0].filter(lut)
    # filtered_np = apply_lut(raws, result)[0]

    references[0].show()
    filtered.show()

    rmse_pre = np.sqrt(np.mean((np.asarray(raws[0]) - np.asarray(references[0])) ** 2))
    rmse_past = np.sqrt(np.mean((filtered - np.asarray(references[0])) ** 2))
    # rmse_past_np = np.sqrt(np.mean((filtered_np - np.asarray(references[0]))**2))

    print(f'rmse pre: {rmse_pre}')
    print(f'rmse past: {rmse_past}')
    # print(f'rmse past np: {rmse_past_np}')

    pass


if __name__ == '__main__':
    paths = [
        (
            '/home/bjoern/Pictures/2022_05_19_hochzeit/fuji/darktable_exported_lut/reference.png',
            '/home/bjoern/Pictures/2022_05_19_hochzeit/fuji/darktable_exported_lut/raw.png',
        )
    ]

    estimate_lut(paths, size=8)
