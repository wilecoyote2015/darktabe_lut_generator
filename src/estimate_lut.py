import numpy as np
import cv2
from PIL import ImageFilter, Image
from scipy import optimize
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


def apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """ Apply LUT transformation

    :param image:
    :param lut: 4 dimensions: first three are raw r,g,b coordinates.
        last has 3 entries for r,g,b target coordinates
    :return:
    """

    raise NotImplementedError


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
        reference = cv2.imread(path_reference, cv2.COLOR_BGR2RGB)
        raw = cv2.imread(path_raw, cv2.COLOR_BGR2RGB)

        # align the images
        reference_aligned, mask = align_images(reference, raw)

        # blur the images a bit to reduce effects of spatial processing of in camera image (sharpening etc.)
        # TODO: how are nans in borders handled?
        size_blur = (5, 5)
        reference_aligned_blurred = cv2.blur(reference_aligned, size_blur)
        raw_blurred = cv2.blur(raw, size_blur)
        mask_blurred = cv2.blur(mask, size_blur)

        references.append(Image.fromarray(reference_aligned_blurred))
        raws.append(Image.fromarray(raw_blurred))
        masks.append(mask_blurred == 255)

    # lut estimation
    lut_start = ImageFilter.Color3DLUT.generate(
        size,
        lambda r, g, b: (r, g, b)
    )

    params_lut = lut_start.table

    def loss(params_lut_):
        print('generating lut')
        lut = ImageFilter.Color3DLUT(size, params_lut_)

        # score over all images
        loss = 0.
        for reference, raw, mask in zip(references, raws, masks):
            print('transforming image')
            raw_transformed = raw.filter(lut)
            loss += np.mean((np.asarray(raw_transformed)[mask] - np.asarray(reference)[mask]) ** 2)
            print('transformed')

        print(f'loss: {loss}')

        return loss

    result_optimization = optimize.minimize(
        loss,
        params_lut,
        method='SLSQP',
        bounds=[(0., 1.) for i in range(len(params_lut))],
        callback=lambda xk: print(xk),
    )

    params_lut_optimized = result_optimization.x

    lut_result = ImageFilter.Color3DLUT(size, params_lut_optimized)

    x = 1

    return params_lut_optimized


if __name__ == '__main__':
    paths = [
        (
            '/home/bjoern/Pictures/2022_05_19_hochzeit/fuji/darktable_exported_lut/reference.png',
            '/home/bjoern/Pictures/2022_05_19_hochzeit/fuji/darktable_exported_lut/raw.png',
        )
    ]

    estimate_lut(paths)
