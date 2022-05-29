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

import argparse
from darktable_lut_generator.estimate_lut import main

parser = argparse.ArgumentParser(
    # description='Generate .cube 3D LUT from jpg/raw sample pairs',
    usage='This package estimates a .cube 3D lookup table for use with the Darktable lut 3D module. \n'
          'A direktory with image pairs of one RAW image and the corresponding OOC image (e.g. JPEG) is used '
          'as input. \n'
          'The images should represent a wide variety of colors; ideally, the whole Adobe RGB color space is covered.\n'
          'The resulting LUT is intended for application in Adobe RGB color space. Hence, it is advisable\n'
          '\n'
          'Estimation is performed by estimating the differences to an identity LUT '
          'using linear regression with LASSO regularization, assuming trilinear interpolation '
          'when applying the LUT. \n'
          'Very sparsely or non-sampled colors will fallback to identity. However, no sophisticated hyperparameter tuning'
          ' regarding the LASSO parameter has been conducted, especially regarding different cube size. \n'
          'n_samples pixels are sampled from the image, as using all pixels is computationally expensive. \n'
          'Sampling is performed weighted by the inverse estimated sample density conditioned on the raw pixel colors '
          'in order to obtain a sample with approximately uniform distribution over the represented colors. \n'
          'This reduces the needed sample count for good results by approx. an order of magnitude compared to drawing '
          'pixels uniformly.'
)
parser.add_argument(
    'dir_images',
    type=str,
    help='Directory with input image pairs. In the directory, for each raw image, exactly one (out of camera) image '
         'must be present. The images of one pair must have the same base name, but different extension.'
)
parser.add_argument(
    'file_lut_output',
    type=str,
    help='Desired filepath to store output 3D .cube LUT (with extension).'
)
parser.add_argument(
    '--n_samples',
    type=int,
    default=100000,
    help='Number of pixels to sample from the images for LUT estimation. '
         'Higher values may produce more accurate results, but are slower and more memory intensive. '
         'The default value works well. Try 10000 if running out of memory. '
         'Set to 0 to use all pixels (recommended with resize)'
)
parser.add_argument(
    '--level',
    type=int,
    default=3,
    help='Level (as defined by HaldClut) of output LUT. Resulting cube resolution per dimension is level^2. '
         'Keep in mind that for high levels, much sample data covering many colors is needed for good generalization '
         'performance.'
)
parser.add_argument(
    '--resize',
    type=int,
    default=0,
    help='If provided, the input images are resized to this maximum border length.'
)
parser.add_argument(
    '--is_grayscale',
    action='store_true',
    help='Provide this flag if the image style is grayscale. Ensures that the resulting'
         ' lookup table contains only grayscale values.'
)
parser.set_defaults(is_grayscale=False)
parser.add_argument(
    '--path_dt_cli',
    type=str,
    default=None,
    help='Path to the darktable-cli executable if it is not in PATH.'
)
parser.add_argument(
    '--path_xmp_raw',
    type=str,
    default=None,
    help='Path to an optional xmp file for processing the raw images of the input image pairs. '
         'Use this, for instance, to use a different color space or a different exposure so that the resulting LUT '
         'will yield the correct result on a raw with the corresponding modules applied. '
         'A practical example might be to shoot the sample images in a controlled environment and apply the color'
         ' calibration module with a color checker on all sample images in order to ensure proper input color space '
         'transformation.'
)
parser.add_argument(
    '--path_xmp_image',
    type=str,
    default=None,
    help='Path to an optional xmp file for processing the out of camera / processed images of the input image pairs. '
         'This can be used to use different color spaces, but no further changes should be made to the image.'
)

args = parser.parse_args()

main(
    args.dir_images,
    args.file_lut_output,
    args.level,
    args.n_samples if args.n_samples > 0 else None,
    args.is_grayscale,
    args.resize,
    args.path_dt_cli,
    args.path_xmp_image,
    args.path_xmp_raw,
)
