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
import sys

from darktable_lut_generator.estimate_lut import main as main_


def main():
    parser = argparse.ArgumentParser(
        # description='Generate .cube 3D LUT from jpg/raw sample pairs',
        usage='This package estimates a .cube 3D lookup table for use with the Darktable lut 3D module. \n'
              'A direktory with image pairs of one RAW image and the corresponding OOC image (e.g. JPEG) is used '
              'as input. \n'
              'The images should represent a wide variety of colors; ideally, the whole Adobe RGB color space is covered.\n'
              'The resulting LUT is intended for application in Adobe RGB color space. Hence, it is advisable to shoot the images in'
              'Adobe RGB.'
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
             'The default value works well. Try 10000 if running out of memory. Values over 500000 usually provide no '
             'significant benefit, but this depends on the images and the lut size'
             'Set to 0 to use all pixels (recommended with resize)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=9,
        help='Resulting cube resolution per dimension. '
             'Keep in mind that for high sizes, much sample data covering many colors is needed for good generalization '
             'performance.'
    )
    parser.add_argument(
        '--resize',
        type=int,
        default=1000,
        help='If provided, the input images are resized to this maximum border length. If 0, images are not resized, which'
             ' may result in long alignment runtimes.'
    )
    parser.add_argument(
        '--is_grayscale',
        action='store_true',
        help='Provide this flag if the image style is grayscale. Ensures that the resulting'
             ' lookup table contains only grayscale values.'
    )
    parser.set_defaults(is_grayscale=False)
    parser.add_argument(
        '--sample_uniform',
        action='store_true',
        help='Try to sample the pixels uniformly over the color space. This may help if particular colors are represented'
             ' by only small regions in the sample images.'
    )
    parser.set_defaults(sample_uniform=False)

    parser.add_argument(
        '--use_lens_correction',
        action='store_true',
        help='Use auto-applied lens correction module for the RAW image. Only effective without --path_style_raw.'
             ' Note that lens correction is a bit tricky as it can change the exposure, so that the resulting LUT may only yield good results'
             'for images with the same lens and lens correction applied. It should be preferred to not use lens correctio and'
             ' also disable lens correction in camera. Then, alignment can usually also be disabled with --disable_image_alignment.'
             ' This setting is mainly intended for use with cameras that do not allow'
             ' disabling in-camera lens correction for the OOC JPEGs.'
    )
    parser.set_defaults(use_lens_correction=False)
    parser.add_argument(
        '--disable_image_alignment',
        action='store_true',
        help='This flag disables the image alignment and just crops the raw image.'
             'Often, developed raws and OOC images do not overlap'
             ' perfectly. One may assume that the developed Raw has the same amount of additional'
             '  pixels on each side and is otherwise geometrically identical to the OOC image.'
             'Then, the developed raw can simply be cropped accordingly. '
             'The assumption does not hold in many real-world cases, though. In particular, in-camera lens correction'
             ' may distort the image.'
             ' A simple image alignment procedure is used'
             ' to align the images and compensate for some distortions by default. '
             'Alignment is tricky, especially as OOC and RAW images usually exhibit different gradiation. '
             'Pixel-Level alignment precision is necessary for good LUT estimation results and this is '
             'not necessarily provided with alignment. Hence, it is important to check the alignment results.'
             'Use the --path_dir_out_info to inspect'
             ' the generated images and assess whether alignment is necessary and if it works.'
             'Generally, the best results are achieved by disabling in camera lens correction and hence . Then,'
             ' image alignment can also be disabled in most cases.'
    )
    parser.set_defaults(disable_image_alignment=False)
    parser.add_argument(
        '--align_translation_only',
        action='store_true',
        help='Use translation instead of affine transform for alignment..'
    )
    parser.set_defaults(align_translation_only=False)
    parser.add_argument(
        '--legacy_color',
        action='store_true',
        help='Use legacy color adaption for raw development'
    )
    parser.set_defaults(legacy_color=False)
    parser.add_argument(
        '--path_dt_cli',
        type=str,
        default=None,
        help='Path to the darktable-cli executable if it is not in PATH.'
    )
    parser.add_argument(
        '--path_style_raw',
        type=str,
        default=None,
        help='Path to an optional .dtstyle file for processing the raw images of the input image pairs. '
             'Use this, for instance, to use a different color space or a different exposure so that the resulting LUT '
             'will yield the correct result on a raw with the corresponding modules applied. '
             'A practical example might be to shoot the sample images in a controlled environment and apply the color'
             ' calibration module with a color checker on all sample images in order to ensure proper input color space '
             'transformation.'
    )
    parser.add_argument(
        '--path_style_image',
        type=str,
        default=None,
        help='Path to an optional .dtstyle file for processing the out of camera / processed images of the input image pairs. '
             'This can be used to use different color spaces, but no further changes should be made to the image.'
    )
    parser.add_argument(
        '--path_dir_intermediate',
        type=str,
        default=None,
        help='Path to directory where intermediate converted images are stored..'
    )
    parser.add_argument(
        '--path_dir_out_info',
        type=str,
        default=None,
        help='Path to directory to output additional information / plots'
    )
    parser.add_argument(
        '--make_interpolated_estimates_red',
        action='store_true',
        help='In the resulting LUT, make estimates of colors that were interpolated due to unreliably few datapoints red. '
             'Only applies if --no_interpolation_unreliable is not set. Useful for debugging and identifying sparsely sampled colors.'
    )
    parser.set_defaults(make_interpolated_estimates_red=False)
    parser.add_argument(
        '--make_unchanged_red',
        action='store_true',
        help='In the resulting LUT, make colors that are estimated as unchanged w.r.t. an identity LUT red. Useful for debugging and identifying sparsely sampled colors.'
    )
    parser.set_defaults(make_unchanged_red=False)
    parser.add_argument(
        '--no_interpolation_unreliable',
        action='store_true',
        help='By default, estimates for colors with unreliably few samples are interpolated. This flag disables the interpolation, which may lead to wrong colors that are not covered well by the sample images..'
    )
    parser.set_defaults(no_interpolation_unreliable=False)
    parser.add_argument(
        '--interpolate_only_missing_data',
        action='store_true',
        help='By default, estimates for colors with unreliably few samples are interpolated. '
             'If this flag is active and --no_interpolation_unreliable is NOT set, only colors with '
             'no samples are considered unreliable. '
             'TODO: Do some statistical inference to determine reliability of estimated parameters for more '
             'sophisticated decision which colors to interpolate. But note that constrained optimization is used, '
             'so that the statistical assumptions for OLS standard errors do not apply. In the one hand, '
             'providing a statistically attractive measure for reliability may not be as trivial as it seems '
             'intuitively. In the other hand, a simple approach might work well enough in practice. '
             'If you like to contribute, you are welcome!'
    )
    parser.set_defaults(interpolate_only_missing_data=False)
    parser.add_argument(
        '--single_pass',
        action='store_true',
        help='By default, two passes of LUT estimation are performed if --disable_image_alignment is not set: '
             'First, a rough estimate ot LUT is calculated without alignment. Then, this LUT is used to transform the '
             'RAW image\'s colors for better alignment of the final pass. This is motivated by the problem that'
             ' the different color rendition of RAW and OOC images make proper alignment difficult.'
    )
    parser.set_defaults(single_pass=False)

    args = parser.parse_args()

    main_(
        args.dir_images,
        args.file_lut_output,
        args.size,
        args.n_samples if args.n_samples > 0 else None,
        args.is_grayscale,
        args.resize,
        args.path_dt_cli,
        args.path_style_image,
        args.path_style_raw,
        args.path_dir_intermediate,
        args.path_dir_out_info,
        args.make_interpolated_estimates_red,
        args.make_unchanged_red,
        not args.no_interpolation_unreliable,
        args.use_lens_correction,
        args.legacy_color,
        not args.disable_image_alignment,
        args.align_translation_only,
        args.sample_uniform,
        args.interpolate_only_missing_data,
        not args.single_pass
    )


if __name__ == '__main__':
    sys.exit(main())
