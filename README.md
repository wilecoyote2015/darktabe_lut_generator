This package estimates a .cube 3D lookup table (LUT) for use with the Darktable lut 3D module.
It was designed to obtain 3D LUTs replicating in-camera jpeg styles.
This is especially if one shoots large sets of RAW photos (e.g. for commission), where most shall simple
resemble the standard out-of-camera (OOC) style when exported by darktable, while still being able to do some quick
corrections on selected images while mainraining the OOC style.

Below is an example using an LUT estimated to match the Provia film simulation on a Fujifilm X-T3.
First is the OOC Jpeg, second is the RAW processed in Darktable with the LUT and third is the RAW processed in Darktable
without any corrections:

![Jpeg](https://raw.githubusercontent.com/wilecoyote2015/darktabe_lut_generator/master/images_readme/jpeg.jpg?raw=true "Jpeg")
![Raw with LUT](https://raw.githubusercontent.com/wilecoyote2015/darktabe_lut_generator/master/images_readme/provia.jpg?raw=true "Raw with LUT")
![Raw](https://raw.githubusercontent.com/wilecoyote2015/darktabe_lut_generator/master/images_readme/raw.jpg?raw=true "Raw")

# Installation

Python 3 must be installed.
Installation of Darktable LUT Generator via pip:
```pip install darktable_lut_generator```

# Usage

Run:
```darktable_lut_generator [path to directory with images] [output .cube file]```
For help and further arguments, run
```darktable_lut_generator --help```

A direktory with image pairs of one RAW image and the corresponding OOC image (e.g. jpeg) is used as input.
The images should represent a wide variety of colors; ideally, the whole Adobe RGB color space is covered.
The resulting LUT is intended for application in Adobe RGB color space.
Hence, it is advisable to also shoot the in-camera jpegs in Adobe RGB in order to cover the whole available gamut.
In default configuration, Darktable may apply an exposure module with camera exposure bias correction automatically
to raw files. The LUTs produced by this module are constructed to resemble the OOC jpeg when used on a raw
image *without* the exposure bias correction. Also, the *filmic rgb* module should be turned off.

# Estimation

Estimation is performed by estimating the differences to an identity LUT using linear regression with LASSO
regularization, assuming trilinear interpolation when applying the LUT.
Very sparsely or non-sampled colors will fallback to identity. However, no sophisticated hyperparameter tuning regarding
the LASSO parameter has been conducted, especially regarding different cube size.
`n_samples` pixels are sampled from the image, as using all pixels is computationally expensive.
Sampling is performed weighted by the inverse estimated sample density conditioned on the raw pixel colors in order to
obtain a sample with approximately uniform distribution over the represented colors.
This reduces the needed sample count for good results by approx. an order of magnitude compared to drawing pixels
uniformly.


