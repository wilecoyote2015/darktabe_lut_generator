import cv2
import numpy as np

max_ = 2 ** 8 - 1

width = 1800
height = 1200
step_constant = 15

n_luma_bands = 10

n_px_segment = int(width / 6)

ramp = np.linspace(0, max_, n_px_segment)

ramp_r = np.concatenate(
    [
        np.full((n_px_segment,), max_, dtype=float),
        np.flip(ramp),
        np.full((n_px_segment * 2,), 0, dtype=float),
        ramp,
        np.full((n_px_segment,), max_, dtype=float)
    ],
    axis=0
)

ramp_g = np.roll(ramp_r, n_px_segment * 2)
ramp_b = np.roll(ramp_g, n_px_segment * 2)
band = np.stack([ramp_r, ramp_g, ramp_b], axis=1)

result = np.zeros((height, width, 3))

luma_band = np.zeros(
    (int(height / n_luma_bands), width, 3)
)
n_steps_saturation = int(luma_band.shape[0] / step_constant)

for idx_step_saturation in range(n_steps_saturation):
    saturation = 1. - (idx_step_saturation / (n_steps_saturation - 1))
    band_saturated = (band - max_) * saturation + max_

    idx_band = step_constant * idx_step_saturation

    luma_band[idx_band:idx_band + step_constant] = np.tile(band_saturated[np.newaxis, ...], (step_constant, 1, 1))

idx_y = 0
for idx_luma_band in range(n_luma_bands):
    brightness_factor = 1. - (idx_luma_band / (n_luma_bands - 1))
    idx_start = idx_luma_band * luma_band.shape[0]
    result[idx_start: idx_start + luma_band.shape[0]] = luma_band * brightness_factor

cv2.imwrite('test.png', result.astype(np.uint8))

pass
