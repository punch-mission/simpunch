# ruff: noqa
from pathlib import Path

import astropy.time
import numpy as np
from matplotlib import pyplot as plt
from punchbowl.data.wcs import calculate_celestial_wcs_from_helio
from regularizepsf import (ArrayPSFTransform, simple_functional_psf,
                           varied_functional_psf)
from regularizepsf.util import calculate_covering

from create_projected_psf import gen_projected_psf_from_image
from simpunch.level1 import generate_spacecraft_wcs, generate_starfield

psf_size = 256  # size of the PSF model to use in pixels
initial_sigma = 2.5 / 2.355
target_sigma = 3.3 / 2.355
img_size = 2048

@simple_functional_psf
def baked_in_initial_psf(row,
                         col,
                         x0=psf_size / 2,
                         y0=psf_size / 2,
                         sigma_x=initial_sigma,
                         sigma_y=initial_sigma,
                         A=0.1):
    return A * np.exp(-(np.square(row - x0) / (2 * np.square(sigma_x)) + np.square(col - y0) / (2 * np.square(sigma_y))))


@simple_functional_psf
def target_psf(row,
                col,
                core_sigma_x=target_sigma,
                core_sigma_y=target_sigma,
                tail_angle=0,
                tail_separation=0,
                ):
    x0 = psf_size / 2
    y0 = psf_size / 2
    A = 0.08
    core = A * np.exp(
        -(np.square(row - x0) / (2 * np.square(core_sigma_x)) + np.square(col - y0) / (2 * np.square(core_sigma_y))))

    A_tail = 0.05
    sigma_x = tail_separation
    sigma_y = tail_separation # core_sigma_y + 0.25
    a = np.square(np.cos(tail_angle)) / (2 * np.square(sigma_x)) + np.square(np.sin(tail_angle)) / (
                2 * np.square(sigma_y))
    b = -np.sin(tail_angle) * np.cos(tail_angle) / (2 * np.square(sigma_x)) + (
                (np.sin(tail_angle) * np.cos(tail_angle)) / (2 * np.square(sigma_y)))
    c = np.square(np.sin(tail_angle)) / (2 * np.square(sigma_x)) + np.square(np.cos(tail_angle)) / (
                2 * np.square(sigma_y))
    tail_x0 = x0 - tail_separation * np.cos(tail_angle)
    tail_y0 = y0 + tail_separation * np.sin(tail_angle)
    tail = A_tail * np.exp(-(a * (row - tail_x0) ** 2 + 2 * b * (row - tail_x0) * (col - tail_y0) + c * (col - tail_y0) ** 2))
    return core + tail


@varied_functional_psf(target_psf)
def synthetic_psf(row, col):
    return {"tail_angle": -np.arctan2(row - img_size//2, col - img_size//2),
            "tail_separation": np.sqrt((row - img_size//2) ** 2 + (col - img_size//2) ** 2)/1200 * 1.5 + 1E-3,
            "core_sigma_x": target_sigma,
            "core_sigma_y": target_sigma}

coords = calculate_covering((img_size, img_size), psf_size)
initial = baked_in_initial_psf.as_array_psf(coords, psf_size)
initial.visualize_psfs(all_patches=False)
plt.show()

synthetic = synthetic_psf.as_array_psf(coords, psf_size)
synthetic.visualize_psfs(all_patches=False)
plt.show()

wcs_helio = generate_spacecraft_wcs("1", 0)
wcs_stellar_input = calculate_celestial_wcs_from_helio(wcs_helio,
                                                       astropy.time.Time.now(),
                                                       (img_size, img_size))

backward_corrector = ArrayPSFTransform.construct(initial, synthetic, alpha=0.5, epsilon=1E-3) #alpha=3.7, epsilon=0.15)
backward_corrector.save(Path("synthetic_backward_psf.fits"))

corrected_psf = gen_projected_psf_from_image(
        wcs_stellar_input, psf_width=psf_size, star_gaussian_sigma=initial_sigma)
forward_corrector = ArrayPSFTransform.construct(synthetic, corrected_psf, alpha=0.7, epsilon=0.515)
forward_corrector.save(Path("synthetic_forward_psf.fits"))
