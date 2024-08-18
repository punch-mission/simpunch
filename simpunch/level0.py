"""
Generates synthetic level 0 data
"""
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from ndcube import NDCube
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            write_ndcube_to_fits)
from punchbowl.data.wcs import calculate_pc_matrix, extract_crota_from_wcs
from regularizepsf import ArrayCorrector
from tqdm import tqdm

from simpunch.util import update_spacecraft_location


def certainty_estimate(input_data):
    return input_data


def photometric_uncalibration(input_data,
                              gain: float = 4.3,
                              bitrate_signal: int = 16):

    # Input data should be in calibrated units of MSB
    # First establish a conversion rate backwards to photons

    msb_def = 2.0090000E7 * u.W / u.m**2 / u.sr

    # Constants
    wavelength = 530 * u.nm  # Wavelength in nanometers
    exposure_time = 49 * u.s  # Exposure in seconds

    # Calculate energy of a single photon
    energy_per_photon = (const.h * const.c / wavelength).to(u.J) / u.photon

    # Get the pixel scaling
    pixel_scale = (proj_plane_pixel_area(input_data.wcs) * u.deg**2).to(u.sr) / u.pixel
    # pixel_scale = (abs(input_data.wcs.pixel_scale_matrix[0, 0]) * u.deg**2).to(u.sr) / u.pixel

    # The instrument aperture is 22mm, however the effective area is a bit smaller due to losses and quantum efficiency
    aperture = 49.57 * u.mm**2

    # Calculate the photon count per pixel
    photons_per_pixel = (msb_def / energy_per_photon * aperture * pixel_scale * exposure_time).decompose()

    # Convert the input data to a photon count
    photon_data = input_data.data * photons_per_pixel.value

    # Now using the gain convert to DN
    dn_data = photon_data / gain

    # TODO - Perhaps the stars at night are a bit too bright. Check the scaling earlier in the pipeline.
    # Clip / scale the data at the signal bitrate
    dn_data = np.clip(dn_data, 0, 2**bitrate_signal-1)

    dn_data = dn_data.astype(int)

    input_data.data[:, :] = dn_data

    return input_data, photon_data


def spiking(input_data, spike_scaling=2**16-5_000):
    spike_index = np.random.choice(input_data.data.shape[0] * input_data.data.shape[1],
                                   np.random.randint(40*49-1000, 40*49+1000))
    spike_index2d = np.unravel_index(spike_index, input_data.data.shape)

    spike_values = np.random.normal(spike_scaling,
                                    input_data.data.max() * 0.1,
                                    len(spike_index))
    spike_values = np.clip(spike_values, 0, spike_scaling)

    input_data.data[spike_index2d] = spike_values
    return input_data


def streak_correction_matrix(
    n: int, exposure_time: float, readout_line_time: float, reset_line_time: float,
) -> np.ndarray:

    lower = np.tril(np.ones((n, n)) * readout_line_time, -1)
    upper = np.triu(np.ones((n, n)) * reset_line_time, 1)
    diagonal = np.diagflat(np.ones(n) * exposure_time)

    return lower + upper + diagonal


def streaking(input_data,
              exposure_time: float = 49 * 1000,
              readout_line_time: float = 163/2148,
              reset_line_time: float = 163/2148):

    streak_matrix = streak_correction_matrix(input_data.data.shape[0],
                                             exposure_time, readout_line_time, reset_line_time)

    input_data.data[:, :] = streak_matrix @ input_data.data[:, :] / exposure_time

    return input_data


def uncorrect_vignetting_lff(input_data):
    if input_data.meta['OBSCODE'].value == 4:
        width, height = 2048, 2048
        sigma_x, sigma_y = width / 1, height / 1
        x, y = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))
        vignetting_function = np.exp(-(((x - width // 2) ** 2) / (2 * sigma_x ** 2)
                                       + ((y - height // 2) ** 2) / (2 * sigma_y ** 2)))
        vignetting_function /= np.max(vignetting_function)
    else:
        with fits.open('data/sample_vignetting.fits') as hdul:
            vignetting_function = hdul[1].data

    input_data.data[:, :] *= np.flipud(vignetting_function)

    return input_data


def add_deficient_pixels(input_data):
    return input_data


def add_stray_light(input_data):
    return input_data


def uncorrect_psf(input_data, psf_model):
    input_data.data[...] = psf_model.correct_image(input_data.data, alpha=1.5, epsilon=0.5)[...]
    return input_data


def starfield_misalignment(input_data, cr_offset_scale: float = 0.1, pc_offset_scale: float = 0.1):
    cr_offsets = np.random.normal(0, cr_offset_scale, 2)
    input_data.wcs.wcs.crval = input_data.wcs.wcs.crval + cr_offsets

    pc_offset = np.random.normal(0, pc_offset_scale) * u.deg
    current_crota = extract_crota_from_wcs(input_data.wcs)
    new_pc = calculate_pc_matrix(current_crota + pc_offset, input_data.wcs.wcs.cdelt)
    input_data.wcs.wcs.pc = new_pc

    return input_data


def generate_l0_pmzp(input_file, path_output, time_obs, time_delta, rotation_stage, spacecraft_id, psf_model):
    """Generates level 0 polarized synthetic data"""

    # Read in the input data
    with fits.open(input_file) as hdul:
        input_data_array = hdul[1].data
        input_header = hdul[1].header
        input_wcs = WCS(input_header, fobj=hdul)

    input_data = NDCube(data=input_data_array, meta=dict(input_header), wcs=input_wcs)

    # Define the output data product
    product_code = input_data.meta['TYPECODE'] + input_data.meta['OBSCODE']
    product_level = '0'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta['DATE-OBS'] = str(time_obs)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(wcs=input_wcs)
    for key in output_header.keys():
        if (key in input_header) and (output_header[key] in ['', None]) and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key].value = input_data.meta[key]

    input_data = NDCube(data=input_data, meta=output_meta, wcs=input_wcs)

    # TODO - fold into reprojection?
    output_data = starfield_misalignment(input_data)

    output_data = uncorrect_psf(output_data, psf_model)

    # TODO - look for stray light model from WFI folks? Or just use some kind of gradient with poisson noise.
    output_data = add_stray_light(output_data)

    output_data = add_deficient_pixels(output_data)

    output_data = uncorrect_vignetting_lff(output_data)

    output_data = streaking(output_data)

    output_data, photon_counts = photometric_uncalibration(output_data)

    # TODO: do in a cleaner way
    photon_counts[np.isnan(photon_counts)] = 0
    photon_counts[photon_counts < 0] = 0
    output_data.data[...] += np.random.poisson(lam=photon_counts, size=output_data.data.shape)

    output_data = spiking(output_data)

    output_data = certainty_estimate(output_data)

    # TODO - Sync up any final header data here

    # Set output dtype
    # TODO - also check this in the output data w/r/t BITPIX
    output_data.data[output_data.data > 2**16-1] = 2**16-1
    write_data = NDCube(data=output_data.data[:, :].astype(np.int32), meta=output_data.meta, wcs=output_data.wcs)
    write_data = update_spacecraft_location(write_data, write_data.meta.astropy_time)

    # Write out
    output_data.meta['FILEVRSN'] = '1'
    write_ndcube_to_fits(write_data, path_output + get_base_file_name(output_data) + '.fits')

    print("done")

# @click.command()
# @click.argument('datadir', type=click.Path(exists=True))
def generate_l0_all(datadir, psf_model_path):
    """Generate all level 0 synthetic data"""

    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, 'synthetic_l0/')
    os.makedirs(outdir, exist_ok=True)
    print(f"Outputting to {outdir}")

    # Parse list of level 1 model data
    files_l1 = glob.glob(datadir + '/synthetic_l1/*.fits')
    print(f"Generating based on {len(files_l1)} files.")
    files_l1.sort()

    # Set the overall start time for synthetic data
    # Note the timing for data products - 32 minutes / low noise ; 8 minutes / clear ; 4 minutes / polarized
    time_start = datetime(2024, 6, 20, 0, 0, 0)

    # Generate a corresponding set of observation times for polarized trefoil / NFI data
    time_delta = timedelta(minutes=4)
    times_obs = np.arange(len(files_l1)) * time_delta + time_start

    psf_model = ArrayCorrector.load(psf_model_path)

    pool = ProcessPoolExecutor()
    futures = []
    # Run individual generators
    for i, (file_l1, time_obs) in tqdm(enumerate(zip(files_l1, times_obs)), total=len(files_l1)):
        rotation_stage = int((i % 16) / 2)
        futures.append(pool.submit(generate_l0_pmzp, file_l1, outdir, time_obs, time_delta,
                                   rotation_stage, None, psf_model))

    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            future.result()
            pbar.update(1)


if __name__ == '__main__':
    generate_l0_all("/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/",
                    "/Users/jhughes/Desktop/repos/simpunch/synthetic_input_psf.h5")
