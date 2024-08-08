"""
Generates synthetic level 0 data
"""
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import click
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from ndcube import NDCube
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            write_ndcube_to_fits)
from tqdm import tqdm


def certainty_estimate(input_data):
    return input_data


def photometric_uncalibration(input_data,
                              gain: float = 4.3,
                              bitrate_signal: int = 16):

    # Input data should be in calibrated units of MSB
    # First establish a conversion rate backwards to photons
    photons_per_MSB = 9.736531166539638e+38

    photon_data = input_data.data * photons_per_MSB

    # Now using the gain convert to DN
    dn_data = photon_data / gain

    # Clip / scale the data at the signal bitrate
    dn_data = np.interp(dn_data,
        (np.min(dn_data), np.max(dn_data)),
        (0, 2**(bitrate_signal-1) - 1),
    )
    # dn_data = np.clip(dn_data, 0, 2**bitrate_signal-1)

    dn_data = dn_data.astype(int)

    input_data.data[:,:] = dn_data

    return input_data


def spiking(input_data, spike_scaling=2**16-1):
    spike_index = np.random.choice(input_data.data.shape[0] * input_data.data.shape[1], np.random.randint(0,20))
    spike_index2d = np.unravel_index(spike_index, input_data.data.shape)

    # spike_values = np.random.normal(input_data.data.max() * spike_scaling, input_data.data.max() * 0.1, len(spike_index))
    spike_values = spike_scaling

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

    input_data.data[:, :] = streak_matrix @ input_data.data[:, :]

    return input_data


def uncorrect_vignetting_lff(input_data):
    if input_data.meta['OBSCODE'].value == 4:
        width, height = 2048, 2048
        sigma_x, sigma_y = width / 1, height / 1
        x, y = np.meshgrid(np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))
        vignetting_function = np.exp(-(((x - width // 2) ** 2) / (2 * sigma_x ** 2) + ((y - height // 2) ** 2) / (2 * sigma_y ** 2)))
        vignetting_function /= np.max(vignetting_function)

        input_data.data[:,:] *= vignetting_function
    else:
        with fits.open('data/sample_vignetting.fits') as hdul:
            vignetting_function = hdul[1].data

        input_data.data[:,:] *= vignetting_function

    return input_data


def add_deficient_pixels(input_data):
    return input_data


def add_stray_light(input_data):
    return input_data


def uncorrect_psf(input_data):
    return input_data


def starfield_misalignment(input_data, cr_offset_scale: float = 0.1, pc_offset_scale: float = 0.1):
    cr_offsets = np.random.normal(0, cr_offset_scale, 2)
    input_data.wcs.wcs.crval = input_data.wcs.wcs.crval + cr_offsets

    pc_offset = np.random.normal(0, pc_offset_scale, 1)[0] * u.deg
    rotation_matrix = np.array([
        [np.cos(pc_offset), -np.sin(pc_offset)],
        [np.sin(pc_offset), np.cos(pc_offset)]
    ])

    input_data.wcs.wcs.pc = np.dot(input_data.wcs.wcs.pc, rotation_matrix)

    return input_data


def generate_l0_pmzp(input_file, path_output, time_obs, time_delta, rotation_stage, spacecraft_id):
    """Generates level 0 polarized synthetic data"""

    # Read in the input data
    with fits.open(input_file) as hdul:
        input_data_array = hdul[1].data
        input_header = hdul[1].header
        input_wcs = WCS(input_header, fobj=hdul)

    input_data = NDCube(data=input_data_array, meta=dict(input_header), wcs=input_wcs)

    # Define the output data product
    product_code = input_data.meta['TYPECODE'] + spacecraft_id
    product_level = '0'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta['DATE-OBS'] = str(time_obs)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(wcs=input_wcs)
    for key in output_header.keys():
        if (key in input_header) and (output_header[key] in ['', None]) and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key].value = input_data.meta[key]

    input_data = NDCube(data=input_data, meta=output_meta, wcs=input_wcs)

    output_data = starfield_misalignment(input_data)

    output_data = uncorrect_psf(output_data)

    output_data = add_stray_light(output_data)

    output_data = add_deficient_pixels(output_data)

    output_data = uncorrect_vignetting_lff(output_data)

    output_data = streaking(output_data)

    output_data = photometric_uncalibration(output_data)

    output_data = spiking(output_data)

    output_data = certainty_estimate(output_data)

    # TODO - Sync up any final header data here

    # Check that output data is of the right DN datatype
    output_data.data[:,:] = output_data.data[:,:].astype(int)

    # Write out
    output_data.meta['FILEVRSN'] = '0'
    write_ndcube_to_fits(output_data, path_output + get_base_file_name(output_data) + '.fits')


# @click.command()
# @click.argument('datadir', type=click.Path(exists=True))
def generate_l0_all(datadir):
    """Generate all level 0 synthetic data"""

    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, 'synthetic_l0/')
    print(f"Outputting to {outdir}")

    # Parse list of level 1 model data
    files_l1 = glob.glob(datadir + '/synthetic_l1/*.fits')
    print(f"Generating based on {len(files_l1)} files.")
    files_l1.sort()

    files_l1 = files_l1[0:2]

    # Set the overall start time for synthetic data
    # Note the timing for data products - 32 minutes / low noise ; 8 minutes / clear ; 4 minutes / polarized
    time_start = datetime(2024, 6, 20, 0, 0, 0)

    # Generate a corresponding set of observation times for polarized trefoil / NFI data
    time_delta = timedelta(minutes=4)
    times_obs = np.arange(len(files_l1)) * time_delta + time_start

    for i, (file_l1, time_obs) in tqdm(enumerate(zip(files_l1, times_obs)), total=len(files_l1)):
        rotation_stage = int((i % 16) / 2)
        generate_l0_pmzp(file_l1, outdir, time_obs, time_delta, rotation_stage, '1')

    # pool = ProcessPoolExecutor()
    # futures = []
    # # Run individual generators
    # for i, (file_l1, time_obs) in tqdm(enumerate(zip(files_l1, times_obs)), total=len(files_l1)):
    #     rotation_stage = int((i % 16) / 2)
    #     futures.append(pool.submit(generate_l0_pmzp, file_l1, outdir, time_obs, time_delta, rotation_stage, '1'))
    #     futures.append(pool.submit(generate_l0_pmzp, file_l1, outdir, time_obs, time_delta, rotation_stage, '2'))
    #     futures.append(pool.submit(generate_l0_pmzp, file_l1, outdir, time_obs, time_delta, rotation_stage, '3'))
    #     futures.append(pool.submit(generate_l0_pmzp, file_l1, outdir, time_obs, time_delta, rotation_stage, '4'))
    #
    # with tqdm(total=len(futures)) as pbar:
    #     for _ in as_completed(futures):
    #         pbar.update(1)


if __name__ == "__main__":
    generate_l0_all('/Users/clowder/data/punch')
