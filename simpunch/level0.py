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
from ndcube import NDCube
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            write_ndcube_to_fits)
from tqdm import tqdm


def certainty_estimate(input_data):
    return input_data


def photometric_uncalibration(input_data):
    return input_data


def spiking(input_data):
    spike_index = np.random.choice(input_data.data.shape[0] * input_data.data.shape[1], np.random.randint(0,20))
    spike_index2d = np.unravel_index(spike_index, input_data.data.shape)

    spike_values = np.random.normal(input_data.data.max() * 0.9, input_data.data.max() * 0.1, len(spike_index))

    input_data.data[spike_index2d] = spike_values

    return input_data


def streaking(input_data):
    return input_data


def uncorrect_vignetting_LFF(input_data):
    return input_data


def add_deficient_pixels(input_data):
    return input_data


def add_stray_light(input_data):
    return input_data


def uncorrect_psf(input_data):
    return input_data


def starfield_misalignment(input_data):
    return input_data


def generate_l0_pmzp(input_file, path_output, time_obs, time_delta, rotation_stage, spacecraft_id):
    """Generates level 0 polarized synthetic data"""

    # Read in the input data
    with fits.open(input_file) as hdul:
        input_data_array = hdul[1].data
        input_header = hdul[1].header
        input_wcs = WCS(input_header).dropaxis(2)

    input_data = NDCube(data=input_data_array, meta=dict(input_header), wcs=input_wcs)

    # Define the output data product
    product_code = input_data.meta['TYPECODE'] + spacecraft_id
    product_level = '0'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header()
    for key in output_header.keys():
        if (key in input_header) and (output_header[key] in ['', None]) and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key].value = input_data.meta[key]

    input_data = NDCube(data=input_data, meta=output_meta, wcs=input_wcs)

    # Starfield misalignment
    output_data = starfield_misalignment(input_data)

    # Uncorrect PSF
    output_data = uncorrect_psf(output_data)

    # Add stray light
    output_data = add_stray_light(output_data)

    # Add deficient pixels
    output_data = add_deficient_pixels(output_data)

    # Uncorrect vignetting and LFF
    output_data = uncorrect_vignetting_LFF(output_data)

    # Add streaks
    output_data = streaking(output_data)

    # Add spikes
    output_data = spiking(output_data)

    # Uncalibrate photometry
    output_data = photometric_uncalibration(output_data)

    # Estimate certainty
    output_data = certainty_estimate(output_data)

    # TODO - Sync up any final header data here

    # Write out
    output_data.meta['FILEVRSN'] = '0'
    write_ndcube_to_fits(output_data, path_output + get_base_file_name(output_data) + '.fits',
                         skip_wcs_conversion=True)


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
