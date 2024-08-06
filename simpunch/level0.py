"""
Generates synthetic level 0 data
"""
import copy
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import astropy.units as u
import click
import numpy as np
import reproject
import solpolpy
from astropy.coordinates import (EarthLocation, SkyCoord, StokesSymbol,
                                 custom_stokes_symbol_mapping)
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, DistortionLookupTable
from astropy.wcs.utils import add_stokes_axis_to_wcs
from ndcube import NDCollection, NDCube
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            write_ndcube_to_fits)
from sunpy.coordinates import frames, sun
from tqdm import tqdm

from simpunch.stars import make_fake_stars_for_wfi


def certainty_estimate(input_data):
    return input_data


def photometric_uncalibration(input_data):
    return input_data


def spiking(input_data):
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


# TODO - add input to split this by polarization state
def generate_l0_pmzp(input_file, path_output, time_obs, time_delta, rotation_stage, spacecraft_id):
    """Generates level 0 polarized synthetic data"""

    # Read in the input data
    with fits.open(input_file) as hdul:
        input_data = hdul[1].data
        input_header = hdul[1].header

    input_pdata = NDCube(data=input_data, meta=dict(input_header), wcs=WCS(input_header))

    # TODO - check product code
    # Define the output data product
    product_code = 'PM' + spacecraft_id
    product_level = '0'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header()
    for key in output_header.keys():
        if (key in input_header) and (output_header[key] in ['', None]) and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key].value = input_pdata.meta[key]


    # Starfield misaligbment
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
    version_number = 0
    write_ndcube_to_fits(output_data, path_output + get_base_file_name(output_data) + str(version_number) + '.fits',
                         skip_wcs_conversion=True)


@click.command()
@click.argument('datadir', type=click.Path(exists=True))
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

    files_ptm = files_l1[0:5]

    # Set the overall start time for synthetic data
    # Note the timing for data products - 32 minutes / low noise ; 8 minutes / clear ; 4 minutes / polarized
    time_start = datetime(2024, 6, 20, 0, 0, 0)

    # Generate a corresponding set of observation times for polarized trefoil / NFI data
    time_delta = timedelta(minutes=4)
    times_obs = np.arange(len(files_ptm)) * time_delta + time_start

    pool = ProcessPoolExecutor()
    futures = []
    # Run individual generators
    for i, (file_ptm, time_obs) in tqdm(enumerate(zip(files_ptm, times_obs)), total=len(files_ptm)):
        rotation_stage = int((i % 16) / 2)
        futures.append(pool.submit(generate_l0_pmzp, file_ptm, outdir, time_obs, time_delta, rotation_stage, '1'))
        futures.append(pool.submit(generate_l0_pmzp, file_ptm, outdir, time_obs, time_delta, rotation_stage, '2'))
        futures.append(pool.submit(generate_l0_pmzp, file_ptm, outdir, time_obs, time_delta, rotation_stage, '3'))
        futures.append(pool.submit(generate_l0_pmzp, file_ptm, outdir, time_obs, time_delta, rotation_stage, '4'))

    with tqdm(total=len(futures)) as pbar:
        for _ in as_completed(futures):
            pbar.update(1)


def make_fake_level0(path_to_input, array_corrector_path, path_to_save):
    with fits.open(path_to_input) as hdul:
        test_header = hdul[0].header
    test_wcs = WCS(test_header)

    fake_star_data = make_fake_stars_for_wfi(test_wcs, array_corrector_path)

    my_data = PUNCHData(data=fake_star_data, wcs=test_wcs, uncertainty=np.zeros_like(fake_star_data))
    my_data.write(path_to_save)


# if __name__ == "__main__":
#     make_fake_level0("/Users/jhughes/Nextcloud/23103_PUNCH_Data/SOC_Data/PUNCH_WFI_EM_Starfield_campaign2_night2_phase3/calibrated/campaign2_night2_phase3_calibrated_000.new",
#                      "/Users/jhughes/Desktop/projects/PUNCH/psf_paper/paper-variable-point-spread-functions/scripts/punch_array_corrector.h5",
#                      "../fake_data.fits")


if __name__ == "__main__":
    generate_l0_all()
