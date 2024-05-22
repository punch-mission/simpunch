"""
Generates synthetic level 2 data
PTM - PUNCH Level-2 Polarized (MZP) Mosaic
"""
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import astropy.units as u
import click
import numpy as np
import reproject
import scipy.ndimage
from astropy.coordinates import get_sun
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import add_stokes_axis_to_wcs
from punchbowl.data import NormalizedMetadata, PUNCHData
from sunpy.coordinates import sun
from sunpy.coordinates.ephemeris import get_earth
from tqdm import tqdm

import solpolpy


def add_fcorona(input_data):
    """Adds synthetic f-corona model"""

    output_data = input_data
    return output_data


def add_starfield(input_data):
    """Adds synthetic starfield"""

    output_data = input_data
    return output_data


def remix_polarization(input_data):
    """Remix polarization from (B, pB) to (M,Z,P) using solpolpy"""

    output_data = input_data
    return output_data


def generate_l2_ptm(input_pdata, path_output, time_obs, time_delta, rotation_stage):
    """Generates level 2 PTM synthetic data"""

    # Define the output data product
    product_code = 'PTM'
    product_level = '2'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_wcs = input_pdata.wcs

    # Synchronize overlapping metadata keys
    for key in output_meta.keys():
        if key in input_pdata.keys():
            output_meta[key].value = input_pdata[key].value

    # Remix polarization
    output_data = remix_polarization(input_pdata)

    # Add starfield
    output_data = add_starfield(output_data)

    # Add f-corona
    output_data = add_fcorona(output_data)

    # Package into a PUNCHdata object
    output_pdata = PUNCHData(data=output_data, wcs=output_wcs, meta=output_meta)

    # Write out
    output_pdata.write(path_output + output_pdata.filename_base + '.fits', skip_wcs_conversion=False)


@click.command()
@click.argument('datadir', type=click.Path(exists=True))
def generate_l2_all(datadir):
    """Generate all level 2 synthetic data
     L2_PTM <- f-corona subtraction <- starfield subtraction <- remix polarization <- L3_PTM"""

    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, 'synthetic_L3/')
    print(f"Outputting to {outdir}")

    # Parse list of level 3 model data
    files_ptm = glob.glob(datadir + '/synthetic_l3/*PTM*.fits')
    print(f"Generating based on {len(files_ptm)} PTM files.")
    files_ptm.sort()

    # Set the overall start time for synthetic data
    # Note the timing for data products - 32 minutes / low noise ; 8 minutes / clear ; 4 minutes / polarized
    time_start = datetime(2023, 7, 4, 0, 0, 0)

    # Generate a corresponding set of observation times for polarized trefoil / NFI data
    time_delta = timedelta(minutes=4)
    times_obs = np.arange(len(files_ptm)) * time_delta + time_start

    pool = ProcessPoolExecutor()
    futures = []
    # Run individual generators
    for i, (file_ptm, time_obs) in tqdm(enumerate(zip(files_ptm, times_obs)), total=len(files_ptm)):
        rotation_stage = i % 8
        futures.append(pool.submit(generate_l2_ptm, file_ptm, outdir, time_obs, time_delta, rotation_stage))

    with tqdm(total=len(futures)) as pbar:
        for _ in as_completed(futures):
            pbar.update(1)
