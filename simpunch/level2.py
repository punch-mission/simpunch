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
from astropy.coordinates import GCRS, EarthLocation, SkyCoord, StokesSymbol, custom_stokes_symbol_mapping
from tqdm import tqdm

from ndcube import NDCube, NDCollection

import solpolpy

PUNCH_STOKES_MAPPING = custom_stokes_symbol_mapping({10: StokesSymbol("pB", "polarized brightness"),
                                                     11: StokesSymbol("B", "total brightness")})

def gen_fcorona(shape):
    # TODO - generate actual model fcorona here
    fcorona = np.zeros(shape)
    return fcorona


def add_fcorona(input_data):
    """Adds synthetic f-corona model"""

    fcorona = gen_fcorona(input_data.data.shape)

    output_data = input_data + fcorona
    return output_data


def gen_starfield(shape):
    # TODO - generate actual starfield here (from existing code)
    starfield = np.zeros(shape)
    return starfield


def add_starfield(input_data):
    """Adds synthetic starfield"""

    starfield = gen_starfield(input_data.data.shape)

    output_data = input_data + starfield
    return output_data


def remix_polarization(input_data):
    """Remix polarization from (B, pB) to (M,Z,P) using solpolpy"""

    # Unpack data into a NDCollection object
    data_collection = NDCollection([("B", input_data[0, :, :]), ("pB", input_data[1, :, :])], aligned_axes='all')

    resolved_data_collection = solpolpy.resolve(data_collection, "MZP", imax_effect=False)

    # Repack data
    data_list = []
    wcs_list = []
    uncertainty_list = []
    for key in resolved_data_collection:
        data_list.append(resolved_data_collection[key].data)
        wcs_list.append(resolved_data_collection[key].wcs)
        uncertainty_list.append(resolved_data_collection[key].uncertainty)

    # Remove alpha channel
    data_list.pop()
    wcs_list.pop()
    uncertainty_list.pop()

    # Repack into a PUNCHData object
    new_data = np.stack(data_list, axis=0)
    if uncertainty_list[0] is not None:
        new_uncertainty = np.stack(uncertainty_list, axis=0)
    else:
        new_uncertainty = None

    new_wcs = input_data.wcs.copy()

    output_data = PUNCHData(data=new_data, wcs=new_wcs, uncertainty=new_uncertainty, meta=input_data.meta)

    return output_data


def generate_l2_ptm(input_file, path_output, time_obs, time_delta, rotation_stage):
    """Generates level 2 PTM synthetic data"""

    # Read in the input data
    # input_pdata = PUNCHData.from_fits(input_file)
    with fits.open(input_file) as hdul:
        input_data = hdul[1].data
        input_header = hdul[1].header

    input_pdata = PUNCHData(data = input_data, meta = input_header, wcs = WCS(input_header))

    # Define the output data product
    product_code = 'PTM'
    product_level = '2'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_wcs = input_pdata.wcs

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header()
    for key in output_header.keys():
        if (key in input_header) and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key].value = input_pdata.meta[key]

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


# @click.command()
# @click.argument('datadir', type=click.Path(exists=True))
def generate_l2_all(datadir):
    """Generate all level 2 synthetic data
     L2_PTM <- f-corona subtraction <- starfield subtraction <- remix polarization <- L3_PTM"""

    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, 'synthetic_l2/')
    print(f"Outputting to {outdir}")

    # Parse list of level 3 model data
    files_ptm = glob.glob(datadir + '/synthetic_l3/*PTM*.fits')
    print(f"Generating based on {len(files_ptm)} PTM files.")
    files_ptm.sort()

    # TODO - remove after testing
    files_ptm = files_ptm[0:2]

    # Set the overall start time for synthetic data
    # Note the timing for data products - 32 minutes / low noise ; 8 minutes / clear ; 4 minutes / polarized
    time_start = datetime(2023, 7, 4, 0, 0, 0)

    # Generate a corresponding set of observation times for polarized trefoil / NFI data
    time_delta = timedelta(minutes=4)
    times_obs = np.arange(len(files_ptm)) * time_delta + time_start

    # TODO - revert after testing
    for i, (file_ptm, time_obs) in tqdm(enumerate(zip(files_ptm, times_obs)), total=len(files_ptm)):
        rotation_stage = i % 8
        generate_l2_ptm(file_ptm, outdir, time_obs, time_delta, rotation_stage)

    # pool = ProcessPoolExecutor()
    # futures = []
    # # Run individual generators
    # for i, (file_ptm, time_obs) in tqdm(enumerate(zip(files_ptm, times_obs)), total=len(files_ptm)):
    #     rotation_stage = i % 8
    #     futures.append(pool.submit(generate_l2_ptm, file_ptm, outdir, time_obs, time_delta, rotation_stage))
    #
    # with tqdm(total=len(futures)) as pbar:
    #     for _ in as_completed(futures):
    #         pbar.update(1)


if __name__ == "__main__":
    generate_l2_all('/Users/clowder/data/punch')
