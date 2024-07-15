"""
Generates synthetic level 1 data
"""
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import click
import numpy as np
import solpolpy
from astropy.coordinates import StokesSymbol, custom_stokes_symbol_mapping
from astropy.io import fits
from astropy.wcs import WCS
from ndcube import NDCollection
from punchbowl.data import NormalizedMetadata, PUNCHData
from tqdm import tqdm

# TODO - mapping for MZP needed?

PUNCH_STOKES_MAPPING = custom_stokes_symbol_mapping({10: StokesSymbol("pB", "polarized brightness"),
                                                     11: StokesSymbol("B", "total brightness")})


def deproject(input_data):
    """Data deprojection"""

    return input_data


def mark_quality(input_data):
    """Data quality marking"""

    return input_data


def remix_polarization(input_data):
    """Remix polarization from (M, Z, P) to (P1, P2, P3) using solpolpy"""

    # Unpack data into a NDCollection object
    data_collection = NDCollection([("M", input_data[0, :, :]),
                                    ("Z", input_data[1, :, :]),
                                    ("P", input_data[1, :, :])], aligned_axes='all')

    resolved_data_collection = solpolpy.resolve(data_collection, "npol", imax_effect=False)

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

    return PUNCHData(data=new_data, wcs=new_wcs, uncertainty=new_uncertainty, meta=input_data.meta)


# TODO - Think about whether the three polarizers should be bundled together, or separate functions
def generate_l1_pm(input_file, path_output, time_obs, time_delta, rotation_stage):
    """Generates level 1 polarized synthetic data"""

    # Read in the input data
    with fits.open(input_file) as hdul:
        input_data = hdul[1].data
        input_header = hdul[1].header

    input_pdata = PUNCHData(data=input_data, meta=input_header, wcs=WCS(input_header))

    # Define the output data product
    product_code = 'PM?'
    product_level = '1'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_wcs = input_pdata.wcs

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header()
    for key in output_header.keys():
        if (key in input_header) and output_header[key] == '' and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key].value = input_pdata.meta[key]

    # Deproject to spacecraft frame
    output_data = deproject(input_pdata)

    # Quality marking
    output_data = mark_quality(output_data)

    # Polarization remixing
    output_data = remix_polarization(output_data)

    # Package into a PUNCHdata object
    output_pdata = PUNCHData(data=output_data.data.astype(np.float32), wcs=output_wcs, meta=output_meta)

    # Write out
    output_pdata.write(path_output + output_pdata.filename_base + '.fits', skip_wcs_conversion=True)


@click.command()
@click.argument('datadir', type=click.Path(exists=True))
def generate_l1_all(datadir):
    """Generate all level 1 synthetic data
     L1 <- polarization deprojection <- quality marking <- deproject to spacecraft FOV <- L2_PTM"""

    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, '/synthetic_L1/')
    print(f"Outputting to {outdir}")

    # Parse list of level 3 model data
    files_ptm = glob.glob(datadir + '/synthetic_L2/*PTM*.fits')
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
        futures.append(pool.submit(generate_l1_pm, file_ptm, outdir, time_obs, time_delta, rotation_stage))

    with tqdm(total=len(futures)) as pbar:
        for _ in as_completed(futures):
            pbar.update(1)


if __name__ == "__main__":
    generate_l1_all('/Users/clowder/data/punch')
