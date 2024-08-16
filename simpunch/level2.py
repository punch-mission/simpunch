"""
Generates synthetic level 2 data
PTM - PUNCH Level-2 Polarized (MZP) Mosaic
"""
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import solpolpy
from astropy.coordinates import StokesSymbol, custom_stokes_symbol_mapping
from astropy.table import QTable
from astropy.wcs import WCS
from ndcube import NDCollection, NDCube
from photutils.datasets import make_gaussian_sources_image, make_noise_image
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            load_ndcube_from_fits, write_ndcube_to_fits)
from punchbowl.data.wcs import calculate_celestial_wcs_from_helio
from tqdm import tqdm

from simpunch.stars import (filter_for_visible_stars, find_catalog_in_image,
                            load_raw_hipparcos_catalog)
from simpunch.util import update_spacecraft_location

PUNCH_STOKES_MAPPING = custom_stokes_symbol_mapping({10: StokesSymbol("pB", "polarized brightness"),
                                                     11: StokesSymbol("B", "total brightness")})


def gen_fcorona(shape):
    fcorona = np.zeros(shape)

    # Superellipse parameters
    a = 600  # Horizontal axis radius
    b = 300  # Vertical axis radius

    tilt_angle_deg = 3  # Tilt angle in degrees

    # Convert tilt angle to radians
    tilt_angle_rad = np.deg2rad(tilt_angle_deg)

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]))
    x_center, y_center = shape[1] // 2, shape[2] // 2

    # Rotate coordinates (x, y) around the center
    x_rotated = (x - x_center) * np.cos(tilt_angle_rad) + (y - y_center) * np.sin(tilt_angle_rad) + x_center
    y_rotated = -(x - x_center) * np.sin(tilt_angle_rad) + (y - y_center) * np.cos(tilt_angle_rad) + y_center

    # Calculate distance from the center normalized by a and b for rotated coordinates
    distance = np.sqrt(((x_rotated - x_center) / a) ** 2 + ((y_rotated - y_center) / b) ** 2)

    # Define a function for varying n with radius
    def n_function(r, max_radius, min_n, max_n):
        # Example of a linear function for n based on radius
        return min_n + (max_n - min_n) * (r / max_radius)

    # Set parameters for varying n
    max_radius = np.sqrt((shape[1] / 2) ** 2 + (shape[2] / 2) ** 2)  # Maximum radius from center
    min_n = 1.54  # Minimum value of n
    max_n = 1.65  # Maximum value of n

    # Calculate n as a function of distance from the center
    n = n_function(distance, max_radius, min_n, max_n)

    # Calculate the superellipse equation
    superellipse = (np.abs((x_rotated - x_center) / a) ** n + np.abs((y_rotated - y_center) / b) ** n) ** (1 / n)

    # Normalize distance to range [0, 1]
    superellipse = superellipse / (2 ** (1 / n))

    # Apply a Gaussian-like profile to simulate intensity variation
    max_distance = 1
    fcorona_profile = np.exp(-superellipse ** 2 / (2 * max_distance ** 2))

    # Normalize profile to [0, 1] and scale to desired magnitude
    fcorona_profile = fcorona_profile / fcorona_profile.max() * 1e-12

    for i in np.arange(fcorona.shape[0]):
        fcorona[i, :, :] = fcorona_profile[:, :]

    return fcorona


def add_fcorona(input_data):
    """Adds synthetic f-corona model"""

    fcorona = gen_fcorona(input_data.data.shape)

    fcorona = fcorona * (input_data.data != 0)

    input_data.data[...] = input_data.data[...] + fcorona
    return  input_data


def gen_starfield(wcs,
                  img_shape,
                  fwhm,
                  wcs_mode: str = 'all',
                  mag_set=0,
                  flux_set=500_000,
                  noise_mean: float | None = 25.0,
                  noise_std: float | None = 5.0,
                  dimmest_magnitude=8):
    sigma = fwhm / 2.355

    catalog = load_raw_hipparcos_catalog()
    filtered_catalog = filter_for_visible_stars(catalog,
                                                dimmest_magnitude=dimmest_magnitude)
    stars = find_catalog_in_image(filtered_catalog,
                                  wcs,
                                  img_shape,
                                  mode=wcs_mode)
    star_mags = stars['Vmag']

    sources = QTable()
    sources['x_mean'] = stars['x_pix']
    sources['y_mean'] = stars['y_pix']
    sources['x_stddev'] = np.ones(len(stars)) * sigma
    sources['y_stddev'] = np.ones(len(stars)) * sigma
    sources['flux'] = flux_set * np.power(10, -0.4 * (star_mags - mag_set))
    sources['theta'] = np.zeros(len(stars))

    fake_image = make_gaussian_sources_image(img_shape, sources)
    if noise_mean is not None and noise_std is not None:  # we only add noise if it's specified
        fake_image += make_noise_image(img_shape, 'gaussian', mean=noise_mean, stddev=noise_std)

    return fake_image, sources


def add_starfield(input_data):
    """Adds synthetic starfield"""

    wcs_stellar_input = calculate_celestial_wcs_from_helio(input_data.wcs,
                                                           input_data.meta.astropy_time,
                                                           input_data.data.shape)

    shape = input_data.data[0,:,:].shape
    wcs_stellar = WCS(naxis=2)
    wcs_stellar.wcs.crpix = shape[1] / 2 - 0.5, shape[0] / 2 - 0.5
    wcs_stellar.wcs.crval = wcs_stellar_input.wcs.crval[0], wcs_stellar_input.wcs.crval[1]
    wcs_stellar.wcs.cdelt = wcs_stellar_input.wcs.cdelt[0], wcs_stellar_input.wcs.cdelt[1]
    wcs_stellar.wcs.ctype = 'RA---ARC', 'DEC--ARC'

    wcs_stellar.wcs.pc = wcs_stellar_input.wcs.pc[0:2,0:2]

    starfield, stars = gen_starfield(wcs_stellar, input_data.data[0,:,:].shape,
                               fwhm=3, dimmest_magnitude=12, noise_mean=0, noise_std=0)

    starfield = starfield / 174037. * 5.4e-11

    starfield_data = np.zeros(input_data.data.shape)
    for i in range(starfield_data.shape[0]):
        starfield_data[i, :, :] = starfield * (input_data.data[i, :, :] != 0)

    input_data.data[...] = input_data.data[...] + starfield_data

    return input_data


def remix_polarization(input_data):
    """Remix polarization from (B, pB) to (M,Z,P) using solpolpy"""

    # Unpack data into a NDCollection object
    data_collection = NDCollection(
        [("B", NDCube(data=input_data.data[0], wcs=input_data.wcs)),
         ("pB", NDCube(data=input_data.data[1], wcs=input_data.wcs))],
        aligned_axes='all')

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

    return NDCube(data=new_data, wcs=new_wcs, uncertainty=new_uncertainty, meta=input_data.meta)


def generate_l2_ptm(input_file, path_output, time_obs, time_delta, rotation_stage):
    """Generates level 2 PTM synthetic data"""

    # Read in the input data
    input_pdata = load_ndcube_from_fits(input_file)

    # Define the output data product
    product_code = 'PTM'
    product_level = '2'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta['DATE-OBS'] = input_pdata.meta['DATE-OBS'].value
    output_wcs = input_pdata.wcs

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(output_wcs)
    for key in output_header.keys():
        if (key in input_pdata.meta) and output_header[key] == '' and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key].value = input_pdata.meta[key].value

    output_data = remix_polarization(input_pdata)
    output_data = add_starfield(output_data)
    output_data = add_fcorona(output_data)

    # Package into a PUNCHdata object
    output_pdata = NDCube(data=output_data.data.astype(np.float32), wcs=output_wcs, meta=output_meta)
    output_pdata = update_spacecraft_location(output_pdata, time_obs)

    # Write out
    write_ndcube_to_fits(output_pdata, path_output + get_base_file_name(output_pdata) + '.fits')


def generate_l2_all(datadir):
    """Generate all level 2 synthetic data
     L2_PTM <- f-corona subtraction <- starfield subtraction <- remix polarization <- L3_PTM"""

    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, 'synthetic_l2/')
    os.makedirs(outdir, exist_ok=True)
    print(f"Outputting to {outdir}")

    # Parse list of level 3 model data
    files_ptm = glob.glob(datadir + '/synthetic_l3/*PTM*.fits')
    print(f"Generating based on {len(files_ptm)} PTM files.")
    files_ptm.sort()

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
        rotation_stage = i % 8
        futures.append(pool.submit(generate_l2_ptm, file_ptm, outdir, time_obs, time_delta, rotation_stage))

    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            future.result()
            pbar.update(1)


if __name__ == '__main__':
    generate_l2_all("/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/")
