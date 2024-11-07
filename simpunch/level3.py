"""
Generate synthetic level 3 data.

PTM - PUNCH Level-3 Polarized Mosaic
CTM - PUNCH Level-3 Clear Mosaic
"""
import glob
import os
from datetime import datetime, timedelta

import astropy.units as u
import numpy as np
import scipy.ndimage
from astropy.constants import R_sun
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from astropy.wcs.utils import add_stokes_axis_to_wcs, proj_plane_pixel_area
from ndcube import NDCube
from prefect import flow, task
from prefect.futures import wait
from prefect_dask import DaskTaskRunner
from punchbowl.data import NormalizedMetadata, write_ndcube_to_fits
from punchbowl.data.io import get_base_file_name
from tqdm import tqdm

from .util import update_spacecraft_location


def define_mask(shape: (int, int) = (4096, 4096), distance_value: float =0.68) -> np.ndarray:
    """Define a mask to describe the FOV for low-noise PUNCH data products."""
    center = (int(shape[0] / 2), int(shape[1] / 2))

    Y, X = np.ogrid[:shape[0], :shape[1]]  # noqa: N806
    dist_arr = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    return (dist_arr / dist_arr.max()) < distance_value


def define_trefoil_mask(rotation_stage: int = 0) -> np.ndarray:
    """Define a mask to describe the FOV for trefoil mosaic PUNCH data products."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return np.load(os.path.join(dir_path, "data/trefoil_mask.npz"))["trefoil_mask"][rotation_stage,:,:]


def generate_uncertainty(pdata: NDCube) -> NDCube:
    """Generate realistic uncertainty for PUNCH."""
    # Input data is scaled to MSB
    # Convert to photons
    # Get the pixel scale in degrees
    pixel_scale = abs(pdata.wcs.pixel_scale_matrix[0, 0]) * u.deg

    # Convert the pixel scale to radians
    pixel_scale_rad = pixel_scale.to(u.rad)

    # Get the physical size of the Sun
    sun_radius = R_sun.to(u.m)

    # Calculate the physical area per pixel
    pixel_area = proj_plane_pixel_area(pdata.wcs) * (u.deg ** 2)
    physical_area_per_pixel = (pixel_area * (sun_radius ** 2) / (pixel_scale_rad ** 2)).to(u.m ** 2)

    # Constants
    h = 6.62607015e-34 * u.m ** 2 * u.kg / u.s  # Planck's constant
    c = 2.99792458e8 * u.m / u.s  # Speed of light
    wavelength = 530 * u.nm  # Wavelength in nanometers
    exposure_time = 60 * u.s  # Exposure in seconds

    # Calculate energy of a single photon
    energy_per_photon = (h * c / wavelength).to(u.J)

    # Now get the energy per unit of irradiance
    # Given irradiance in W/m^2/sr
    irradiance = 1 * u.W / (u.m ** 2 * u.sr)

    # Convert irradiance to energy per second per pixel
    energy_per_second = irradiance * 4 * np.pi * u.sr * physical_area_per_pixel

    # Convert energy per second to photon count
    photon_count = (energy_per_second * exposure_time).to(u.J) / energy_per_photon

    photon_array = pdata.data * photon_count
    photon_noise = np.sqrt(photon_array)
    uncertainty = photon_noise
    uncertainty[pdata.data == 0] = np.inf
    pdata.uncertainty.array = uncertainty

    return pdata


def assemble_punchdata_polarized(input_tb: str, input_pb: str,
                                 wcs: WCS, product_code: str,
                                 product_level: str, mask: np.ndarray | None = None) -> NDCube:
    """Assemble a punchdata object with correct metadata."""
    with fits.open(input_tb) as hdul:
        data_tb = hdul[1].data / 1e8  # the 1e8 comes from the units on FORWARD output
        if data_tb.shape == (2048, 2048):
            data_tb = scipy.ndimage.zoom(data_tb, 2, order=0)
        data_tb[np.where(data_tb == -9.999e-05)] = 0. # noqa: PLR2004
        if mask is not None:
            data_tb = data_tb * mask

    with fits.open(input_pb) as hdul:
        data_pb = hdul[1].data / 1e8 # the 1e8 comes from the units on FORWARD output
        if data_pb.shape == (2048, 2048):
            data_pb = scipy.ndimage.zoom(data_pb, 2, order=0)
        data_pb[np.where(data_pb == -9.999e-05)] = 0. # noqa: PLR2004
        if mask is not None:
            data_pb = data_pb * mask

    datacube = np.stack([data_tb, data_pb]).astype("float32")
    uncertainty = StdDevUncertainty(np.zeros(datacube.shape))
    uncertainty.array[datacube == 0] = 1
    meta = NormalizedMetadata.load_template(product_code, product_level)
    return NDCube(data=datacube, wcs=wcs, meta=meta, uncertainty=uncertainty)


def assemble_punchdata_clear(input_tb: str, wcs: WCS,
                             product_code: str, product_level: str, mask: np.ndarray | None =None) -> NDCube:
    """Assemble a punchdata object with correct metadata for a clear data product."""
    with fits.open(input_tb) as hdul:
        data_tb = hdul[1].data / 1e8  # the 1e8 comes from the units on FORWARD output
        if data_tb.shape == (2048, 2048):
            data_tb = scipy.ndimage.zoom(data_tb, 2, order=0)
        data_tb[np.where(data_tb == -9.999e-05)] = 0. # noqa: PLR2004
        if mask is not None:
            data_tb = data_tb * mask

    datacube = data_tb.astype("float32")
    uncertainty = StdDevUncertainty(np.zeros(datacube.shape))
    uncertainty.array[datacube == 0] = 1
    meta = NormalizedMetadata.load_template(product_code, product_level)
    return NDCube(data=datacube, wcs=wcs, meta=meta, uncertainty=uncertainty)


@task
def generate_l3_ptm(input_tb: str, input_pb: str, path_output: str,
                    time_obs: datetime, time_delta: timedelta, rotation_stage: int) -> None:
    """Generate PTM - PUNCH Level-3 Polarized Mosaic."""
    # Define the mosaic WCS (helio)
    mosaic_shape = (4096, 4096)
    mosaic_wcs = WCS(naxis=2)
    mosaic_wcs.wcs.crpix = mosaic_shape[1] / 2 - 0.5, mosaic_shape[0] / 2 - 0.5
    mosaic_wcs.wcs.crval = 0, 0
    mosaic_wcs.wcs.cdelt = 0.0225, 0.0225
    mosaic_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
    mosaic_wcs = add_stokes_axis_to_wcs(mosaic_wcs, 2)

    # Mask data to define the field of view
    mask = define_trefoil_mask(rotation_stage=rotation_stage)
    mask = define_mask(shape=(4096, 4096), distance_value=0.68)
    mask = None

    # Read data and assemble into PUNCHData object
    pdata = assemble_punchdata_polarized(input_tb, input_pb,
                                         mosaic_wcs, product_code="PTM", product_level="3", mask=mask)

    # Update required metadata
    tstring_start = time_obs.strftime("%Y-%m-%dT%H:%M:%S.000")
    tstring_end = (time_obs + time_delta).strftime("%Y-%m-%dT%H:%M:%S.000")
    tstring_avg = (time_obs + time_delta / 2).strftime("%Y-%m-%dT%H:%M:%S.000")

    pdata.meta["DATE-OBS"] = tstring_start
    pdata.meta["DATE-BEG"] = tstring_start
    pdata.meta["DATE-END"] = tstring_end
    pdata.meta["DATE-AVG"] = tstring_avg
    pdata.meta["DATE"] = (time_obs + time_delta + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S.000")
    pdata.meta["DESCRPTN"] = "Simulated " + pdata.meta["DESCRPTN"].value
    pdata.meta["TITLE"] = "Simulated " + pdata.meta["TITLE"].value

    pdata = update_spacecraft_location(pdata, time_obs)
    pdata = generate_uncertainty(pdata)
    write_ndcube_to_fits(pdata, path_output + get_base_file_name(pdata) + ".fits")


@task
def generate_l3_ctm(input_tb: str,
                    path_output: str,
                    time_obs: datetime,
                    time_delta: timedelta,
                    rotation_stage: int) -> None:
    """Generate CTM - PUNCH Level-3 Clear Mosaic."""
    # Define the mosaic WCS (helio)
    mosaic_shape = (4096, 4096)
    mosaic_wcs = WCS(naxis=2)
    mosaic_wcs.wcs.crpix = mosaic_shape[1] / 2 - 0.5, mosaic_shape[0] / 2 - 0.5
    mosaic_wcs.wcs.crval = 0, 0
    mosaic_wcs.wcs.cdelt = 0.0225, 0.0225
    mosaic_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"

    # Mask data to define the field of view
    mask = define_trefoil_mask(rotation_stage=rotation_stage)
    mask = define_mask(shape=(4096, 4096), distance_value=0.68)
    mask = None

    # Read data and assemble into PUNCHData object
    pdata = assemble_punchdata_clear(input_tb, mosaic_wcs, product_code="CTM", product_level="3", mask=mask)

    # Update required metadata
    tstring_start = time_obs.strftime("%Y-%m-%dT%H:%M:%S.000")
    tstring_end = (time_obs + time_delta).strftime("%Y-%m-%dT%H:%M:%S.000")
    tstring_avg = (time_obs + time_delta / 2).strftime("%Y-%m-%dT%H:%M:%S.000")

    pdata.meta["DATE-OBS"] = tstring_start
    pdata.meta["DATE-BEG"] = tstring_start
    pdata.meta["DATE-END"] = tstring_end
    pdata.meta["DATE-AVG"] = tstring_avg
    pdata.meta["DATE"] = (time_obs + time_delta + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S.000")
    pdata.meta["DESCRPTN"] = "Simulated " + pdata.meta["DESCRPTN"].value
    pdata.meta["TITLE"] = "Simulated " + pdata.meta["TITLE"].value

    pdata = update_spacecraft_location(pdata, time_obs)
    pdata = generate_uncertainty(pdata)
    write_ndcube_to_fits(pdata, path_output + get_base_file_name(pdata) + ".fits")


@flow(log_prints=True, task_runner=DaskTaskRunner(
    cluster_kwargs={"n_workers": 8, "threads_per_worker": 2},
))
def generate_l3_all(datadir: str, start_time: datetime, num_repeats: int = 1) -> None:
    """Generate all level 3 synthetic data."""
    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, "synthetic_l3/")
    os.makedirs(outdir, exist_ok=True)
    print(f"Outputting to {outdir}")

    # Parse list of model data
    files_tb = glob.glob(datadir + "/synthetic_cme/*_TB.fits")
    files_pb = glob.glob(datadir + "/synthetic_cme/*_PB.fits")
    print(f"Generating based on {len(files_tb)} TB files and {len(files_pb)} PB files.")
    files_tb.sort()
    files_pb.sort()

    # Stack and repeat these data for testing
    files_tb = np.tile(files_tb, num_repeats)
    files_pb = np.tile(files_pb, num_repeats)

    # Generate a corresponding set of observation times for polarized trefoil / NFI data
    time_delta = timedelta(minutes=4)
    times_obs = np.arange(len(files_tb)) * time_delta + start_time

    rotation_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    runs = []
    for i, (file_tb, file_pb, time_obs) \
            in tqdm(enumerate(zip(files_tb, files_pb, times_obs, strict=False)), total=len(files_tb)):
        runs.append(generate_l3_ptm.submit(file_tb, file_pb, outdir, time_obs, time_delta, rotation_indices[i % 8]))
        runs.append(generate_l3_ctm.submit(file_tb, outdir, time_obs, time_delta, rotation_indices[i % 8]))
    wait(runs)
