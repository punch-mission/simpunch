"""Generate synthetic level 0 data."""
import glob
import os
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from ndcube import NDCube
from prefect import flow, task
from prefect.futures import wait
from prefect_dask import DaskTaskRunner
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            load_ndcube_from_fits, write_ndcube_to_fits)
from punchbowl.data.units import msb_to_dn
from punchbowl.data.wcs import calculate_pc_matrix, extract_crota_from_wcs
from punchbowl.level1.initial_uncertainty import compute_noise
from punchbowl.level1.sqrt import encode_sqrt
from regularizepsf import ArrayCorrector
from tqdm import tqdm

from simpunch.spike import generate_spike_image
from simpunch.util import update_spacecraft_location


def perform_photometric_uncalibration(input_data: NDCube) -> NDCube:
    """Undo quartic fit calibration."""
    return input_data


def add_spikes(input_data: NDCube) -> (NDCube, np.ndarray):
    """Add spikes to images."""
    spike_image = generate_spike_image(input_data.data.shape)
    input_data.data[...] += spike_image
    return input_data, spike_image

def create_streak_matrix(
    n: int, exposure_time: float, readout_line_time: float, reset_line_time: float,
) -> np.ndarray:
    """Construct the matrix that streaks an image."""
    lower = np.tril(np.ones((n, n)) * readout_line_time, -1)
    upper = np.triu(np.ones((n, n)) * reset_line_time, 1)
    diagonal = np.diagflat(np.ones(n) * exposure_time)

    return lower + upper + diagonal


def apply_streaks(input_data: NDCube,
                  exposure_time: float = 49 * 1000,
                  readout_line_time: float = 163/2148,
                  reset_line_time: float = 163/2148) -> NDCube:
    """Apply the streak matrix to the image."""
    streak_matrix = create_streak_matrix(input_data.data.shape[0],
                                         exposure_time, readout_line_time, reset_line_time)
    input_data.data[:, :] = streak_matrix @ input_data.data[:, :] / exposure_time
    return input_data


def uncorrect_vignetting(input_data: NDCube,
                         wfi_vignetting_model_path: str, nfi_vignetting_model_path: str) -> None:
    """Apply the vignetting function."""
    if int(input_data.meta["OBSCODE"].value) == 4:  # noqa: PLR2004
        vignetting_function_path = Path(nfi_vignetting_model_path)
    else:
        vignetting_function_path = Path(wfi_vignetting_model_path)
    cube = load_ndcube_from_fits(vignetting_function_path)

    input_data.data[:, :] *= cube.data[:, :]

    return input_data


def add_deficient_pixels(input_data: NDCube) -> NDCube:
    """Add deficient pixels to the image."""
    return input_data


def add_stray_light(input_data: NDCube) -> NDCube:
    """Add stray light to the image."""
    return input_data


def uncorrect_psf(input_data: NDCube, psf_model: ArrayCorrector) -> NDCube:
    """Apply an inverse PSF to an image."""
    input_data.data[...] = psf_model.correct_image(input_data.data, alpha=3.0, epsilon=0.3)[...]
    return input_data


def starfield_misalignment(input_data: NDCube,
                           cr_offset_scale: float = 0.1,
                           pc_offset_scale: float = 0.1) -> NDCube:
    """Offset the pointing in an image to simulate spacecraft uncertainty."""
    cr_offsets = np.random.normal(0, cr_offset_scale, 2)
    input_data.wcs.wcs.crval = input_data.wcs.wcs.crval + cr_offsets

    pc_offset = np.random.normal(0, pc_offset_scale) * u.deg
    current_crota = extract_crota_from_wcs(input_data.wcs)
    new_pc = calculate_pc_matrix(current_crota + pc_offset, input_data.wcs.wcs.cdelt)
    input_data.wcs.wcs.pc = new_pc

    return input_data


@task
def generate_l0_pmzp(input_file: NDCube,
                     path_output: str,
                     psf_model: ArrayCorrector,
                     wfi_vignetting_model_path: str,
                     nfi_vignetting_model_path: str) -> None:
    """Generate level 0 polarized synthetic data."""
    input_data = load_ndcube_from_fits(input_file)

    # Define the output data product
    product_code = input_data.meta["TYPECODE"].value + input_data.meta["OBSCODE"].value
    product_level = "0"
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta["DATE-OBS"] = str(input_data.meta.datetime)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(input_data.wcs)
    for key in output_header:
        if (key in input_data.meta) and output_header[key] == "" and key not in ("COMMENT", "HISTORY"):
            output_meta[key] = input_data.meta[key].value

    input_data = NDCube(data=input_data.data, meta=output_meta, wcs=input_data.wcs)
    output_data = starfield_misalignment(input_data)

    output_data = uncorrect_psf(output_data, psf_model)

    # TODO - look for stray light model from WFI folks? Or just use some kind of gradient with poisson noise.
    output_data = add_stray_light(output_data)

    output_data = add_deficient_pixels(output_data)

    output_data = uncorrect_vignetting(output_data, wfi_vignetting_model_path, nfi_vignetting_model_path)

    output_data = apply_streaks(output_data)

    output_data = perform_photometric_uncalibration(output_data)

    if input_data.meta["OBSCODE"].value == "4":
        scaling = {"gain": 4.9 * u.photon / u.DN,
              "wavelength": 530. * u.nm,
              "exposure": 49 * u.s,
              "aperture": 49.57 * u.mm**2}
    else:
        scaling = {"gain": 4.9 * u.photon / u.DN,
              "wavelength": 530. * u.nm,
              "exposure": 49 * u.s,
              "aperture": 34 * u.mm ** 2}
    output_data.data[:, :] = msb_to_dn(output_data.data[:, :], output_data.wcs, **scaling)

    noise = compute_noise(output_data.data)
    output_data.data[...] += noise[...]

    output_data, spike_image = add_spikes(output_data)

    output_data.data[:, :] = encode_sqrt(output_data.data[:, :], to_bits=10)

    # TODO - Sync up any final header data here

    # Set output dtype
    # TODO - also check this in the output data w/r/t BITPIX
    output_data.data[output_data.data > 2**10-1] = 2**10-1
    write_data = NDCube(data=output_data.data[:, :].astype(np.int32),
                        uncertainty=None,
                        meta=output_data.meta,
                        wcs=output_data.wcs)
    write_data = update_spacecraft_location(write_data, write_data.meta.astropy_time)

    # Write out
    output_data.meta["FILEVRSN"] = "1"
    write_ndcube_to_fits(write_data, path_output + get_base_file_name(output_data) + ".fits")
    fits.writeto(path_output + get_base_file_name(output_data) + "_spike.fits", spike_image, overwrite=True)


@task
def generate_l0_cr(input_file: NDCube, path_output: str,
                   psf_model: ArrayCorrector,
                   wfi_vignetting_model_path: str, nfi_vignetting_model_path: str) -> None:
    """Generate level 0 polarized synthetic data."""
    input_data = load_ndcube_from_fits(input_file)

    # Define the output data product
    product_code = input_data.meta["TYPECODE"].value + input_data.meta["OBSCODE"].value
    product_level = "0"
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta["DATE-OBS"] = str(input_data.meta.datetime)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(input_data.wcs)
    for key in output_header:
        if (key in input_data.meta) and output_header[key] == "" and key not in ("COMMENT", "HISTORY"):
            output_meta[key] = input_data.meta[key].value

    input_data = NDCube(data=input_data.data, meta=output_meta, wcs=input_data.wcs)

    output_data = starfield_misalignment(input_data)

    output_data = uncorrect_psf(output_data, psf_model)

    # TODO - look for stray light model from WFI folks? Or just use some kind of gradient with poisson noise.
    output_data = add_stray_light(output_data)

    output_data = add_deficient_pixels(output_data)

    output_data = uncorrect_vignetting(output_data, wfi_vignetting_model_path, nfi_vignetting_model_path)

    output_data = apply_streaks(output_data)

    output_data = perform_photometric_uncalibration(output_data)

    if input_data.meta["OBSCODE"].value == "4":
        scaling = {"gain": 4.9 * u.photon / u.DN,
              "wavelength": 530. * u.nm,
              "exposure": 49 * u.s,
              "aperture": 49.57 * u.mm**2}
    else:
        scaling = {"gain": 4.9 * u.photon / u.DN,
              "wavelength": 530. * u.nm,
              "exposure": 49 * u.s,
              "aperture": 34 * u.mm ** 2}
    output_data.data[:, :] = msb_to_dn(output_data.data[:, :], output_data.wcs, **scaling)

    noise = compute_noise(output_data.data)
    output_data.data[...] += noise[...]

    output_data, spike_image = add_spikes(output_data)

    # TODO - Sync up any final header data here

    # Set output dtype
    # TODO - also check this in the output data w/r/t BITPIX
    output_data.data[:, :] = encode_sqrt(output_data.data[:, :], to_bits=10)

    output_data.data[output_data.data > 2**10-1] = 2**10-1
    write_data = NDCube(data=output_data.data[:, :].astype(np.int32),
                        uncertainty=None,
                        meta=output_data.meta,
                        wcs=output_data.wcs)
    write_data = update_spacecraft_location(write_data, write_data.meta.astropy_time)

    # Write out
    output_data.meta["FILEVRSN"] = "1"
    write_ndcube_to_fits(write_data, path_output + get_base_file_name(output_data) + ".fits")
    fits.writeto(path_output + get_base_file_name(output_data) + "_spike.fits", spike_image, overwrite=True)


@flow(log_prints=True,
      task_runner=DaskTaskRunner(cluster_kwargs={"n_workers": 8, "threads_per_worker": 2},
))
def generate_l0_all(datadir: str, psf_model_path: str,
                    wfi_vignetting_model_path: str, nfi_vignetting_model_path: str) -> None:
    """Generate all level 0 synthetic data."""
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, "synthetic_l0/")
    os.makedirs(outdir, exist_ok=True)
    print(f"Outputting to {outdir}")

    # Parse list of level 1 model data
    files_l1 = glob.glob(datadir + "/synthetic_l1/*L1_P*.fits")
    files_cr = glob.glob(datadir + "/synthetic_l1/*CR*.fits")
    print(f"Generating based on {len(files_l1)} files.")
    files_l1.sort()

    psf_model = ArrayCorrector.load(psf_model_path)

    futures = []
    for file_l1 in tqdm(files_l1, total=len(files_l1)):
        futures.append(generate_l0_pmzp.submit(file_l1, outdir, psf_model, # noqa: PERF401
                                  wfi_vignetting_model_path, nfi_vignetting_model_path))

    for file_cr in tqdm(files_cr, total=len(files_cr)):
        futures.append(generate_l0_cr.submit(file_cr, outdir, psf_model,  # noqa: PERF401
                                    wfi_vignetting_model_path, nfi_vignetting_model_path))
    wait(futures)
