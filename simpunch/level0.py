"""Generate synthetic level 0 data."""
import copy
import os
import warnings
from pathlib import Path
from random import random

import astropy.units as u
import astropy.wcs
import numpy as np
from ndcube import NDCube
from prefect import get_run_logger, task
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            load_ndcube_from_fits, write_ndcube_to_fits)
from punchbowl.data.units import msb_to_dn
from punchbowl.data.wcs import calculate_pc_matrix, extract_crota_from_wcs
from punchbowl.level1.initial_uncertainty import compute_noise
from punchbowl.level1.sqrt import encode_sqrt
from regularizepsf import ArrayPSFTransform

from simpunch.spike import generate_spike_image
from simpunch.util import (fill_metadata_defaults, generate_stray_light,
                           get_subdirectory, update_spacecraft_location,
                           write_array_to_fits)


def perform_photometric_uncalibration(input_data: NDCube, coefficient_array: np.ndarray) -> NDCube:
    """Undo quartic fit calibration."""
    num_coefficients = coefficient_array.shape[0]
    new_data = np.nansum(
        [coefficient_array[i] * np.power(input_data.data, num_coefficients - i - 1)
         for i in range(num_coefficients)], axis=0)
    input_data.data[...] = new_data
    return input_data


def add_spikes(input_data: NDCube) -> tuple[NDCube, np.ndarray]:
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
                  readout_line_time: float = 163 / 2148,
                  reset_line_time: float = 163 / 2148) -> NDCube:
    """Apply the streak matrix to the image."""
    streak_matrix = create_streak_matrix(input_data.data.shape[0],
                                         exposure_time, readout_line_time, reset_line_time)
    input_data.data[:, :] = streak_matrix @ input_data.data / exposure_time
    return input_data


def add_deficient_pixels(input_data: NDCube) -> NDCube:
    """Add deficient pixels to the image."""
    return input_data


def add_stray_light(input_data: NDCube,
                    inst: str = "WFI",
                    polar: str = "mzp") -> NDCube:
    """Add stray light to the image."""
    straydata = generate_stray_light(input_data.data.shape, instrument=inst, pstate="pb" if polar == "mzp" else "b")
    input_data.data[:, :] += straydata
    return input_data


def uncorrect_psf(input_data: NDCube, psf_model: ArrayPSFTransform) -> NDCube:
    """Apply an inverse PSF to an image."""
    input_data.data[...] = psf_model.apply(input_data.data)
    return input_data


def add_transients(input_data: NDCube,
                   transient_area: int = 600 ** 2,
                   transient_probability: float = 0.03,
                   transient_brightness_range: tuple[float, float] = (0.6, 0.8)) -> NDCube:
    """Add a block of brighter transient data to simulate aurora."""
    transient_image = np.zeros_like(input_data.data)
    if random() < transient_probability:
        width = int(np.sqrt(transient_area) * random())
        height = int(transient_area / width)
        i, j = int(random() * input_data.data.shape[0]), int(random() * input_data.data.shape[1])
        transient_brightness = np.random.uniform(transient_brightness_range[0], transient_brightness_range[1])
        transient_value = np.mean(input_data.data[i:i + width, j:j + height]) * transient_brightness
        input_data.data[i:i + width, j:j + height] += transient_value
        transient_image[i:i + width, j:j + height] = transient_value
    return input_data, transient_image


def starfield_misalignment(input_data: NDCube,
                           cr_offset_scale: float = 0.1,
                           pc_offset_scale: float = 0.1) -> NDCube:
    """Offset the pointing in an image to simulate spacecraft uncertainty."""
    original_wcs = copy.deepcopy(input_data.wcs)
    cr_offsets = np.random.normal(0, cr_offset_scale, 2)
    input_data.wcs.wcs.crval = input_data.wcs.wcs.crval + cr_offsets

    pc_offset = np.random.normal(0, pc_offset_scale) * u.deg
    current_crota = extract_crota_from_wcs(input_data.wcs)
    new_pc = calculate_pc_matrix(current_crota + pc_offset, input_data.wcs.wcs.cdelt)
    input_data.wcs.wcs.pc = new_pc

    return input_data, original_wcs

def apply_mask(input_data: NDCube) -> NDCube:
    """Apply the appropriate instrument mask to a NDCube."""
    this_directory = Path(__file__).parent.resolve()
    if input_data.meta["OBSCODE"].value == "4":
        path = this_directory / "data" / "imt_nfi.bin"
    else:
        path = this_directory / "data" / "imt_wfi.bin"

    with open(path, "rb") as f:
        data = f.read()
    mask = np.unpackbits(np.frombuffer(data, dtype=np.uint8)).reshape(2048, 2048)
    input_data.data[np.logical_not(mask)] = 0
    return input_data

@task
def generate_l0_pmzp(input_file: str,
                     path_output: str,
                     psf_model_path: str,  # ArrayPSFTransform,
                     wfi_quartic_coeffs_path: str,  # np.ndarray,
                     nfi_quartic_coeffs_path: str,  # np.ndarray,
                     transient_probability: float = 0.03,
                     shift_pointing: bool = False) -> str:
    """Generate level 0 polarized synthetic data."""
    logger = get_run_logger()
    input_data = load_ndcube_from_fits(input_file)
    logger.info(f"Read input file {input_file}")
    psf_model = ArrayPSFTransform.load(Path(psf_model_path))
    logger.info("PSF model loaded")
    wfi_quartic_coefficients = load_ndcube_from_fits(wfi_quartic_coeffs_path, include_provenance=False).data
    nfi_quartic_coefficients = load_ndcube_from_fits(nfi_quartic_coeffs_path, include_provenance=False).data
    logger.info("Quartic coefficients loaded loaded")

    # Define the output data product
    product_code = input_data.meta["TYPECODE"].value + input_data.meta["OBSCODE"].value
    product_level = "0"
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    fill_metadata_defaults(output_meta)

    output_meta["DATE-OBS"] = input_data.meta.datetime.isoformat()

    quartic_coefficients = wfi_quartic_coefficients \
        if input_data.meta["OBSCODE"].value != "4" else nfi_quartic_coefficients

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(input_data.wcs)
    for key in output_header:
        if (key in input_data.meta) and output_header[key] == "" and key not in ("COMMENT", "HISTORY"):
            output_meta[key] = input_data.meta[key].value

    input_data = NDCube(data=input_data.data, meta=output_meta, wcs=input_data.wcs)
    if shift_pointing:
        output_data, original_wcs = starfield_misalignment(input_data)
        logger.info("Pointing shifted")
    else:
        output_data = input_data
        original_wcs = input_data.wcs.copy()
    output_data, transient = add_transients(output_data, transient_probability=transient_probability)
    logger.info("Transients added")
    output_data = uncorrect_psf(output_data, psf_model)
    logger.info("Beautiful PSF ruined")

    inst = "WFI" if input_data.meta["OBSCODE"].value != "4" else "NFI"

    output_data = add_stray_light(output_data, inst = inst, polar="mzp")
    logger.info("Stray light added")
    output_data = add_deficient_pixels(output_data)
    logger.info("Pixels broken")
    output_data = apply_streaks(output_data)
    logger.info("Streaks added")
    output_data = apply_mask(output_data)
    logger.info("Mask applied")
    output_data = perform_photometric_uncalibration(output_data, quartic_coefficients)
    logger.info("Photometry scrambled")

    if input_data.meta["OBSCODE"].value == "4":
        scaling = {"gain_left": input_data.meta['GAINLEFT'].value * u.photon / u.DN,
                   "gain_right": input_data.meta['GAINRGHT'].value * u.photon / u.DN,
                   "wavelength": 530. * u.nm,
                   "exposure": input_data.meta['EXPTIME'].value * u.s,
                   "aperture": 49.57 * u.mm ** 2}
    else:
        scaling = {"gain_left": input_data.meta['GAINLEFT'].value * u.photon / u.DN,
                   "gain_right": input_data.meta['GAINRGHT'].value * u.photon / u.DN,
                   "wavelength": 530. * u.nm,
                   "exposure": input_data.meta['EXPTIME'].value * u.s,
                   "aperture": 34 * u.mm ** 2}
    output_data.data[:, :] = msb_to_dn(
        output_data.data[:, :], output_data.wcs, **scaling, pixel_area_stride=3)
    logger.info("Units scaled")

    data, noise = compute_noise(output_data.data, bias_level=output_data.meta['OFFSET'].value)
    output_data.data[...] = data + noise
    logger.info("Noise added")

    output_data, spike_image = add_spikes(output_data)
    logger.info("Spikes added")

    output_data.data[:, :] = encode_sqrt(output_data.data[:, :], to_bits=10)
    logger.info("Sqrt encoded")
    output_data = apply_mask(output_data)
    logger.info("Mask applied")
    # TODO - Sync up any final header data here

    # Set output dtype
    # TODO - also check this in the output data w/r/t BITPIX
    output_data.data[output_data.data > 2 ** 10 - 1] = 2 ** 10 - 1
    output_data.meta["DESCRPTN"] = "Simulated " + output_data.meta["DESCRPTN"].value
    output_data.meta["TITLE"] = "Simulated " + output_data.meta["TITLE"].value

    write_data = NDCube(data=output_data.data[:, :].astype(np.int32),
                        uncertainty=None,
                        meta=output_data.meta,
                        wcs=output_data.wcs)
    write_data = update_spacecraft_location(write_data, write_data.meta.astropy_time)

    # Write out
    output_data.meta["FILEVRSN"] = "1"
    out_dir = os.path.join(path_output, get_subdirectory(output_data))
    os.makedirs(out_dir, exist_ok=True)
    basename = get_base_file_name(output_data)

    main_output_path = os.path.join(out_dir, basename + ".fits")
    logger.info(f"Writing {main_output_path}")
    write_ndcube_to_fits(write_data, main_output_path)

    path = os.path.join(out_dir, basename + "_spike.fits")
    logger.info(f"Writing {path}")
    write_array_to_fits(path, spike_image)

    path = os.path.join(out_dir, basename + "_transient.fits")
    logger.info(f"Writing {path}")
    write_array_to_fits(path, transient)

    path = os.path.join(out_dir, basename + "_original_wcs.txt")
    logger.info(f"Writing {path}")
    original_wcs.to_header().tofile(path)

    logger.info("All data written")
    return main_output_path

@task
def generate_l0_cr(input_file: str, path_output: str,
                   psf_model_path: str,  # ArrayPSFTransform,
                   wfi_quartic_coeffs_path: str,  # np.ndarray,
                   nfi_quartic_coeffs_path: str,  # np.ndarray,
                   transient_probability: float = 0.03,
                   shift_pointing: bool = False) -> str:
    """Generate level 0 clear synthetic data."""
    logger = get_run_logger()
    input_data = load_ndcube_from_fits(input_file)
    logger.info(f"Read input file {input_file}")
    psf_model = ArrayPSFTransform.load(Path(psf_model_path))
    logger.info("PSF model loaded")
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=astropy.wcs.FITSFixedWarning,
                                message=r".*[A-Z]*_OBS.*\n.*a floating-point value was expected.*")
        wfi_quartic_coefficients = load_ndcube_from_fits(wfi_quartic_coeffs_path, include_provenance=False).data
        nfi_quartic_coefficients = load_ndcube_from_fits(nfi_quartic_coeffs_path, include_provenance=False).data
    logger.info("Quartic coefficients loaded")

    # Define the output data product
    product_code = input_data.meta["TYPECODE"].value + input_data.meta["OBSCODE"].value
    product_level = "0"
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    fill_metadata_defaults(output_meta)
    output_meta["DATE-OBS"] = input_data.meta.datetime.isoformat()

    quartic_coefficients = wfi_quartic_coefficients \
        if input_data.meta["OBSCODE"].value != "4" else nfi_quartic_coefficients

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(input_data.wcs)
    for key in output_header:
        if (key in input_data.meta) and output_header[key] == "" and key not in ("COMMENT", "HISTORY"):
            output_meta[key] = input_data.meta[key].value

    input_data = NDCube(data=input_data.data, meta=output_meta, wcs=input_data.wcs)
    if shift_pointing:
        output_data, original_wcs = starfield_misalignment(input_data)
        logger.info("Pointing shifted")
    else:
        output_data = input_data
        original_wcs = input_data.wcs.copy()
    inst = "WFI" \
        if input_data.meta["OBSCODE"].value != "4" else "NFI"
    output_data, transient = add_transients(output_data, transient_probability=transient_probability)
    logger.info("Transients added")
    output_data = uncorrect_psf(output_data, psf_model)
    logger.info("Beautiful PSF ruined")
    output_data = add_stray_light(output_data, inst=inst, polar="clear")
    logger.info("Stray light added")
    output_data = add_deficient_pixels(output_data)
    logger.info("Pixels broken")
    output_data = apply_streaks(output_data)
    logger.info("Streaks added")
    output_data = apply_mask(output_data)
    logger.info("Mask applied")
    output_data = perform_photometric_uncalibration(output_data, quartic_coefficients)
    logger.info("Photometry scrambled")

    if input_data.meta["OBSCODE"].value == "4":
        scaling = {"gain_left": input_data.meta['GAINLEFT'].value * u.photon / u.DN,
                   "gain_right": input_data.meta['GAINRGHT'].value * u.photon / u.DN,
                   "wavelength": 530. * u.nm,
                   "exposure": input_data.meta['EXPTIME'].value * u.s,
                   "aperture": 49.57 * u.mm ** 2}
    else:
        scaling = {"gain_left": input_data.meta['GAINLEFT'].value * u.photon / u.DN,
                   "gain_right": input_data.meta['GAINRGHT'].value * u.photon / u.DN,
                   "wavelength": 530. * u.nm,
                   "exposure": input_data.meta['EXPTIME'].value * u.s,
                   "aperture": 34 * u.mm ** 2}
    output_data.data[:, :] = msb_to_dn(
        output_data.data[:, :], output_data.wcs, **scaling, pixel_area_stride=3)
    logger.info("Units scaled")

    data, noise = compute_noise(output_data.data, bias_level=output_data.meta['OFFSET'].value)
    output_data.data[...] = data + noise
    logger.info("Noise added")

    output_data, spike_image = add_spikes(output_data)
    logger.info("Spikes added")

    output_data.data[:, :] = encode_sqrt(output_data.data[:, :], to_bits=10)
    output_data.meta["ISSQRT"] = True
    logger.info("Sqrt encoded")
    output_data = apply_mask(output_data)
    logger.info("Mask applied")

    output_data.data[output_data.data > 2 ** 10 - 1] = 2 ** 10 - 1
    output_data.meta["DESCRPTN"] = "Simulated " + output_data.meta["DESCRPTN"].value
    output_data.meta["TITLE"] = "Simulated " + output_data.meta["TITLE"].value

    write_data = NDCube(data=output_data.data[:, :].astype(np.int32),
                        uncertainty=None,
                        meta=output_data.meta,
                        wcs=output_data.wcs)
    write_data = update_spacecraft_location(write_data, write_data.meta.astropy_time)

    # Write out
    output_data.meta["FILEVRSN"] = "1"
    out_dir = os.path.join(path_output, get_subdirectory(output_data))
    os.makedirs(out_dir, exist_ok=True)
    basename = get_base_file_name(output_data)

    main_output_path = os.path.join(out_dir, basename + ".fits")
    logger.info(f"Writing {main_output_path}")
    write_ndcube_to_fits(write_data, main_output_path)

    path = os.path.join(out_dir, basename + "_spike.fits")
    logger.info(f"Writing {path}")
    write_array_to_fits(path, spike_image)

    path = os.path.join(out_dir, basename + "_transient.fits")
    logger.info(f"Writing {path}")
    write_array_to_fits(path, transient)

    path = os.path.join(out_dir, basename + "_original_wcs.txt")
    logger.info(f"Writing {path}")
    original_wcs.to_header().tofile(path)

    logger.info("All data written")
    return main_output_path
