"""
Generates synthetic level 0 data
"""
import glob
import os
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.nddata import StdDevUncertainty
from ndcube import NDCube
from prefect import flow, task
from prefect.futures import wait
from prefect_dask import DaskTaskRunner
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            load_ndcube_from_fits, write_ndcube_to_fits)
from punchbowl.data.units import msb_to_dn
from punchbowl.level1.initial_uncertainty import compute_noise
from punchbowl.level1.sqrt import encode_sqrt
from regularizepsf import ArrayCorrector
from tqdm import tqdm

from simpunch.util import update_spacecraft_location


def certainty_estimate(input_data, noise_level):
    input_data.uncertainty = StdDevUncertainty(noise_level)
    return input_data


def photometric_uncalibration(input_data,
                              gain: float = 4.93,
                              bitrate_signal: int = 16):

    # # Input data should be in calibrated units of MSB
    # # First establish a conversion rate backwards to photons
    #
    # msb_def = 2.0090000E7 * u.W / u.m**2 / u.sr
    #
    # # Constants
    # wavelength = 530 * u.nm  # Wavelength in nanometers
    # exposure_time = 49 * u.s  # Exposure in seconds
    #
    # # Calculate energy of a single photon
    # energy_per_photon = (const.h * const.c / wavelength).to(u.J) / u.photon
    #
    # # Get the pixel scaling
    # pixel_scale = (proj_plane_pixel_area(input_data.wcs) * u.deg**2).to(u.sr) / u.pixel
    # # pixel_scale = (abs(input_data.wcs.pixel_scale_matrix[0, 0]) * u.deg**2).to(u.sr) / u.pixel
    #
    # # The instrument aperture is 22mm,
    # however the effective area is a bit smaller due to losses and quantum efficiency
    # aperture = 49.57 * u.mm**2
    # # aperture = 0.392699082 * u.cm**2
    #
    # # Calculate the photon count per pixel
    # photons_per_pixel = (msb_def / energy_per_photon * aperture * pixel_scale * exposure_time).decompose()
    #
    # # Convert the input data to a photon count
    # photon_data = input_data.data * photons_per_pixel.value
    #
    # # Now using the gain convert to DN
    # dn_data = photon_data / gain
    #
    # # TODO - Perhaps the stars at night are a bit too bright. Check the scaling earlier in the pipeline.
    # # Clip / scale the data at the signal bitrate
    # dn_data = np.clip(dn_data, 0, 2**bitrate_signal-1)
    #
    # dn_data = dn_data.astype(int)
    #
    # input_data.data[:, :] = dn_data

    return input_data


def spiking(input_data, spike_scaling=2**16-5_000):
    spike_index = np.random.choice(input_data.data.shape[0] * input_data.data.shape[1],
                                   np.random.randint(40*49-1000, 40*49+1000))
    spike_index2d = np.unravel_index(spike_index, input_data.data.shape)

    spike_values = np.random.normal(spike_scaling,
                                    spike_scaling * 0.01,
                                    len(spike_index))
    spike_values = np.clip(spike_values, 0, spike_scaling)

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

    input_data.data[:, :] = streak_matrix @ input_data.data[:, :] / exposure_time

    return input_data


def uncorrect_vignetting_lff(input_data, wfi_vignetting_model_path, nfi_vignetting_model_path):
    if int(input_data.meta['OBSCODE'].value) == 4:
        vignetting_function_path = Path(nfi_vignetting_model_path)
    else:
        vignetting_function_path = Path(wfi_vignetting_model_path)
    cube = load_ndcube_from_fits(vignetting_function_path)

    input_data.data[:, :] *= cube.data[:, :]

    return input_data


def add_deficient_pixels(input_data):
    return input_data


def add_stray_light(input_data):
    return input_data


def uncorrect_psf(input_data, psf_model):
    input_data.data[...] = psf_model.correct_image(input_data.data, alpha=3.0, epsilon=0.3)[...]
    return input_data


def starfield_misalignment(input_data, cr_offset_scale: float = 0.1, pc_offset_scale: float = 0.1):
    # TODO - Removed temporarily for getting the pipeline moving
    #cr_offsets = np.random.normal(0, cr_offset_scale, 2)
    #input_data.wcs.wcs.crval = input_data.wcs.wcs.crval + cr_offsets

    #pc_offset = np.random.normal(0, pc_offset_scale) * u.deg
    #current_crota = extract_crota_from_wcs(input_data.wcs)
    #new_pc = calculate_pc_matrix(current_crota + pc_offset, input_data.wcs.wcs.cdelt)
    #input_data.wcs.wcs.pc = new_pc

    return input_data


@task
def generate_l0_pmzp(input_file, path_output, psf_model, wfi_vignetting_model_path, nfi_vignetting_model_path):
    """Generates level 0 polarized synthetic data"""

    input_data = load_ndcube_from_fits(input_file)

    # Define the output data product
    product_code = input_data.meta['TYPECODE'].value + input_data.meta['OBSCODE'].value
    product_level = '0'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta['DATE-OBS'] = str(input_data.meta.datetime)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(input_data.wcs)
    for key in output_header.keys():
        if (key in input_data.meta) and output_header[key] == '' and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key] = input_data.meta[key].value

    # input_data = NDCube(data=input_data.data+1E-13, meta=output_meta, wcs=input_data.wcs)
    input_data = NDCube(data=input_data.data, meta=output_meta, wcs=input_data.wcs)

    # TODO - fold into reprojection?
    output_data = starfield_misalignment(input_data)

    output_data = uncorrect_psf(output_data, psf_model)

    # TODO - look for stray light model from WFI folks? Or just use some kind of gradient with poisson noise.
    output_data = add_stray_light(output_data)

    output_data = add_deficient_pixels(output_data)

    output_data = uncorrect_vignetting_lff(output_data, wfi_vignetting_model_path, nfi_vignetting_model_path)

    output_data = streaking(output_data)

    output_data = photometric_uncalibration(output_data)

    if input_data.meta['OBSCODE'].value == "4":
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

    output_data = spiking(output_data)

    output_data = certainty_estimate(output_data, noise)  # TODO: shouldn't certainty take into account spikes?

    output_data.data[:, :] = encode_sqrt(output_data.data[:, :])

    # TODO - Sync up any final header data here

    # Set output dtype
    # TODO - also check this in the output data w/r/t BITPIX
    output_data.data[output_data.data > 2**16-1] = 2**16-1
    write_data = NDCube(data=output_data.data[:, :].astype(np.int32),
                        uncertainty=None,
                        meta=output_data.meta,
                        wcs=output_data.wcs)
    write_data = update_spacecraft_location(write_data, write_data.meta.astropy_time)

    # Write out
    output_data.meta['FILEVRSN'] = '1'
    write_ndcube_to_fits(write_data, path_output + get_base_file_name(output_data) + '.fits')

@task
def generate_l0_cr(input_file, path_output, psf_model, wfi_vignetting_model_path, nfi_vignetting_model_path):
    """Generates level 0 polarized synthetic data"""

    input_data = load_ndcube_from_fits(input_file)

    # Define the output data product
    product_code = input_data.meta['TYPECODE'].value + input_data.meta['OBSCODE'].value
    product_level = '0'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta['DATE-OBS'] = str(input_data.meta.datetime)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(input_data.wcs)
    for key in output_header.keys():
        if (key in input_data.meta) and output_header[key] == '' and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key] = input_data.meta[key].value

    # input_data = NDCube(data=input_data.data+1E-13, meta=output_meta, wcs=input_data.wcs)
    input_data = NDCube(data=input_data.data, meta=output_meta, wcs=input_data.wcs)

    # TODO - fold into reprojection?
    output_data = starfield_misalignment(input_data)

    output_data = uncorrect_psf(output_data, psf_model)

    # TODO - look for stray light model from WFI folks? Or just use some kind of gradient with poisson noise.
    output_data = add_stray_light(output_data)

    output_data = add_deficient_pixels(output_data)

    output_data = uncorrect_vignetting_lff(output_data, wfi_vignetting_model_path, nfi_vignetting_model_path)

    output_data = streaking(output_data)

    output_data = photometric_uncalibration(output_data)

    if input_data.meta['OBSCODE'].value == "4":
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

    output_data = spiking(output_data)

    output_data = certainty_estimate(output_data, noise)  # TODO: shouldn't certainty take into account spikes?

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
    output_data.meta['FILEVRSN'] = '1'
    write_ndcube_to_fits(write_data, path_output + get_base_file_name(output_data) + '.fits')


@flow(log_prints=True, task_runner=DaskTaskRunner(
    cluster_kwargs={"n_workers": 8, "threads_per_worker": 2}
))
def generate_l0_all(datadir, psf_model_path, wfi_vignetting_model_path, nfi_vignetting_model_path):
    """Generate all level 0 synthetic data"""

    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, 'synthetic_l0_build4/')
    os.makedirs(outdir, exist_ok=True)
    print(f"Outputting to {outdir}")

    # Parse list of level 1 model data
    files_l1 = glob.glob(datadir + '/synthetic_l1_build4/*L1_P*.fits')
    files_cr = glob.glob(datadir + '/synthetic_l1_build4/*CR*.fits')
    print(f"Generating based on {len(files_l1)} files.")
    files_l1.sort()

    psf_model = ArrayCorrector.load(psf_model_path)

    futures = []
    # Run individual generators
    for file_l1 in tqdm(files_l1, total=len(files_l1)):
        futures.append(generate_l0_pmzp.submit(file_l1, outdir, psf_model,
                                  wfi_vignetting_model_path, nfi_vignetting_model_path))

    for file_cr in tqdm(files_cr, total=len(files_cr)):
        futures.append(generate_l0_cr.submit(file_cr, outdir, psf_model,
                                    wfi_vignetting_model_path, nfi_vignetting_model_path))

    wait(futures)

if __name__ == '__main__':
    generate_l0_all("/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/",
                    "/Users/jhughes/Desktop/repos/simpunch/synthetic_input_psf.h5",
                    "/Users/jhughes/Desktop/repos/simpunch/PUNCH_L1_GM1_20240817174727_v2.fits",
                    "/Users/jhughes/Desktop/repos/simpunch/PUNCH_L1_GM4_20240819045110_v1.fits")
