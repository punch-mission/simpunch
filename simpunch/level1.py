"""Generates synthetic level 1 data."""
import copy
import os
from math import floor

import astropy.units as u
import numpy as np
import reproject
import solpolpy
from astropy.io import fits
from astropy.modeling.functional_models import Gaussian2D
from astropy.table import QTable
from astropy.wcs import WCS, DistortionLookupTable
from ndcube import NDCollection, NDCube
from photutils.datasets import make_model_image, make_noise_image
from prefect import get_run_logger, task
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            load_ndcube_from_fits, write_ndcube_to_fits)
from punchbowl.data.wcs import (calculate_celestial_wcs_from_helio,
                                calculate_pc_matrix, get_p_angle)

from simpunch.stars import (filter_for_visible_stars, find_catalog_in_image,
                            load_raw_hipparcos_catalog)
from simpunch.util import (fill_metadata_defaults, get_subdirectory,
                           update_spacecraft_location)

CURRENT_DIR = os.path.dirname(__file__)


def generate_spacecraft_wcs(spacecraft_id: str, rotation_stage: int) -> WCS:
    """Generate the spacecraft world coordinate system."""
    angle_step = 30

    if spacecraft_id in ["1", "2", "3"]:
        if spacecraft_id == "1":
            angle_wfi = (0 + angle_step * rotation_stage) % 360 * u.deg
        elif spacecraft_id == "2":
            angle_wfi = (120 + angle_step * rotation_stage) % 360 * u.deg
        elif spacecraft_id == "3":
            angle_wfi = (240 + angle_step * rotation_stage) % 360 * u.deg

        out_wcs_shape = [2048, 2048]
        out_wcs = WCS(naxis=2)

        out_wcs.wcs.crpix = out_wcs_shape[1] / 2 - 0.5, out_wcs_shape[0] / 2 - 0.5
        out_wcs.wcs.crval = (24.75 * np.sin(angle_wfi), 24.75 * np.cos(angle_wfi))
        out_wcs.wcs.cdelt = 0.02, 0.02
        out_wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
        out_wcs.wcs.cunit = "deg", "deg"
        out_wcs.wcs.pc = calculate_pc_matrix(360 * u.deg - angle_wfi, out_wcs.wcs.cdelt)
        out_wcs.wcs.set_pv([(2, 1, 0.0)])

    elif spacecraft_id == "4":
        angle_nfi = (0 + angle_step * rotation_stage) % 360 * u.deg
        out_wcs_shape = [2048, 2048]
        out_wcs = WCS(naxis=2)
        out_wcs.wcs.crpix = out_wcs_shape[1] / 2 + 0.5, out_wcs_shape[0] / 2 + 0.5
        out_wcs.wcs.crval = 0, 0
        out_wcs.wcs.cdelt = 30 / 3600, 30 / 3600

        out_wcs.wcs.pc = calculate_pc_matrix(angle_nfi, out_wcs.wcs.cdelt)

        out_wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        out_wcs.wcs.cunit = "deg", "deg"
    else:
        msg = "Invalid spacecraft_id."
        raise ValueError(msg)

    return out_wcs


def deproject_polar(input_data: NDCube, output_wcs: WCS, adaptive_reprojection: bool = False) -> tuple[NDCube, WCS]:
    """Deproject a polarized image."""
    reconstructed_wcs = WCS(naxis=3)
    reconstructed_wcs.wcs.ctype = input_data.wcs.wcs.ctype
    reconstructed_wcs.wcs.cunit = input_data.wcs.wcs.cunit
    reconstructed_wcs.wcs.cdelt = input_data.wcs.wcs.cdelt
    reconstructed_wcs.wcs.crpix = input_data.wcs.wcs.crpix
    reconstructed_wcs.wcs.crval = input_data.wcs.wcs.crval
    reconstructed_wcs.wcs.pc = input_data.wcs.wcs.pc

    reconstructed_wcs = calculate_celestial_wcs_from_helio(reconstructed_wcs,
                                                           input_data.meta.astropy_time,
                                                           input_data.data.shape)
    reconstructed_wcs = reconstructed_wcs.dropaxis(2)

    output_wcs_helio = copy.deepcopy(output_wcs)
    output_wcs = calculate_celestial_wcs_from_helio(output_wcs,
                                                    input_data.meta.astropy_time,
                                                    input_data.data.shape).dropaxis(2)

    if adaptive_reprojection:
        reprojected_data = reproject.reproject_adaptive((input_data.data,
                                                                  reconstructed_wcs),
                                                                 output_wcs,
                                                                 (2048, 2048),
                                                                 roundtrip_coords=False, return_footprint=False,
                                                                 kernel="Gaussian", boundary_mode="ignore")
    else:
        reprojected_data = reproject.reproject_interp((input_data.data,
                                                                reconstructed_wcs),
                                                               output_wcs, (2048, 2048),
                                                               roundtrip_coords=False, return_footprint=False)

    reprojected_data[np.isnan(reprojected_data)] = 0

    return NDCube(data=reprojected_data, wcs=output_wcs_helio, meta=input_data.meta), output_wcs_helio


def deproject_clear(input_data: NDCube, output_wcs: WCS, adaptive_reprojection: bool = False) -> tuple[NDCube, WCS]:
    """Deproject a clear image."""
    reconstructed_wcs = WCS(naxis=2)
    reconstructed_wcs.wcs.ctype = input_data.wcs.wcs.ctype
    reconstructed_wcs.wcs.cunit = input_data.wcs.wcs.cunit
    reconstructed_wcs.wcs.cdelt = input_data.wcs.wcs.cdelt
    reconstructed_wcs.wcs.crpix = input_data.wcs.wcs.crpix
    reconstructed_wcs.wcs.crval = input_data.wcs.wcs.crval
    reconstructed_wcs.wcs.pc = input_data.wcs.wcs.pc

    reconstructed_wcs = calculate_celestial_wcs_from_helio(reconstructed_wcs,
                                                           input_data.meta.astropy_time,
                                                           input_data.data.shape)

    output_wcs_helio = copy.deepcopy(output_wcs)
    output_wcs = calculate_celestial_wcs_from_helio(output_wcs,
                                                    input_data.meta.astropy_time,
                                                    input_data.data.shape)

    if adaptive_reprojection:
        reprojected_data= reproject.reproject_adaptive((input_data.data,
                                                               reconstructed_wcs),
                                                              output_wcs,
                                                              (2048, 2048),
                                                              roundtrip_coords=False, return_footprint=False,
                                                              kernel="Gaussian", boundary_mode="ignore")
    else:
        reprojected_data = reproject.reproject_interp((input_data.data,
                                                             reconstructed_wcs),
                                                            output_wcs, (2048, 2048),
                                                            roundtrip_coords=False, return_footprint=False)

    reprojected_data[np.isnan(reprojected_data)] = 0

    return NDCube(data=reprojected_data, wcs=output_wcs_helio, meta=input_data.meta), output_wcs_helio


def mark_quality(input_data: NDCube) -> NDCube:
    """Mark the quality of image patches."""
    return input_data


def remix_polarization(input_data: NDCube) -> NDCube:
    """Remix polarization from (M, Z, P) to (P1, P2, P3) using solpolpy."""
    # Unpack data into a NDCollection object
    w = WCS(naxis=2)
    data_collection = NDCollection([("M", NDCube(data=input_data.data[0], wcs=w, meta={})),
                                    ("Z", NDCube(data=input_data.data[1], wcs=w, meta={})),
                                    ("P", NDCube(data=input_data.data[2], wcs=w, meta={}))])

    data_collection["M"].meta["POLAR"] = -60. * u.degree
    data_collection["Z"].meta["POLAR"] = 0. * u.degree
    data_collection["P"].meta["POLAR"] = 60. * u.degree

    # TODO - Remember that this needs to be the instrument frame MZP, not the mosaic frame
    resolved_data_collection = solpolpy.resolve(data_collection, "npol",
                                                out_angles=[-60, 0, 60] * u.deg, imax_effect=False)

    # Repack data
    data_list = []
    wcs_list = []
    uncertainty_list = []
    for key in resolved_data_collection:
        data_list.append(resolved_data_collection[key].data)
        wcs_list.append(resolved_data_collection[key].wcs)
        uncertainty_list.append(resolved_data_collection[key].uncertainty)

    # Remove alpha channel if present
    if "alpha" in resolved_data_collection:
        data_list.pop()
        wcs_list.pop()
        uncertainty_list.pop()

    # Repack into an NDCube object
    new_data = np.stack(data_list, axis=0)
    if uncertainty_list[0] is not None:  # noqa: SIM108
        new_uncertainty = np.stack(uncertainty_list, axis=0)
    else:
        new_uncertainty = None

    new_wcs = input_data.wcs.copy()

    return NDCube(data=new_data, wcs=new_wcs, uncertainty=new_uncertainty, meta=input_data.meta)


# TODO - add scaling factor
def add_distortion(input_data: NDCube) -> NDCube:
    """Add a distortion model to the WCS."""
    filename_distortion = (
        os.path.join(CURRENT_DIR, "data/distortion_NFI.fits")
        if input_data.meta["OBSCODE"].value == "4"
        else os.path.join(CURRENT_DIR, "data/distortion_WFI.fits")
    )

    with fits.open(filename_distortion) as hdul:
        err_x = hdul[1].data
        err_y = hdul[2].data

    crpix = err_x.shape[1] / 2 + 0.5, err_x.shape[0] / 2 + 0.5
    coord = input_data.wcs.pixel_to_world(crpix[0], crpix[1])
    crval = (coord.Tx.to(u.deg).value, coord.Ty.to(u.deg).value)
    cdelt = (input_data.wcs.wcs.cdelt[0] * input_data.wcs.wcs.cdelt[0] / err_x.shape[1],
             input_data.wcs.wcs.cdelt[0] * input_data.wcs.wcs.cdelt[0] / err_x.shape[1])

    cpdis1 = DistortionLookupTable(
        -err_x.astype(np.float32), crpix, crval, cdelt,
    )
    cpdis2 = DistortionLookupTable(
        -err_y.astype(np.float32), crpix, crval, cdelt,
    )

    input_data.wcs.cpdis1 = cpdis1
    input_data.wcs.cpdis2 = cpdis2

    return input_data


def generate_starfield(wcs: WCS,
                       img_shape: (int, int),
                       fwhm: float,
                       wcs_mode: str = "all",
                       mag_set: float = 0,
                       flux_set: float = 500_000,
                       noise_mean: float | None = 25.0,
                       noise_std: float | None = 5.0,
                       dimmest_magnitude: float = 8) -> (np.ndarray, QTable):
    """Generate a realistic starfield."""
    sigma = fwhm / 2.355

    catalog = load_raw_hipparcos_catalog()
    filtered_catalog = filter_for_visible_stars(catalog,
                                                dimmest_magnitude=dimmest_magnitude)
    stars = find_catalog_in_image(filtered_catalog,
                                  wcs,
                                  img_shape,
                                  mode=wcs_mode)
    star_mags = stars["Vmag"]

    sources = QTable()
    sources["x_mean"] = stars["x_pix"]
    sources["y_mean"] = stars["y_pix"]
    sources["x_stddev"] = np.ones(len(stars)) * sigma
    sources["y_stddev"] = np.ones(len(stars)) * sigma
    sources["amplitude"] = flux_set * np.power(10, -0.4 * (star_mags - mag_set))
    sources["theta"] = np.zeros(len(stars))

    model = Gaussian2D()
    model_shape = (25, 25)

    fake_image = make_model_image(img_shape, model, sources, model_shape=model_shape, x_name="x_mean", y_name="y_mean")
    if noise_mean is not None and noise_std is not None:  # we only add noise if it's specified
        fake_image += make_noise_image(img_shape, "gaussian", mean=noise_mean, stddev=noise_std)

    return fake_image, sources


def add_starfield_polarized(input_collection: NDCollection, polfactor: tuple = (0.2, 0.3, 0.5)) -> NDCollection:
    """Add synthetic polarized starfield."""
    input_data = input_collection["Z"]
    wcs_stellar_input = calculate_celestial_wcs_from_helio(input_data.wcs,
                                                           input_data.meta.astropy_time,
                                                           input_data.data.shape)

    starfield, stars = generate_starfield(wcs_stellar_input, input_data.data.shape,
                                          flux_set=100*2.0384547E-9, fwhm=3, dimmest_magnitude=12,
                                          noise_mean=None, noise_std=None)

    starfield_data = np.zeros(input_data.data.shape)
    starfield_data[:, :] = starfield * (np.logical_not(np.isclose(input_data.data, 0, atol=1E-18)))

    # Converting the input data polarization to celestial basis
    mzp_angles = ([input_cube.meta["POLAR"].value for label, input_cube in input_collection.items() if
                   label != "alpha"]) * u.degree
    cel_north_off = get_p_angle(time=input_collection["Z"].meta["DATE-OBS"].value)
    new_angles = (mzp_angles + cel_north_off).value * u.degree

    valid_keys = [key for key in input_collection if key != "alpha"]

    meta_a = dict(NormalizedMetadata.to_fits_header(input_collection[valid_keys[0]].meta,
                                                    wcs=input_collection[valid_keys[0]].wcs))
    meta_b = dict(NormalizedMetadata.to_fits_header(input_collection[valid_keys[1]].meta,
                                                    wcs=input_collection[valid_keys[1]].wcs))
    meta_c = dict(NormalizedMetadata.to_fits_header(input_collection[valid_keys[2]].meta,
                                                    wcs=input_collection[valid_keys[2]].wcs))

    meta_a["POLAR"] = meta_a["POLAR"] * u.degree
    meta_b["POLAR"] = meta_b["POLAR"] * u.degree
    meta_c["POLAR"] = meta_c["POLAR"] * u.degree

    data_collection = NDCollection(
        [(str(valid_keys[0]), NDCube(data=input_collection[valid_keys[0]].data,
                                     meta=meta_a, wcs=input_collection[valid_keys[0]].wcs)),
         (str(valid_keys[1]), NDCube(data=input_collection[valid_keys[1]].data,
                                     meta=meta_b, wcs=input_collection[valid_keys[1]].wcs)),
         (str(valid_keys[2]), NDCube(data=input_collection[valid_keys[2]].data,
                                     meta=meta_c, wcs=input_collection[valid_keys[2]].wcs))],
        aligned_axes="all")

    input_data_cel = solpolpy.resolve(data_collection, "npol", reference_angle=0 * u.degree, out_angles=new_angles)
    valid_keys = [key for key in input_data_cel if key != "alpha"]

    dummy_polarmaps = []
    for k, _ in enumerate(valid_keys):
        # Generate an all-sky polarization map for each of the three polarization states
        dummy_polarmaps.append(generate_dummy_polarization(pol_factor=polfactor[k]))
    polarmap_wcs = dummy_polarmaps[0].wcs
    dummy_polarmaps = [d.data for d in dummy_polarmaps]

    # Reproject the polarization maps in one go into the frame of the input image
    polar_rois = reproject.reproject_adaptive(
        (np.array(dummy_polarmaps), polarmap_wcs), wcs_stellar_input, input_data.data.shape,
        roundtrip_coords=False, return_footprint=False, x_cyclic=True,
        conserve_flux=True, center_jacobian=True, despike_jacobian=True)

    # Apply the polarization maps to the starfield and add them to the data
    for k, key in enumerate(valid_keys):
        input_data_cel[key].data[...] = input_data_cel[key].data + polar_rois[k] * starfield_data

    mzp_data_instru = solpolpy.resolve(input_data_cel, "mzpinstru", reference_angle=0 * u.degree)  # Instrument MZP

    valid_keys = [key for key in mzp_data_instru if key != "alpha"]
    out_meta = {"M": copy.deepcopy(input_collection["M"].meta),
                "Z": copy.deepcopy(input_collection["Z"].meta),
                "P": copy.deepcopy(input_collection["P"].meta)}
    for out_pol, meta_item in out_meta.items():
        for key, kind in zip(["POLAR", "POLARREF", "POLAROFF"], [int, str, float], strict=False):
            if isinstance(mzp_data_instru[out_pol].meta[key], u.Quantity):
                meta_item[key] = kind(mzp_data_instru[out_pol].meta[key].value)
            else:
                meta_item[key] = kind(mzp_data_instru[out_pol].meta[key])

    return NDCollection(
        [(str(key), NDCube(data=mzp_data_instru[key].data,
                           meta=out_meta[key],
                           wcs=mzp_data_instru[key].wcs)) for key in valid_keys],
        aligned_axes="all")


def add_starfield_clear(input_data: NDCube) -> NDCube:
    """Add synthetic starfield."""
    wcs_stellar_input = calculate_celestial_wcs_from_helio(input_data.wcs,
                                                           input_data.meta.astropy_time,
                                                           input_data.data.shape)

    starfield, stars = generate_starfield(wcs_stellar_input, input_data.data[:, :].shape,
                                          flux_set=0.1*2.0384547E-9,
                                          fwhm=3, dimmest_magnitude=12,
                                          noise_mean=None, noise_std=None)

    starfield_data = np.zeros(input_data.data.shape)
    starfield_data[:, :] = starfield * (np.logical_not(np.isclose(input_data.data[:, :], 0, atol=1E-18)))

    input_data.data[...] = input_data.data[...] + starfield_data

    return input_data


def generate_dummy_polarization(map_scale: float = 0.225,
                                pol_factor: float = 0.5) -> NDCube:
    """Create a synthetic polarization map."""
    shape = [floor(180 / map_scale), floor(360 / map_scale)]
    xcoord = np.linspace(-pol_factor, pol_factor, shape[1])
    ycoord = np.linspace(-pol_factor, pol_factor, shape[0])
    xin, yin = np.meshgrid(xcoord, ycoord)
    zin = pol_factor - (xin ** 2 + yin ** 2)

    wcs_sky = WCS(naxis=2)
    wcs_sky.wcs.crpix = [shape[1] / 2 + .5, shape[0] / 2 + .5]
    wcs_sky.wcs.cdelt = np.array([map_scale, map_scale])
    wcs_sky.wcs.crval = [180.0, 0.0]
    wcs_sky.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    wcs_sky.wcs.cunit = "deg", "deg"

    return NDCube(data=zin, wcs=wcs_sky)

@task
def generate_l1_pmzp(input_file: str, path_output: str, rotation_stage: int, spacecraft_id: str) -> list[str]:
    """Generate level 1 polarized synthetic data."""
    logger = get_run_logger()
    input_pdata = load_ndcube_from_fits(input_file)
    logger.info(f"Read input file {input_file}")

    # Define the output data product
    product_code = "PM" + spacecraft_id
    product_level = "1"
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    fill_metadata_defaults(output_meta)
    output_meta["DATE-OBS"] = input_pdata.meta["DATE-OBS"].value
    output_meta["DESCRPTN"] = "Simulated " + output_meta["DESCRPTN"].value
    output_meta["TITLE"] = "Simulated " + output_meta["TITLE"].value

    output_wcs = generate_spacecraft_wcs(spacecraft_id, rotation_stage)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(output_wcs)
    for key in output_header:
        if (key in input_pdata.meta) and output_header[key] == "" and key not in ("COMMENT", "HISTORY"):
            output_meta[key].value = input_pdata.meta[key].value

    output_data, output_wcs = deproject_polar(input_pdata, output_wcs)
    logger.info("Deprojected")
    output_data = mark_quality(output_data)
    logger.info("Quality marked")
    output_data = remix_polarization(output_data)
    logger.info("Polarization mixed")

    output_mmeta = copy.deepcopy(output_meta)
    output_zmeta = copy.deepcopy(output_meta)
    output_pmeta = copy.deepcopy(output_meta)

    output_mwcs = copy.deepcopy(output_wcs)
    output_zwcs = copy.deepcopy(output_wcs)
    output_pwcs = copy.deepcopy(output_wcs)

    output_mdata = NDCube(data=output_data.data[0, :, :].astype(np.float32), wcs=output_mwcs, meta=output_mmeta)
    output_zdata = NDCube(data=output_data.data[1, :, :].astype(np.float32), wcs=output_zwcs, meta=output_zmeta)
    output_pdata = NDCube(data=output_data.data[2, :, :].astype(np.float32), wcs=output_pwcs, meta=output_pmeta)

    output_mdata.meta["TYPECODE"] = "PM"
    output_zdata.meta["TYPECODE"] = "PZ"
    output_pdata.meta["TYPECODE"] = "PP"

    output_mdata.meta["POLAR"] = -60
    output_zdata.meta["POLAR"] = 0
    output_pdata.meta["POLAR"] = 60

    # Add distortion
    # output_mdata = add_distortion(output_mdata)
    # output_zdata = add_distortion(output_zdata)
    # output_pdata = add_distortion(output_pdata)
    # logger.info("Distortion added")

    output_collection = NDCollection(
        [("M", output_mdata),
         ("Z", output_zdata),
         ("P", output_pdata)],
        aligned_axes="all")

    output_mzp = add_starfield_polarized(output_collection)
    logger.info("Starfield added")
    output_mdata = output_mzp["M"]
    output_zdata = output_mzp["Z"]
    output_pdata = output_mzp["P"]

    output_pdata = update_spacecraft_location(output_pdata, output_pdata.meta.astropy_time)
    output_mdata = update_spacecraft_location(output_mdata, output_mdata.meta.astropy_time)
    output_zdata = update_spacecraft_location(output_zdata, output_zdata.meta.astropy_time)

    # Write out
    paths = []
    path = os.path.join(path_output, get_subdirectory(output_mdata), get_base_file_name(output_mdata) + ".fits")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    paths.append(path)
    logger.info(f"Writing data to {path}")
    write_ndcube_to_fits(output_mdata, path)

    path = os.path.join(path_output, get_subdirectory(output_zdata), get_base_file_name(output_zdata) + ".fits")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    paths.append(path)
    logger.info(f"Writing data to {path}")
    write_ndcube_to_fits(output_zdata, path)

    path = os.path.join(path_output, get_subdirectory(output_pdata), get_base_file_name(output_pdata) + ".fits")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    paths.append(path)
    logger.info(f"Writing data to {path}")
    write_ndcube_to_fits(output_pdata, path)

    logger.info("All data written")
    return paths

@task
def generate_l1_cr(input_file: str, path_output: str, rotation_stage: int, spacecraft_id: str) -> str:
    """Generate level 1 clear synthetic data."""
    logger = get_run_logger()
    input_pdata = load_ndcube_from_fits(input_file)
    logger.info(f"Read input file {input_file}")

    # Define the output data product
    product_code = "CR" + spacecraft_id
    product_level = "1"
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    fill_metadata_defaults(output_meta)
    output_meta["DATE-OBS"] = input_pdata.meta["DATE-OBS"].value
    output_wcs = generate_spacecraft_wcs(spacecraft_id, rotation_stage)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(output_wcs)
    for key in output_header:
        if (key in input_pdata.meta) and output_header[key] == "" and key not in ("COMMENT", "HISTORY"):
            output_meta[key].value = input_pdata.meta[key].value

    # Deproject to spacecraft frame
    output_data, output_wcs = deproject_clear(input_pdata, output_wcs)
    logger.info("Deprojected")

    # Quality marking
    output_data = mark_quality(output_data)
    logger.info("Quality marked")
    # output_data = add_distortion(output_data)  # noqa: ERA001
    # logger.info("Distortion added")  # noqa: ERA001

    output_data = add_starfield_clear(output_data)
    logger.info("Starfield added")

    output_cmeta = copy.deepcopy(output_meta)
    output_cwcs = copy.deepcopy(output_wcs)

    # Package into NDCube objects
    output_cdata = NDCube(data=output_data.data[:, :].astype(np.float32), wcs=output_cwcs, meta=output_cmeta)

    output_cdata.meta["TYPECODE"] = "CR"
    output_cdata.meta["DESCRPTN"] = "Simulated" + output_cdata.meta["DESCRPTN"].value
    output_cdata.meta["TITLE"] = "Simulated " + output_cdata.meta["TITLE"].value

    output_cdata.meta["POLAR"] = 9999

    output_cdata = update_spacecraft_location(output_cdata, output_cdata.meta.astropy_time)

    # Write out
    out_path = os.path.join(path_output, get_subdirectory(output_cdata), get_base_file_name(output_cdata) + ".fits")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    logger.info(f"Writing data to {out_path}")
    write_ndcube_to_fits(output_cdata, out_path)
    logger.info("Data written")
    return out_path
