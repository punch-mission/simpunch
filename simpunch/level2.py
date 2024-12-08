"""Generate synthetic level 2 data.

PTM - PUNCH Level-2 Polarized (MZP) Mosaic
"""
import copy
import glob
import os
from math import floor

import astropy.time
import astropy.units as u
import numpy as np
import reproject
import solpolpy
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from astropy.wcs import WCS
from ndcube import NDCollection, NDCube
from photutils.datasets import make_model_image, make_noise_image
from prefect import flow, task
from prefect.futures import wait
from prefect_dask import DaskTaskRunner
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            load_ndcube_from_fits, write_ndcube_to_fits)
from punchbowl.data.wcs import calculate_celestial_wcs_from_helio, get_p_angle

from simpunch.stars import (filter_for_visible_stars, find_catalog_in_image,
                            load_raw_hipparcos_catalog)
from simpunch.util import update_spacecraft_location


def get_fcorona_parameters(date_obs: astropy.time.Time) -> dict[str, float]:
    """Get time dependent F corona model parameters."""
    phase = date_obs.decimalyear - int(date_obs.decimalyear)

    tilt_angle = 3 * u.deg * np.sin(phase * 2 * np.pi)
    b = 300. + 50 * np.cos(phase * 2 * np.pi)

    return {"tilt_angle": tilt_angle,
            "b": b}


def generate_fcorona(shape: (int, int),
                     tilt_angle: float = 3 * u.deg,
                     a: float = 600.,
                     b: float = 300.,
                     tilt_offset: tuple[float] = (0, 0)) -> np.ndarray:
    """Generate an F corona model."""
    fcorona = np.zeros(shape)

    if len(shape) > 2:  # noqa: PLR2004
        xdim = 1
        ydim = 2
    else:
        xdim = 0
        ydim = 1

    x, y = np.meshgrid(np.arange(shape[xdim]), np.arange(shape[ydim]))
    x_center, y_center = shape[xdim] // 2 + tilt_offset[0], shape[ydim] // 2 + tilt_offset[1]

    x_rotated = (x - x_center) * np.cos(tilt_angle) + (y - y_center) * np.sin(tilt_angle) + x_center
    y_rotated = -(x - x_center) * np.sin(tilt_angle) + (y - y_center) * np.cos(tilt_angle) + y_center

    distance = np.sqrt(((x_rotated - x_center) / a) ** 2 + ((y_rotated - y_center) / b) ** 2)

    max_radius = np.sqrt((shape[xdim] / 2) ** 2 + (shape[ydim] / 2) ** 2)
    min_n = 1.54
    max_n = 1.65

    n = min_n + (max_n - min_n) * (distance / max_radius)

    superellipse = (np.abs((x_rotated - x_center) / a) ** n +
                    np.abs((y_rotated - y_center) / b) ** n) ** (1 / n) / (2 ** (1 / n))

    max_distance = 1
    fcorona_profile = np.exp(-superellipse ** 2 / (2 * max_distance ** 2))

    fcorona_profile = fcorona_profile / fcorona_profile.max() * 1e-12

    if len(shape) > 2:  # noqa: PLR2004
        for i in np.arange(fcorona.shape[0]):
            fcorona[i, :, :] = fcorona_profile[:, :]
    else:
        fcorona[:, :] = fcorona_profile[:, :]

    return fcorona


def add_fcorona(input_data: NDCube) -> NDCube:
    """Add synthetic f-corona model."""
    fcorona_parameters = get_fcorona_parameters(input_data.meta.astropy_time)

    fcorona = generate_fcorona(input_data.data.shape, **fcorona_parameters)

    fcorona = fcorona * (input_data.data != 0)

    input_data.data[...] = input_data.data[...] + fcorona

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


def generate_dummy_polarization(map_scale: float = 0.225,
                                pol_factor: float = 0.5) -> NDCube:
    """Create a synthetic polarization map."""
    shape = [int(floor(180 / map_scale)), int(floor(360 / map_scale))]
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

    for k, key in enumerate(valid_keys):
        dummy_polarmap = generate_dummy_polarization(pol_factor=polfactor[k])
        # Extract ROI corresponding to input wcs
        polar_roi = reproject.reproject_adaptive(
            (dummy_polarmap.data, dummy_polarmap.wcs), wcs_stellar_input, input_data.data.shape,
            roundtrip_coords=False, return_footprint=False, x_cyclic=True,
            conserve_flux=True, center_jacobian=True, despike_jacobian=True)
        input_data_cel[key].data[...] = input_data_cel[key].data + polar_roi * starfield_data

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
                                          flux_set=10*2.0384547E-9,
                                          fwhm=3, dimmest_magnitude=12,
                                          noise_mean=None, noise_std=None)

    starfield_data = np.zeros(input_data.data.shape)
    starfield_data[:, :] = starfield * (np.logical_not(np.isclose(input_data.data[:, :], 0, atol=1E-18)))

    input_data.data[...] = input_data.data[...] + starfield_data

    return input_data


def remix_polarization(input_data: NDCube) -> NDCube:
    """Remix polarization from (B, pB) to (M,Z,P) using solpolpy."""
    # Unpack data into a NDCollection object
    data_collection = NDCollection(
        [("B", NDCube(data=input_data.data[0], wcs=input_data.wcs)),
         ("pB", NDCube(data=input_data.data[1], wcs=input_data.wcs))],
        aligned_axes="all")

    resolved_data_collection = solpolpy.resolve(data_collection, "mzpsolar", imax_effect=False)

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
    if uncertainty_list[0] is not None:  # noqa: SIM108
        new_uncertainty = np.stack(uncertainty_list, axis=0)
    else:
        new_uncertainty = None

    new_wcs = input_data.wcs.copy()

    return NDCube(data=new_data, wcs=new_wcs, uncertainty=new_uncertainty, meta=input_data.meta)


@task
def generate_l2_ptm(input_file: str, path_output: str) -> None:
    """Generate level 2 PTM synthetic data."""
    # Read in the input data
    input_pdata = load_ndcube_from_fits(input_file)

    # Define the output data product
    product_code = "PTM"
    product_level = "2"
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta["DATE-OBS"] = input_pdata.meta["DATE-OBS"].value
    output_wcs = input_pdata.wcs

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(output_wcs)
    for key in output_header:
        if (key in input_pdata.meta) and output_header[key] == "" and key not in ("COMMENT", "HISTORY"):
            output_meta[key].value = input_pdata.meta[key].value
    output_meta["DESCRPTN"] = "Simulated " + output_meta["DESCRPTN"].value
    output_meta["TITLE"] = "Simulated " + output_meta["TITLE"].value

    output_data = remix_polarization(input_pdata)
    output_data = add_fcorona(output_data)

    # Package into a PUNCHdata object
    output_pdata = NDCube(data=output_data.data.astype(np.float32), wcs=output_wcs, meta=output_meta)
    output_pdata = update_spacecraft_location(output_pdata, input_pdata.meta.astropy_time)

    # Write out
    write_ndcube_to_fits(output_pdata, path_output + get_base_file_name(output_pdata) + ".fits")


@task
def generate_l2_ctm(input_file: str, path_output: str) -> None:
    """Generate level 2 CTM synthetic data."""
    # Read in the input data
    input_pdata = load_ndcube_from_fits(input_file)

    # Define the output data product
    product_code = "CTM"
    product_level = "2"
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta["DATE-OBS"] = input_pdata.meta["DATE-OBS"].value

    output_wcs = input_pdata.wcs

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(output_wcs)
    for key in output_header:
        if (key in input_pdata.meta) and output_header[key] == "" and key not in ("COMMENT", "HISTORY"):
            output_meta[key].value = input_pdata.meta[key].value
    output_meta["DESCRPTN"] = "Simulated " + output_meta["DESCRPTN"].value
    output_meta["TITLE"] = "Simulated " + output_meta["TITLE"].value

    output_data = add_fcorona(input_pdata)

    # Package into a PUNCHdata object
    output_pdata = NDCube(data=output_data.data.astype(np.float32), wcs=output_wcs, meta=output_meta)
    output_pdata = update_spacecraft_location(output_pdata, input_pdata.meta.astropy_time)

    # Write out
    write_ndcube_to_fits(output_pdata, path_output + get_base_file_name(output_pdata) + ".fits")


@flow(log_prints=True, task_runner=DaskTaskRunner(
    cluster_kwargs={"n_workers": 128, "threads_per_worker": 2},
))
def generate_l2_all(datadir: str, outdir: str) -> None:
    """Generate all level 2 synthetic data.

    L2_PTM <- f-corona subtraction <- starfield subtraction <- remix polarization <- L3_PTM
    """
    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(outdir, "synthetic_l2/")
    os.makedirs(outdir, exist_ok=True)
    print(f"Outputting to {outdir}")

    # Parse list of level 3 model data
    files_ptm = glob.glob(datadir + "/synthetic_l3/*PTM*.fits")
    files_ctm = glob.glob(datadir + "/synthetic_l3/*CTM*.fits")
    print(f"Generating based on {len(files_ptm)} PTM files.")
    print(f"Generating based on {len(files_ctm)} CTM files.")
    files_ptm.sort()

    futures = []
    futures.extend(generate_l2_ptm.map(files_ptm, outdir))
    futures.extend(generate_l2_ctm.map(files_ctm, outdir))
    wait(futures)
