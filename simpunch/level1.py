"""
Generates synthetic level 1 data
"""
import copy
import glob
import os

import astropy.units as u
import numpy as np
import reproject
import solpolpy
from astropy.coordinates import StokesSymbol, custom_stokes_symbol_mapping
from astropy.wcs import WCS, DistortionLookupTable
from ndcube import NDCollection, NDCube
from prefect import flow, task
from prefect.futures import wait
from prefect_dask import DaskTaskRunner
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            load_ndcube_from_fits, write_ndcube_to_fits)
from punchbowl.data.wcs import calculate_celestial_wcs_from_helio
from sunpy.coordinates import sun
from tqdm import tqdm

from simpunch.level2 import add_starfield, add_starfield_clear
from simpunch.util import update_spacecraft_location

PUNCH_STOKES_MAPPING = custom_stokes_symbol_mapping({10: StokesSymbol("pB", "polarized brightness"),
                                                     11: StokesSymbol("B", "total brightness")})


def calculate_pc_matrix(crota: float, cdelt: (float, float)) -> np.ndarray:
    """Calculate a PC matrix given CROTA and CDELT

    Parameters
    ----------
    crota : float
        rotation angle from the WCS
    cdelt : float
        pixel size from the WCS

    Returns
    -------
    np.ndarray
        PC matrix
    """
    return np.array(
        [
            [np.cos(crota), np.sin(crota) * (cdelt[1] / cdelt[0])],
            [-np.sin(crota) * (cdelt[0] / cdelt[1]), np.cos(crota)],
        ]
    )


def generate_spacecraft_wcs(spacecraft_id, rotation_stage, time) -> WCS:
    angle_step = 30

    if spacecraft_id in ['1', '2', '3']:
        if spacecraft_id == '1':
            angle_wfi = (0 + angle_step * rotation_stage) % 360 * u.deg
        elif spacecraft_id == '2':
            angle_wfi = (120 + angle_step * rotation_stage) % 360 * u.deg
        elif spacecraft_id == '3':
            angle_wfi = (240 + angle_step * rotation_stage) % 360 * u.deg

        out_wcs_shape = [2048, 2048]
        out_wcs = WCS(naxis=2)

        out_wcs.wcs.crpix = (1024.5, 150)
        out_wcs.wcs.crval = (0.0, 0.0)
        out_wcs.wcs.cdelt = 88 / 3600 * 0.9, 88 / 3600 * 0.9

        out_wcs.wcs.pc = calculate_pc_matrix(angle_wfi, out_wcs.wcs.cdelt)
        out_wcs.wcs.set_pv([(2, 1, (-sun.earth_distance(time)/sun.constants.radius).decompose().value)])

        out_wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"
        out_wcs.wcs.cunit = "deg", "deg"
    elif spacecraft_id == '4':
        angle_nfi = (0 + angle_step * rotation_stage) % 360 * u.deg
        out_wcs_shape = [2048, 2048]
        out_wcs = WCS(naxis=2)
        out_wcs.wcs.crpix = out_wcs_shape[1] / 2 + 0.5, out_wcs_shape[0] / 2 + 0.5
        out_wcs.wcs.crval = 0, 0
        out_wcs.wcs.cdelt = 30 / 3600, 30 / 3600

        out_wcs.wcs.pc = calculate_pc_matrix(angle_nfi, out_wcs.wcs.cdelt)

        out_wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'
        out_wcs.wcs.cunit = "deg", "deg"
    else:
        raise ValueError("Invalid spacecraft_id.")

    return out_wcs


def deproject(input_data, output_wcs, adaptive_reprojection=False):
    """Data deprojection"""
    # input_wcs = input_data.wcs.copy()
    # input_header = input_wcs.to_header()
    # # input_header['HGLN_OBS'] = input_data.meta['HGLN_OBS'].value
    # # input_header['HGLT_OBS'] = input_data.meta['HGLT_OBS'].value
    # # input_header['DSUN_OBS'] = input_data.meta['DSUN_OBS'].value
    # input_wcs = WCS(input_header)

    reconstructed_wcs = WCS(naxis=3)
    reconstructed_wcs.wcs.ctype = input_data.wcs.wcs.ctype
    reconstructed_wcs.wcs.cunit = input_data.wcs.wcs.cunit
    reconstructed_wcs.wcs.cdelt = input_data.wcs.wcs.cdelt
    reconstructed_wcs.wcs.crpix = input_data.wcs.wcs.crpix
    reconstructed_wcs.wcs.crval = input_data.wcs.wcs.crval
    reconstructed_wcs.wcs.pc = input_data.wcs.wcs.pc

    # print(input_wcs)
    reconstructed_wcs = calculate_celestial_wcs_from_helio(reconstructed_wcs,
                                                           input_data.meta.astropy_time,
                                                           input_data.data.shape)
    reconstructed_wcs = reconstructed_wcs.dropaxis(2)
    #
    # output_header = output_wcs.copy().to_header()
    # output_header['HGLN_OBS'] = input_data.meta['HGLN_OBS'].value
    # output_header['HGLT_OBS'] = input_data.meta['HGLT_OBS'].value
    # output_header['DSUN_OBS'] = input_data.meta['DSUN_OBS'].value
    # output_wcs_helio = WCS(output_header)
    # output_wcs = output_wcs_helio
    # print(output_wcs)
    output_wcs_helio = copy.deepcopy(output_wcs)
    output_wcs = calculate_celestial_wcs_from_helio(output_wcs,
                                                    input_data.meta.astropy_time,
                                                    input_data.data.shape).dropaxis(2)
    # output_wcs_helio = output_wcs

    reprojected_data = np.zeros((3, 2048, 2048), dtype=input_data.data.dtype)

    for i in np.arange(3):
        if adaptive_reprojection:
            reprojected_data[i, :, :] = reproject.reproject_adaptive((input_data.data[i, :, :],
                                                                      reconstructed_wcs),
                                                                     output_wcs,
                                                                     (2048, 2048),
                                                                     roundtrip_coords=False, return_footprint=False,
                                                                     kernel='Gaussian', boundary_mode='ignore')
        else:
            reprojected_data[i, :, :] = reproject.reproject_interp((input_data.data[i, :, :],
                                                                    reconstructed_wcs),
                                                                   output_wcs, (2048, 2048),
                                                                   roundtrip_coords=False, return_footprint=False)

    reprojected_data[np.isnan(reprojected_data)] = 0

    return NDCube(data=reprojected_data, wcs=output_wcs_helio, meta=input_data.meta), output_wcs_helio


def deproject_clear(input_data, output_wcs, adaptive_reprojection=False):
    """Data deprojection"""

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

    reprojected_data = np.zeros((2048, 2048), dtype=input_data.data.dtype)

    if adaptive_reprojection:
        reprojected_data[:, :] = reproject.reproject_adaptive((input_data.data[:, :],
                                                                  reconstructed_wcs),
                                                                 output_wcs,
                                                                 (2048, 2048),
                                                                 roundtrip_coords=False, return_footprint=False,
                                                                 kernel='Gaussian', boundary_mode='ignore')
    else:
        reprojected_data[:, :] = reproject.reproject_interp((input_data.data[:, :],
                                                                reconstructed_wcs),
                                                               output_wcs, (2048, 2048),
                                                               roundtrip_coords=False, return_footprint=False)

    reprojected_data[np.isnan(reprojected_data)] = 0

    return NDCube(data=reprojected_data, wcs=output_wcs_helio, meta=input_data.meta), output_wcs_helio


def mark_quality(input_data):
    """Data quality marking"""

    return input_data


def remix_polarization(input_data):
    """Remix polarization from (M, Z, P) to (P1, P2, P3) using solpolpy"""

    # Unpack data into a NDCollection object
    w = WCS(naxis=2)
    data_collection = NDCollection([("M", NDCube(data=input_data.data[0], wcs=w, meta={})),
                                     ("Z", NDCube(data=input_data.data[1], wcs=w, meta={})),
                                      ("P", NDCube(data=input_data.data[2], wcs=w, meta={}))])

    data_collection['M'].meta['POLAR'] = -60. * u.degree
    data_collection['Z'].meta['POLAR'] = 0. * u.degree
    data_collection['P'].meta['POLAR'] = 60. * u.degree

    # TODO - Remember that this needs to be the instrument frame MZP, not the mosaic frame
    resolved_data_collection = solpolpy.resolve(data_collection, 'npol',
                                                out_angles=[-60, 0, 60]*u.deg, imax_effect=False)

    # Repack data
    data_list = []
    wcs_list = []
    uncertainty_list = []
    for key in resolved_data_collection:
        data_list.append(resolved_data_collection[key].data)
        wcs_list.append(resolved_data_collection[key].wcs)
        uncertainty_list.append(resolved_data_collection[key].uncertainty)

    # Remove alpha channel if present
    if 'alpha' in resolved_data_collection.keys():
        data_list.pop()
        wcs_list.pop()
        uncertainty_list.pop()

    # Repack into an NDCube object
    new_data = np.stack(data_list, axis=0)
    if uncertainty_list[0] is not None:
        new_uncertainty = np.stack(uncertainty_list, axis=0)
    else:
        new_uncertainty = None

    new_wcs = input_data.wcs.copy()

    return NDCube(data=new_data, wcs=new_wcs, uncertainty=new_uncertainty, meta=input_data.meta)


def add_distortion(input_data, num_bins: int = 100):
    # make an initial empty distortion model
    r = np.linspace(0, input_data.data.shape[0], num_bins + 1)
    c = np.linspace(0, input_data.data.shape[1], num_bins + 1)
    r = (r[1:] + r[:-1]) / 2
    c = (c[1:] + c[:-1]) / 2

    err_px, err_py = r, c
    err_x = np.zeros((num_bins, num_bins))
    err_y = np.zeros((num_bins, num_bins))

    cpdis1 = DistortionLookupTable(
        -err_x.astype(np.float32), (0, 0), (err_px[0], err_py[0]), ((err_px[1] - err_px[0]), (err_py[1] - err_py[0]))
    )
    cpdis2 = DistortionLookupTable(
        -err_y.astype(np.float32), (0, 0), (err_px[0], err_py[0]), ((err_px[1] - err_px[0]), (err_py[1] - err_py[0]))
    )

    input_data.wcs.cpdis1 = cpdis1
    input_data.wcs.cpdis2 = cpdis2

    return input_data

@task
def generate_l1_pmzp(input_file, path_output, rotation_stage, spacecraft_id):
    """Generates level 1 polarized synthetic data"""
    input_pdata = load_ndcube_from_fits(input_file)

    # Define the output data product
    product_code = 'PM' + spacecraft_id
    product_level = '1'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta['DATE-OBS'] = input_pdata.meta['DATE-OBS'].value
    output_wcs = generate_spacecraft_wcs(spacecraft_id, rotation_stage, input_pdata.meta.astropy_time)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(output_wcs)
    for key in output_header.keys():
        if (key in input_pdata.meta) and output_header[key] == '' and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key].value = input_pdata.meta[key].value

    # Deproject to spacecraft frame
    output_data, output_wcs = deproject(input_pdata, output_wcs)

    # Quality marking
    output_data = mark_quality(output_data)

    output_data = add_starfield(output_data)

    # Polarization remixing
    output_data = remix_polarization(output_data)

    output_mmeta = copy.deepcopy(output_meta)
    output_zmeta = copy.deepcopy(output_meta)
    output_pmeta = copy.deepcopy(output_meta)

    output_mwcs = copy.deepcopy(output_wcs)
    output_zwcs = copy.deepcopy(output_wcs)
    output_pwcs = copy.deepcopy(output_wcs)

    # Package into NDCube objects
    output_mdata = NDCube(data=output_data.data[0, :, :].astype(np.float32), wcs=output_mwcs, meta=output_mmeta)
    output_zdata = NDCube(data=output_data.data[1, :, :].astype(np.float32), wcs=output_zwcs, meta=output_zmeta)
    output_pdata = NDCube(data=output_data.data[2, :, :].astype(np.float32), wcs=output_pwcs, meta=output_pmeta)

    output_mdata.meta['TYPECODE'] = 'PM'
    output_zdata.meta['TYPECODE'] = 'PZ'
    output_pdata.meta['TYPECODE'] = 'PP'

    output_mdata.meta['POLAR'] = -60
    output_zdata.meta['POLAR'] = 0
    output_pdata.meta['POLAR'] = 60

    # Add distortion
    output_mdata = add_distortion(output_mdata)
    output_zdata = add_distortion(output_zdata)
    output_pdata = add_distortion(output_pdata)

    output_pdata = update_spacecraft_location(output_pdata, output_pdata.meta.astropy_time)
    output_mdata = update_spacecraft_location(output_mdata, output_mdata.meta.astropy_time)
    output_zdata = update_spacecraft_location(output_zdata, output_zdata.meta.astropy_time)

    # Write out
    write_ndcube_to_fits(output_mdata, path_output + get_base_file_name(output_mdata) + '.fits')
    write_ndcube_to_fits(output_zdata, path_output + get_base_file_name(output_zdata) + '.fits')
    write_ndcube_to_fits(output_pdata, path_output + get_base_file_name(output_pdata) + '.fits')


@task
def generate_l1_cr(input_file, path_output, rotation_stage, spacecraft_id):
    """Generates level 1 clear synthetic data"""
    input_pdata = load_ndcube_from_fits(input_file)

    # Define the output data product
    product_code = 'CR' + spacecraft_id
    product_level = '1'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)
    output_meta['DATE-OBS'] = input_pdata.meta['DATE-OBS'].value
    output_wcs = generate_spacecraft_wcs(spacecraft_id, rotation_stage, input_pdata.meta.astropy_time)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header(output_wcs)
    for key in output_header.keys():
        if (key in input_pdata.meta) and output_header[key] == '' and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key].value = input_pdata.meta[key].value

    # Deproject to spacecraft frame
    output_data, output_wcs = deproject_clear(input_pdata, output_wcs)

    # Quality marking
    output_data = mark_quality(output_data)

    output_data = add_starfield_clear(output_data)

    output_cmeta = copy.deepcopy(output_meta)
    output_cwcs = copy.deepcopy(output_wcs)

    # Package into NDCube objects
    output_cdata = NDCube(data=output_data.data[:, :].astype(np.float32), wcs=output_cwcs, meta=output_cmeta)

    output_cdata.meta['TYPECODE'] = 'CR'

    output_cdata.meta['POLAR'] = 9999

    # Add distortion
    output_cdata = add_distortion(output_cdata)

    output_cdata = update_spacecraft_location(output_cdata, output_cdata.meta.astropy_time)

    # Write out
    write_ndcube_to_fits(output_cdata, path_output + get_base_file_name(output_cdata) + '.fits')


@flow(log_prints=True, task_runner=DaskTaskRunner(
    cluster_kwargs={"n_workers": 8, "threads_per_worker": 2}
))
def generate_l1_all(datadir):
    """Generate all level 1 synthetic data
     L1 <- polarization deprojection <- quality marking <- deproject to spacecraft FOV <- L2_PTM"""

    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, 'synthetic_l1/')
    os.makedirs(outdir, exist_ok=True)
    print(f"Outputting to {outdir}")

    # Parse list of level 3 model data
    files_ptm = glob.glob(datadir + '/synthetic_l2/*PTM*.fits')
    files_ctm = glob.glob(datadir + '/synthetic_l2/*CTM*.fits')
    print(f"Generating based on {len(files_ptm)} PTM files.")
    files_ptm.sort()

    futures = []

    # Run individual generators
    for i, file_ptm in tqdm(enumerate(files_ptm), total=len(files_ptm)):
        rotation_stage = int((i % 16) / 2)
        futures.append(generate_l1_pmzp.submit(file_ptm, outdir, rotation_stage, '1'))
        futures.append(generate_l1_pmzp.submit(file_ptm, outdir, rotation_stage, '2'))
        futures.append(generate_l1_pmzp.submit(file_ptm, outdir, rotation_stage, '3'))
        futures.append(generate_l1_pmzp.submit(file_ptm, outdir, rotation_stage, '4'))

    for i, file_ctm in tqdm(enumerate(files_ctm), total=len(files_ctm)):
        rotation_stage = int((i % 16) / 2)
        futures.append(generate_l1_cr.submit(file_ctm, outdir, rotation_stage, '1'))
        futures.append(generate_l1_cr.submit(file_ctm, outdir, rotation_stage, '2'))
        futures.append(generate_l1_cr.submit(file_ctm, outdir, rotation_stage, '3'))
        futures.append(generate_l1_cr.submit(file_ctm, outdir, rotation_stage, '4'))

    wait(futures)

if __name__ == '__main__':
    generate_l1_all("/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/")
