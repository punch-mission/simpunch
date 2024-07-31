"""
Generates synthetic level 1 data
"""
import glob
import os
from datetime import datetime, timedelta

import astropy.units as u
import numpy as np
import reproject
import solpolpy
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.coordinates import StokesSymbol, custom_stokes_symbol_mapping
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import add_stokes_axis_to_wcs
from ndcube import NDCollection, NDCube
from punchbowl.data import (NormalizedMetadata, get_base_file_name,
                            write_ndcube_to_fits)
from sunpy.coordinates import frames, sun
from tqdm import tqdm

PUNCH_STOKES_MAPPING = custom_stokes_symbol_mapping({10: StokesSymbol("pB", "polarized brightness"),
                                                     11: StokesSymbol("B", "total brightness")})


def generate_spacecraft_wcs(spacecraft_id, rotation_stage) -> WCS:
    angle_step = 30

    if spacecraft_id in ['1', '2', '3']:

        if spacecraft_id == '1':
            angle_wfi = (0 + angle_step * rotation_stage) % 360
        elif spacecraft_id == '2':
            angle_wfi = (120 + angle_step * rotation_stage) % 360
        elif spacecraft_id == '3':
            angle_wfi = (240 + angle_step * rotation_stage) % 360

        out_wcs_shape = [2048, 2048]
        out_wcs = WCS(naxis=2)

        out_wcs.wcs.crpix = out_wcs_shape[1] / 2 - 0.5, out_wcs_shape[0] / 2 - 0.5
        out_wcs.wcs.crval = (24.75 * np.sin(angle_wfi * u.deg) + (0.5 * np.sin(angle_wfi * u.deg)),
                             24.75 * np.cos(angle_wfi * u.deg) - (0.5 * np.cos(angle_wfi * u.deg)))
        out_wcs.wcs.cdelt = 0.02, 0.02
        out_wcs.wcs.lonpole = angle_wfi
        out_wcs.wcs.ctype = "HPLN-AZP", "HPLT-AZP"

    if spacecraft_id == '4':
        angle_nfi1 = (0 + angle_step * rotation_stage) % 360
        out_wcs_shape = [2048, 2048]
        out_wcs = WCS(naxis=2)
        out_wcs.wcs.crpix = out_wcs_shape[1] / 2 - 0.5, out_wcs_shape[0] / 2 - 0.5
        out_wcs.wcs.crval = 0, 0
        out_wcs.wcs.cdelt = 0.01, 0.01
        out_wcs.wcs.lonpole = angle_nfi1
        out_wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'

    return out_wcs


def deproject(input_data, output_wcs, adaptive_reprojection=False):
    """Data deprojection"""

    input_wcs = input_data.wcs

    output_header = output_wcs.to_header()
    output_header['HGLN_OBS'] = input_data.meta['HGLN_OBS']
    output_header['HGLT_OBS'] = input_data.meta['HGLT_OBS']
    output_header['DSUN_OBS'] = input_data.meta['DSUN_OBS']
    output_wcs = WCS(output_header)

    reprojected_data = np.zeros((3, 2048, 2048), dtype=input_data.data.dtype)

    time_current = Time(input_data.meta['DATE-OBS'])
    skycoord_origin = SkyCoord(0 * u.deg, 0 * u.deg,
                               frame=frames.Helioprojective,
                               obstime=time_current,
                               observer='earth')

    with frames.Helioprojective.assume_spherical_screen(skycoord_origin.observer):
        for i in np.arange(3):
            if adaptive_reprojection:
                reprojected_data[i, :, :] = reproject.reproject_adaptive((input_data.data[i, :, :], input_wcs[i]),
                                                                         output_wcs,
                                                                         (2048, 2048),
                                                                         roundtrip_coords=False, return_footprint=False,
                                                                         kernel='Gaussian', boundary_mode='ignore')
            else:
                reprojected_data[i, :, :] = reproject.reproject_interp((input_data.data[i, :, :], input_wcs[i]),
                                                                       output_wcs, (2048, 2048),
                                                                       roundtrip_coords=False, return_footprint=False)

    output_wcs = add_stokes_axis_to_wcs(output_wcs, 2)

    reprojected_data[np.isnan(reprojected_data)] = 0

    return NDCube(data=reprojected_data, wcs=output_wcs, meta=input_data.meta)


def mark_quality(input_data):
    """Data quality marking"""

    return input_data


def remix_polarization(input_data):
    """Remix polarization from (M, Z, P) to (P1, P2, P3) using solpolpy"""

    # Unpack data into a NDCollection object
    data_collection = NDCollection([("Bm", input_data[0, :, :]),
                                    ("Bz", input_data[1, :, :]),
                                    ("Bp", input_data[2, :, :])], aligned_axes='all')

    data_collection['Bm'].meta['POLAR'] = -60.
    data_collection['Bz'].meta['POLAR'] = 0.
    data_collection['Bp'].meta['POLAR'] = 60.

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


def generate_l1_pmzp(input_file, path_output, time_obs, time_delta, rotation_stage, spacecraft_id):
    """Generates level 1 polarized synthetic data"""

    # Read in the input data
    with fits.open(input_file) as hdul:
        input_data = hdul[1].data
        input_header = hdul[1].header

    input_pdata = NDCube(data=input_data, meta=dict(input_header), wcs=WCS(input_header))

    # Define the output data product
    product_code = 'PM' + spacecraft_id
    product_level = '1'
    output_meta = NormalizedMetadata.load_template(product_code, product_level)

    # Synchronize overlapping metadata keys
    output_header = output_meta.to_fits_header()
    for key in output_header.keys():
        if (key in input_header) and (output_header[key] in ['', None]) and (key != 'COMMENT') and (key != 'HISTORY'):
            output_meta[key].value = input_pdata.meta[key]

    # Deproject to spacecraft frame
    output_wcs = generate_spacecraft_wcs(spacecraft_id, rotation_stage)
    output_data = deproject(input_pdata, output_wcs)

    # Quality marking
    output_data = mark_quality(output_data)

    # Polarization remixing
    output_data = remix_polarization(output_data)

    # Write out positional data assuming Earth-center in the absence of orbital data
    output_meta['GEOD_LAT'] = 0.
    output_meta['GEOD_LON'] = 0.
    output_meta['GEOD_ALT'] = 0.

    earth_center = EarthLocation(lat=0 * u.deg, lon=0 * u.deg, height=0 * u.km)
    earth_coord = SkyCoord(0 * u.deg, 0 * u.deg, frame=frames.HeliographicStonyhurst,
                           obstime=Time(output_meta['DATE-OBS'].value), location=earth_center, observer='earth')

    output_meta['CRLN_OBS'] = earth_coord.heliographic_carrington.lon.value
    output_meta['CRLT_OBS'] = earth_coord.heliographic_carrington.lat.value

    output_meta['HEEX_OBS'] = earth_coord.heliocentricearthecliptic.cartesian.x.to(u.m).value
    output_meta['HEEY_OBS'] = earth_coord.heliocentricearthecliptic.cartesian.y.to(u.m).value
    output_meta['HEEZ_OBS'] = earth_coord.heliocentricearthecliptic.cartesian.z.to(u.m).value

    output_meta['CAR_ROT'] = sun.carrington_rotation_number(time_obs)

    # Package into NDCube objects
    output_mdata = NDCube(data=output_data.data[0, :, :].astype(np.float32), wcs=output_wcs, meta=output_meta)
    output_zdata = NDCube(data=output_data.data[1, :, :].astype(np.float32), wcs=output_wcs, meta=output_meta)
    output_pdata = NDCube(data=output_data.data[2, :, :].astype(np.float32), wcs=output_wcs, meta=output_meta)

    output_mdata.meta['TYPECODE'] = 'PM'
    output_zdata.meta['TYPECODE'] = 'PZ'
    output_pdata.meta['TYPECODE'] = 'PP'

    output_mdata.meta['POLAR'] = -60
    output_zdata.meta['POLAR'] = 0
    output_pdata.meta['POLAR'] = 60

    # Write out
    version_number = 0
    write_ndcube_to_fits(output_mdata, path_output + get_base_file_name(output_mdata) + str(version_number) + '.fits',
                         skip_wcs_conversion=True)
    write_ndcube_to_fits(output_zdata, path_output + get_base_file_name(output_zdata) + str(version_number) + '.fits',
                         skip_wcs_conversion=True)
    write_ndcube_to_fits(output_pdata, path_output + get_base_file_name(output_pdata) + str(version_number) + '.fits',
                         skip_wcs_conversion=True)


# @click.command()
# @click.argument('datadir', type=click.Path(exists=True))
def generate_l1_all(datadir):
    """Generate all level 1 synthetic data
     L1 <- polarization deprojection <- quality marking <- deproject to spacecraft FOV <- L2_PTM"""

    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, 'synthetic_L1/')
    print(f"Outputting to {outdir}")

    # Parse list of level 3 model data
    files_ptm = glob.glob(datadir + '/synthetic_L2/*PTM*.fits')
    print(f"Generating based on {len(files_ptm)} PTM files.")
    files_ptm.sort()

    files_ptm = files_ptm[0:5]

    # Set the overall start time for synthetic data
    # Note the timing for data products - 32 minutes / low noise ; 8 minutes / clear ; 4 minutes / polarized
    time_start = datetime(2024, 6, 20, 0, 0, 0)

    # Generate a corresponding set of observation times for polarized trefoil / NFI data
    time_delta = timedelta(minutes=4)
    times_obs = np.arange(len(files_ptm)) * time_delta + time_start

    for i, (file_ptm, time_obs) in tqdm(enumerate(zip(files_ptm, times_obs)), total=len(files_ptm)):
        rotation_stage = int((i % 16) / 2)
        generate_l1_pmzp(file_ptm, outdir, time_obs, time_delta, rotation_stage, '1')
        generate_l1_pmzp(file_ptm, outdir, time_obs, time_delta, rotation_stage, '2')
        generate_l1_pmzp(file_ptm, outdir, time_obs, time_delta, rotation_stage, '3')
        generate_l1_pmzp(file_ptm, outdir, time_obs, time_delta, rotation_stage, '4')

    # pool = ProcessPoolExecutor()
    # futures = []
    # # Run individual generators
    # for i, (file_ptm, time_obs) in tqdm(enumerate(zip(files_ptm, times_obs)), total=len(files_ptm)):
    #     rotation_stage = int((i % 16) / 2)
    #     futures.append(pool.submit(generate_l1_pm, file_ptm, outdir, time_obs, time_delta, rotation_stage, '1'))
    #     futures.append(pool.submit(generate_l1_pm, file_ptm, outdir, time_obs, time_delta, rotation_stage, '2'))
    #     futures.append(pool.submit(generate_l1_pm, file_ptm, outdir, time_obs, time_delta, rotation_stage, '3'))
    #     futures.append(pool.submit(generate_l1_pm, file_ptm, outdir, time_obs, time_delta, rotation_stage, '4'))
    #
    # with tqdm(total=len(futures)) as pbar:
    #     for _ in as_completed(futures):
    #         pbar.update(1)


if __name__ == "__main__":
    generate_l1_all('/Users/clowder/data/punch')
