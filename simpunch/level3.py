"""
Generates synthetic level 3 data
PAM - PUNCH Level-3 Polarized Low Noise Mosaic
PAN - PUNCH Level-3 Polarized Low Noise NFI Image
PTM - PUNCH Level-3 Polarized Mosaic
PNN - PUNCH Level-3 Polarized NFI Image
"""
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

import astropy.units as u
import numpy as np
import reproject
import scipy.ndimage
from astropy.constants import R_sun
from astropy.coordinates import GCRS, EarthLocation, SkyCoord, get_sun
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import add_stokes_axis_to_wcs, proj_plane_pixel_area
from ndcube import NDCube
from punchbowl.data import NormalizedMetadata, write_ndcube_to_fits
from punchbowl.data.io import get_base_file_name
from sunpy.coordinates import frames, sun
from sunpy.coordinates.ephemeris import get_earth
from sunpy.coordinates.sun import _sun_north_angle_to_z
from sunpy.map import make_fitswcs_header
from tqdm import tqdm


def extract_crota_from_wcs(wcs):
    return np.arctan2(wcs.wcs.pc[1, 0], wcs.wcs.pc[0, 0]) * u.rad


def compute_celestial_from_helio(pdata: NDCube) -> NDCube:
    """Updates secondary celestial WCS for level 3 data products, using coordinate transformation"""

    wcs_helio = pdata.wcs.copy()
    date_obs = Time(pdata.meta['DATE-OBS'].value)
    data_shape = pdata.data.shape

    is_3d = len(data_shape) == 3

    # we're at the center of the Earth
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    reference_coord = SkyCoord(
        wcs_helio.wcs.crval[0] * u.Unit(wcs_helio.wcs.cunit[0]),
        wcs_helio.wcs.crval[1] * u.Unit(wcs_helio.wcs.cunit[1]),
        frame="gcrs",
        obstime=date_obs,
        obsgeoloc=test_gcrs.cartesian,
        obsgeovel=test_gcrs.velocity.to_cartesian(),
        distance=test_gcrs.hcrs.distance,
    )

    reference_coord_arcsec = reference_coord.transform_to(frames.Helioprojective(observer=test_gcrs))

    cdelt1 = (np.abs(wcs_helio.wcs.cdelt[0]) * u.deg).to(u.arcsec)
    cdelt2 = (np.abs(wcs_helio.wcs.cdelt[1]) * u.deg).to(u.arcsec)

    geocentric = GCRS(obstime=date_obs)
    p_angle = _sun_north_angle_to_z(geocentric)

    crota = extract_crota_from_wcs(wcs_helio)

    new_header = make_fitswcs_header(
        data_shape[1:] if is_3d else data_shape,
        reference_coord_arcsec,
        reference_pixel=u.Quantity(
            [wcs_helio.wcs.crpix[0] - 1, wcs_helio.wcs.crpix[1] - 1] * u.pixel
        ),
        scale=u.Quantity([cdelt1, cdelt2] * u.arcsec / u.pix),
        rotation_angle=-p_angle - crota,
        observatory="PUNCH",
        projection_code=wcs_helio.wcs.ctype[0][-3:]
    )

    wcs_celestial = WCS(new_header)
    wcs_celestial.wcs.ctype = "RA---ARC", "DEC--ARC"
    sun_location = get_sun_ra_dec(pdata.meta['DATE-OBS'].value)
    wcs_celestial.wcs.crval = sun_location[0], sun_location[1]
    wcs_celestial.wcs.cdelt = wcs_celestial.wcs.cdelt * (-1, 1)

    if is_3d:
        wcs_celestial = add_stokes_axis_to_wcs(wcs_celestial, 2)
        wcs_celestial.array_shape = pdata.data.shape[0], pdata.data.shape[1], pdata.data.shape[2]

    pdata.meta['CRVAL1A'] = sun_location[0]
    pdata.meta['CRVAL2A'] = sun_location[1]

    celestial_pc = wcs_celestial.wcs.pc
    pdata.meta['PC1_1A'] = celestial_pc[0,0]
    pdata.meta['PC1_2A'] = celestial_pc[1,0]
    pdata.meta['PC1_3A'] = celestial_pc[2,0]
    pdata.meta['PC2_1A'] = celestial_pc[0,1]
    pdata.meta['PC2_2A'] = celestial_pc[1,1]
    pdata.meta['PC2_3A'] = celestial_pc[2,1]
    pdata.meta['PC3_1A'] = celestial_pc[0,2]
    pdata.meta['PC3_2A'] = celestial_pc[1,2]
    pdata.meta['PC3_3A'] = celestial_pc[2,2]

    return pdata


def get_sun_ra_dec(dt: datetime):
    position = get_sun(Time(str(dt), scale='utc'))
    return position.ra.value, position.dec.value


def define_mask(shape=(4096, 4096), distance_value=0.68):
    """Define a mask to describe the FOV for low-noise PUNCH data products"""
    center = (int(shape[0] / 2), int(shape[1] / 2))

    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_arr = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    return (dist_arr / dist_arr.max()) < distance_value


def define_trefoil_mask(rotation_stage=0):
    """Define a mask to describe the FOV for trefoil mosaic PUNCH data products"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return np.load(os.path.join(dir_path, 'data/trefoil_mask.npz'))['trefoil_mask'][rotation_stage,:,:]


def generate_uncertainty(pdata: NDCube) -> NDCube:

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

    # photon_noise = generate_noise_photon(photon_array)
    photon_noise = np.sqrt(photon_array)

    uncertainty = photon_noise  # / photon_noise.max()

    uncertainty[pdata.data == 0] = np.inf

    pdata.uncertainty.array = uncertainty

    return pdata


def assemble_punchdata(input_tb, input_pb, wcs, product_code, product_level, mask=None):
    """Assemble a punchdata object with correct metadata"""

    with fits.open(input_tb) as hdul:
        data_tb = hdul[1].data / 1e8  # the 1e8 comes from the units on FORWARD output
        if data_tb.shape == (2048, 2048):
            data_tb = scipy.ndimage.zoom(data_tb, 2, order=0)
        data_tb[np.where(data_tb == -9.999e-05)] = 0
        if mask is not None:
            data_tb = data_tb * mask

    with fits.open(input_pb) as hdul:
        data_pb = hdul[1].data / 1e8 # the 1e8 comes from the units on FORWARD output
        if data_pb.shape == (2048, 2048):
            data_pb = scipy.ndimage.zoom(data_pb, 2, order=0)
        data_pb[np.where(data_pb == -9.999e-05)] = 0
        if mask is not None:
            data_pb = data_pb * mask

    datacube = np.stack([data_tb, data_pb]).astype('float32')
    uncertainty = StdDevUncertainty(np.zeros(datacube.shape))
    uncertainty.array[datacube == 0] = 1
    meta = NormalizedMetadata.load_template(product_code, product_level)
    return NDCube(data=datacube, wcs=wcs, meta=meta, uncertainty=uncertainty)


def update_spacecraft_location(input_data, time_obs):
    input_data.meta['GEOD_LAT'] = 0.
    input_data.meta['GEOD_LON'] = 0.
    input_data.meta['GEOD_ALT'] = 0.

    coord = get_earth(time_obs)
    coord.observer = 'earth'

    # S/C Heliographic Stonyhurst
    input_data.meta['HGLN_OBS'] = coord.heliographic_stonyhurst.lon.value
    input_data.meta['HGLT_OBS'] = coord.heliographic_stonyhurst.lat.value

    # S/C Heliographic Carrington
    input_data.meta['CRLN_OBS'] = coord.heliographic_carrington.lon.value
    input_data.meta['CRLT_OBS'] = coord.heliographic_carrington.lat.value

    input_data.meta['DSUN_OBS'] = sun.earth_distance(time_obs).to(u.m).value

    # S/C Heliocentric Earth Ecliptic
    input_data.meta['HEEX_OBS'] = coord.heliocentricearthecliptic.cartesian.x.to(u.m).value
    input_data.meta['HEEY_OBS'] = coord.heliocentricearthecliptic.cartesian.y.to(u.m).value
    input_data.meta['HEEZ_OBS'] = coord.heliocentricearthecliptic.cartesian.z.to(u.m).value

    # S/C Heliocentric Inertial
    input_data.meta['HCIX_OBS'] = coord.heliocentricinertial.cartesian.x.to(u.m).value
    input_data.meta['HCIY_OBS'] = coord.heliocentricinertial.cartesian.y.to(u.m).value
    input_data.meta['HCIZ_OBS'] = coord.heliocentricinertial.cartesian.z.to(u.m).value

    # S/C Heliocentric Earth Equatorial
    input_data.meta['HEQX_OBS'] = (coord.heliographic_stonyhurst.cartesian.x.value * u.AU).to(u.m).value
    input_data.meta['HEQY_OBS'] = (coord.heliographic_stonyhurst.cartesian.y.value * u.AU).to(u.m).value
    input_data.meta['HEQZ_OBS'] = (coord.heliographic_stonyhurst.cartesian.z.value * u.AU).to(u.m).value

    input_data.meta['SOLAR_EP'] = sun.P(time_obs).value
    input_data.meta['CAR_ROT'] = float(sun.carrington_rotation_number(time_obs))

    return input_data


def generate_l3_ptm(input_tb, input_pb, path_output, time_obs, time_delta, rotation_stage):
    """Generate PTM - PUNCH Level-3 Polarized Mosaic"""
    # Define the mosaic WCS (helio)
    mosaic_shape = (4096, 4096)
    mosaic_wcs = WCS(naxis=2)
    mosaic_wcs.wcs.crpix = mosaic_shape[1] / 2 - 0.5, mosaic_shape[0] / 2 - 0.5
    mosaic_wcs.wcs.crval = 0, 0
    mosaic_wcs.wcs.cdelt = 0.0225, 0.0225
    mosaic_wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'
    mosaic_wcs = add_stokes_axis_to_wcs(mosaic_wcs, 2)

    # Mask data to define the field of view
    mask = define_trefoil_mask(rotation_stage=rotation_stage)

    # Read data and assemble into PUNCHData object
    pdata = assemble_punchdata(input_tb, input_pb, mosaic_wcs, product_code='PTM', product_level='3', mask=mask)

    # Update required metadata
    tstring_start = time_obs.strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_end = (time_obs + time_delta).strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_avg = (time_obs + time_delta / 2).strftime('%Y-%m-%dT%H:%M:%S.000')

    pdata.meta['DATE-OBS'] = tstring_start
    pdata.meta['DATE-BEG'] = tstring_start
    pdata.meta['DATE-END'] = tstring_end
    pdata.meta['DATE-AVG'] = tstring_avg
    pdata.meta['DATE'] = (time_obs + time_delta + timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%S.000')

    pdata = update_spacecraft_location(pdata, time_obs)
    pdata = compute_celestial_from_helio(pdata)
    pdata = generate_uncertainty(pdata)
    write_ndcube_to_fits(pdata, path_output + get_base_file_name(pdata) + '.fits', skip_wcs_conversion=True)


def generate_l3_pnn(input_tb, input_pb, path_output, time_obs, time_delta):
    """Generate PNN - PUNCH Level-3 Polarized NFI Image"""
    # Define the mosaic WCS (helio)
    mosaic_shape = (4096, 4096)
    mosaic_wcs = WCS(naxis=2)
    mosaic_wcs.wcs.crpix = mosaic_shape[1] / 2 - 0.5, mosaic_shape[0] / 2 - 0.5
    mosaic_wcs.wcs.crval = 0, 0
    mosaic_wcs.wcs.cdelt = 0.0225, 0.0225
    mosaic_wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'
    mosaic_wcs = add_stokes_axis_to_wcs(mosaic_wcs, 2)

    # Define the NFI WCS (helio)
    nfi1_shape = [2048, 2048]
    nfi1_wcs = WCS(naxis=2)
    nfi1_wcs.wcs.crpix = nfi1_shape[1] / 2 - 0.5, nfi1_shape[0] / 2 - 0.5
    nfi1_wcs.wcs.crval = 0, 0
    nfi1_wcs.wcs.cdelt = 0.01, 0.01
    nfi1_wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'

    # Mask data to define the field of view
    mask = define_mask(shape=(4096, 4096), distance_value=0.155)

    # Read data and assemble into PUNCHData object
    pdata = assemble_punchdata(input_tb, input_pb, nfi1_wcs, product_code='PNN', product_level='3', mask=mask)

    reprojected_data = np.zeros((2, 2048, 2048), dtype=pdata.data.dtype)

    for i in np.arange(2):
        reprojected_data[i, :, :] = reproject.reproject_adaptive((pdata.data[i, :, :], mosaic_wcs[i]), nfi1_wcs,
                                                                 (2048, 2048),
                                                                 roundtrip_coords=False, return_footprint=False,
                                                                 kernel='Gaussian', boundary_mode='ignore')

    uncert = StdDevUncertainty(np.zeros(reprojected_data.shape))
    uncert.array[reprojected_data == 0] = 1

    meta = NormalizedMetadata.load_template('PNN', '3')
    tstring_start = time_obs.strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_end = (time_obs + time_delta).strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_avg = (time_obs + time_delta / 2).strftime('%Y-%m-%dT%H:%M:%S.000')
    meta['DATE-OBS'] = tstring_start
    meta['DATE-BEG'] = tstring_start
    meta['DATE-END'] = tstring_end
    meta['DATE-AVG'] = tstring_avg
    nfi1_wcs = add_stokes_axis_to_wcs(nfi1_wcs, 2)
    outdata = NDCube(data=reprojected_data, wcs=nfi1_wcs, meta=meta, uncertainty=uncert)

    # Update required metadata
    tstring_start = time_obs.strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_end = (time_obs + time_delta).strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_avg = (time_obs + time_delta / 2).strftime('%Y-%m-%dT%H:%M:%S.000')

    outdata.meta['DATE-OBS'] = tstring_start
    outdata.meta['DATE-BEG'] = tstring_start
    outdata.meta['DATE-END'] = tstring_end
    outdata.meta['DATE-AVG'] = tstring_avg
    outdata.meta['DATE'] = (time_obs + time_delta + timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%S.000')

    outdata = update_spacecraft_location(outdata, time_obs)
    outdata = compute_celestial_from_helio(outdata)
    outdata = generate_uncertainty(outdata)
    write_ndcube_to_fits(outdata, path_output + get_base_file_name(outdata) + '.fits', skip_wcs_conversion=True)


def generate_l3_pam(input_tb, input_pb, path_output, time_obs, time_delta):
    """Generate PAM - PUNCH Level-3 Polarized Low Noise Mosaic"""
    # Define the mosaic WCS (helio)
    mosaic_shape = (4096, 4096)
    mosaic_wcs = WCS(naxis=2)
    mosaic_wcs.wcs.crpix = mosaic_shape[1] / 2 - 0.5, mosaic_shape[0] / 2 - 0.5
    mosaic_wcs.wcs.crval = 0, 0
    mosaic_wcs.wcs.cdelt = 0.0225, 0.0225
    mosaic_wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'
    mosaic_wcs = add_stokes_axis_to_wcs(mosaic_wcs, 2)

    # Mask data to define the field of view
    mask = define_mask(shape=(4096, 4096), distance_value=0.68)

    # Read data and assemble into PUNCHData object
    pdata = assemble_punchdata(input_tb, input_pb, mosaic_wcs, product_code='PAM', product_level='3', mask=mask)

    # Update required metadata
    tstring_start = time_obs.strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_end = (time_obs + time_delta).strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_avg = (time_obs + time_delta / 2).strftime('%Y-%m-%dT%H:%M:%S.000')

    pdata.meta['DATE-OBS'] = tstring_start
    pdata.meta['DATE-BEG'] = tstring_start
    pdata.meta['DATE-END'] = tstring_end
    pdata.meta['DATE-AVG'] = tstring_avg
    pdata.meta['DATE'] = (time_obs + time_delta + timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%S.000')

    pdata = update_spacecraft_location(pdata, time_obs)
    pdata = compute_celestial_from_helio(pdata)
    pdata = generate_uncertainty(pdata)
    write_ndcube_to_fits(pdata,  path_output + get_base_file_name(pdata) + '.fits', skip_wcs_conversion=True)


def generate_l3_pan(input_tb, input_pb, path_output, time_obs, time_delta):
    """Generate PAN - PUNCH Level-3 Polarized Low Noise NFI Image"""
    # Define the mosaic WCS (helio)
    mosaic_shape = (4096, 4096)
    mosaic_wcs = WCS(naxis=2)
    mosaic_wcs.wcs.crpix = mosaic_shape[1] / 2 - 0.5, mosaic_shape[0] / 2 - 0.5
    mosaic_wcs.wcs.crval = 0, 0
    mosaic_wcs.wcs.cdelt = 0.0225, 0.0225
    mosaic_wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'
    mosaic_wcs = add_stokes_axis_to_wcs(mosaic_wcs, 2)

    # Define the NFI WCS (helio)
    nfi1_shape = [2048, 2048]
    nfi1_wcs = WCS(naxis=2)
    nfi1_wcs.wcs.crpix = nfi1_shape[1] / 2 - 0.5, nfi1_shape[0] / 2 - 0.5
    nfi1_wcs.wcs.crval = 0, 0
    nfi1_wcs.wcs.cdelt = 0.01, 0.01
    nfi1_wcs.wcs.ctype = 'HPLN-ARC', 'HPLT-ARC'

    # Mask data to define the field of view
    mask = define_mask(shape=(4096, 4096), distance_value=0.155)

    # Read data and assemble into PUNCHData object
    pdata = assemble_punchdata(input_tb, input_pb, nfi1_wcs, product_code='PAN', product_level='3', mask=mask)

    reprojected_data = np.zeros((2, 2048, 2048), dtype=pdata.data.dtype)

    for i in np.arange(2):
        reprojected_data[i, :, :] = reproject.reproject_adaptive((pdata.data[i, :, :], mosaic_wcs[i]), nfi1_wcs,
                                                                 (2048, 2048),
                                                                 roundtrip_coords=False, return_footprint=False,
                                                                 kernel='Gaussian', boundary_mode='ignore')

    uncert = StdDevUncertainty(np.zeros(reprojected_data.shape))
    uncert.array[reprojected_data == 0] = 1

    meta = NormalizedMetadata.load_template('PAN', '3')
    tstring_start = time_obs.strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_end = (time_obs + time_delta).strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_avg = (time_obs + time_delta / 2).strftime('%Y-%m-%dT%H:%M:%S.000')
    meta['DATE-OBS'] = tstring_start
    meta['DATE-BEG'] = tstring_start
    meta['DATE-END'] = tstring_end
    meta['DATE-AVG'] = tstring_avg
    meta['DATE'] = (time_obs + time_delta + timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%S.000')
    nfi1_wcs = add_stokes_axis_to_wcs(nfi1_wcs, 2)
    outdata = NDCube(data=reprojected_data, wcs=nfi1_wcs, meta=meta, uncertainty=uncert)

    # Update required metadata
    tstring_start = time_obs.strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_end = (time_obs + time_delta).strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_avg = (time_obs + time_delta / 2).strftime('%Y-%m-%dT%H:%M:%S.000')

    outdata.meta['DATE-OBS'] = tstring_start
    outdata.meta['DATE-BEG'] = tstring_start
    outdata.meta['DATE-END'] = tstring_end
    outdata.meta['DATE-AVG'] = tstring_avg
    outdata.meta['DATE'] = (time_obs + time_delta + timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%S.000')

    outdata = update_spacecraft_location(outdata, time_obs)
    outdata = compute_celestial_from_helio(outdata)
    outdata = generate_uncertainty(outdata)
    write_ndcube_to_fits(outdata, path_output + get_base_file_name(outdata) + '.fits', skip_wcs_conversion=True)


# @click.command()
# @click.argument('datadir', type=click.Path(exists=True))
# @click.argument('num_repeats', type=int, default=5)
def generate_l3_all(datadir, num_repeats):
    """Generate all level 3 synthetic data"""

    # Set file output path
    print(f"Running from {datadir}")
    outdir = os.path.join(datadir, 'synthetic_L3_v2/')
    os.makedirs(outdir, exist_ok=True)
    print(f"Outputting to {outdir}")

    # Parse list of model data
    files_tb = glob.glob(datadir + '/synthetic_cme/*_TB.fits')
    files_pb = glob.glob(datadir + '/synthetic_cme/*_PB.fits')
    print(f"Generating based on {len(files_tb)} TB files and {len(files_pb)} PB files.")
    files_tb.sort()
    files_pb.sort()

    # Stack and repeat these data for testing - about 25 times to get around 5 days of data
    files_tb = np.tile(files_tb, num_repeats)
    files_pb = np.tile(files_pb, num_repeats)

    # Set the overall start time for synthetic data
    # Note the timing for data products - 32 minutes / low noise ; 8 minutes / clear ; 4 minutes / polarized
    time_start = datetime(2024, 6, 20, 0, 0, 0)

    # Generate a corresponding set of observation times for polarized trefoil / NFI data
    time_delta = timedelta(minutes=4)
    times_obs = np.arange(len(files_tb)) * time_delta + time_start

    # Generate a corresponding set of observation times for low-noise mosaic / NFI data
    time_delta_ln = timedelta(minutes=32)

    rotation_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    pool = ProcessPoolExecutor()
    futures = []
    # Run individual generators
    for i, (file_tb, file_pb, time_obs) in tqdm(enumerate(zip(files_tb, files_pb, times_obs)), total=len(files_tb)):
        generate_l3_ptm(file_tb, file_pb, outdir, time_obs, time_delta, rotation_indices[i % 8])
        futures.append(pool.submit(generate_l3_ptm, file_tb, file_pb, outdir, time_obs, time_delta,
                                   rotation_indices[i % 8]))
        futures.append(pool.submit(generate_l3_pnn, file_tb, file_pb, outdir, time_obs, time_delta))

        if i % 8 == 0:
            futures.append(pool.submit(generate_l3_pam, file_tb, file_pb, outdir, time_obs, time_delta_ln))
            futures.append(pool.submit(generate_l3_pan, file_tb, file_pb, outdir, time_obs, time_delta_ln))

    with tqdm(total=len(futures)) as pbar:
        for _ in as_completed(futures):
            pbar.update(1)

    pool.shutdown()


if __name__ == '__main__':
    generate_l3_all('/Users/jhughes/Desktop/data/gamera_mosaic_jan2024/', 1)
