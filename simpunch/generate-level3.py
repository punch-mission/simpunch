"""
Generates synthetic level 3 data
PAM - PUNCH Level-3 Polarized Low Noise Mosaic
PAN - PUNCH Level-3 Polarized Low Noise NFI Image

PTM - PUNCH Level-3 Polarized Mosaic
PNN - PUNCH Level-3 Polarized NFI Image
"""

import numpy as np
import glob

from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs.utils import add_stokes_axis_to_wcs
from astropy.time import Time
from astropy.coordinates import get_sun
import reproject
import scipy.ndimage


from datetime import datetime, timedelta

from punchbowl.data import PUNCHData, NormalizedMetadata


def get_sun_ra_dec(dt: datetime):
    position =  get_sun(Time(str(dt), scale='utc'))
    return position.ra.value, position.dec.value


def define_mask(shape=(4096, 4096), distance_value=0.68):
    """Define a mask to describe the FOV for low-noise PUNCH data products"""
    center = (int(shape[0] / 2), int(shape[1] / 2))
    radius = min(center[0], center[1], shape[0] - center[0], shape[1] - center[1])

    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_arr = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = (dist_arr / dist_arr.max()) < distance_value

    return mask


def define_trefoil_mask(rotation_stage=0):
    """Define a mask to describe the FOV for trefoil mosaic PUNCH data products"""

    trefoil_mask = np.load('data/trefoil_mask.npz')['trefoil_mask'][rotation_stage,:,:]

    return trefoil_mask


def assemble_punchdata(input_tb, input_pb, wcs, product_code, product_level, mask=None):
    """Assemble a punchdata object with correct metadata"""

    with fits.open(input_tb) as hdul:
        data_tb = hdul[1].data
        data_tb = scipy.ndimage.zoom(data_tb, 2, order=0)
        data_tb[np.where(data_tb == -9999.0)] = 0
        if mask is not None:
            data_tb = data_tb * mask
    with fits.open(input_pb) as hdul:
        data_pb = hdul[1].data
        data_pb = scipy.ndimage.zoom(data_pb, 2, order=0)
        data_pb[np.where(data_pb == -9999.0)] = 0
        if mask is not None:
            data_pb = data_pb * mask

    datacube = np.stack([data_tb, data_pb]).astype('float32')

    # TODO - Data / uncertainty scaling?

    uncert = (np.sqrt(datacube) / np.sqrt(datacube).max() * 255).astype('uint8')

    meta = NormalizedMetadata.load_template(product_code, product_level)
    data = PUNCHData(data=datacube, wcs=wcs, meta=meta, uncertainty=uncert)

    return data


def generate_l3_ptm(input_tb, input_pb, path_output, time_obs, time_delta, rotation_stage):
    """Generate PTM - PUNCH Level-3 Polarized Mosaic"""

    # Define the mosaic WCS
    mosaic_shape = (4096, 4096)
    mosaic_wcs = WCS(naxis=2)
    mosaic_wcs.wcs.crpix = mosaic_shape[1] / 2 - 0.5, mosaic_shape[0] / 2 - 0.5
    mosaic_wcs.wcs.crval = get_sun_ra_dec(time_obs + time_delta)
    mosaic_wcs.wcs.cdelt = 0.0225, 0.0225
    mosaic_wcs.wcs.ctype = "RA---ARC", "DEC--ARC"
    mosaic_wcs = add_stokes_axis_to_wcs(mosaic_wcs, 0)

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

    # Write out
    pdata.write(path_output + pdata.filename_base + '.fits')


def generate_l3_pnn(input_tb, input_pb, path_output, time_obs, time_delta):
    """Generate PNN - PUNCH Level-3 Polarized NFI Image"""

    # Define the mosaic WCS
    mosaic_shape = (4096, 4096)
    mosaic_wcs = WCS(naxis=2)
    mosaic_wcs.wcs.crpix = mosaic_shape[1] / 2 - 0.5, mosaic_shape[0] / 2 - 0.5
    mosaic_wcs.wcs.crval = get_sun_ra_dec(time_obs + time_delta)
    mosaic_wcs.wcs.cdelt = 0.0225, 0.0225
    mosaic_wcs.wcs.ctype = "RA---ARC", "DEC--ARC"

    # Define the NFI WCS
    nfi1_shape = [2048, 2048]
    nfi1_wcs = WCS(naxis=2)
    nfi1_wcs.wcs.crpix = nfi1_shape[1] / 2 - 0.5, nfi1_shape[0] / 2 - 0.5
    nfi1_wcs.wcs.crval = get_sun_ra_dec(time_obs + time_delta)
    nfi1_wcs.wcs.cdelt = 0.01, 0.01
    nfi1_wcs.wcs.ctype = "RA---TAN", "DEC--TAN"

    # Mask data to define the field of view
    mask = define_mask(shape=(4096, 4096), distance_value=0.155)

    # Read data and assemble into PUNCHData object
    pdata = assemble_punchdata(input_tb, input_pb, nfi1_wcs, product_code='PNN', product_level='3', mask=mask)

    reprojected_data = np.zeros((2, 2048, 2048), dtype=pdata.data.dtype)

    for i in np.arange(2):
        reprojected_data[i, :, :] = reproject.reproject_adaptive((pdata.data[i, :, :], mosaic_wcs), nfi1_wcs,
                                                                 (2048, 2048),
                                                                 roundtrip_coords=False, return_footprint=False,
                                                                 kernel='Gaussian', boundary_mode='ignore')

    uncert = (np.sqrt(reprojected_data) / np.sqrt(reprojected_data).max() * 255).astype('uint8')

    meta = NormalizedMetadata.load_template('PAN', '3')
    tstring_start = time_obs.strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_end = (time_obs + time_delta).strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_avg = (time_obs + time_delta / 2).strftime('%Y-%m-%dT%H:%M:%S.000')
    meta['DATE-OBS'] = tstring_start
    meta['DATE-BEG'] = tstring_start
    meta['DATE-END'] = tstring_end
    meta['DATE-AVG'] = tstring_avg
    nfi1_wcs = add_stokes_axis_to_wcs(nfi1_wcs, 0)
    outdata = PUNCHData(data=reprojected_data, wcs=nfi1_wcs, meta=meta, uncertainty=uncert)

    # Update required metadata
    tstring_start = time_obs.strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_end = (time_obs + time_delta).strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_avg = (time_obs + time_delta / 2).strftime('%Y-%m-%dT%H:%M:%S.000')

    pdata.meta['DATE-OBS'] = tstring_start
    pdata.meta['DATE-BEG'] = tstring_start
    pdata.meta['DATE-END'] = tstring_end
    pdata.meta['DATE-AVG'] = tstring_avg
    pdata.meta['DATE'] = (time_obs + time_delta + timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%S.000')

    # Write out
    outdata.write(path_output + pdata.filename_base + '.fits')


def generate_l3_pam(input_tb, input_pb, path_output, time_obs, time_delta):
    """Generate PAM - PUNCH Level-3 Polarized Low Noise Mosaic"""

    # Define the mosaic WCS
    mosaic_shape = (4096, 4096)
    mosaic_wcs = WCS(naxis=2)
    mosaic_wcs.wcs.crpix = mosaic_shape[1] / 2 - 0.5, mosaic_shape[0] / 2 - 0.5
    mosaic_wcs.wcs.crval = get_sun_ra_dec(time_obs + time_delta)
    mosaic_wcs.wcs.cdelt = 0.0225, 0.0225
    mosaic_wcs.wcs.ctype = "RA---ARC", "DEC--ARC"
    mosaic_wcs = add_stokes_axis_to_wcs(mosaic_wcs, 0)


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

    # Write out
    pdata.write(path_output + pdata.filename_base + '.fits')


def generate_l3_pan(input_tb, input_pb, path_output, time_obs, time_delta):
    """Generate PAN - PUNCH Level-3 Polarized Low Noise NFI Image"""

    # Define the mosaic WCS
    mosaic_shape = (4096, 4096)
    mosaic_wcs = WCS(naxis=2)
    mosaic_wcs.wcs.crpix = mosaic_shape[1] / 2 - 0.5, mosaic_shape[0] / 2 - 0.5
    mosaic_wcs.wcs.crval = get_sun_ra_dec(time_obs + time_delta)
    mosaic_wcs.wcs.cdelt = 0.0225, 0.0225
    mosaic_wcs.wcs.ctype = "RA---ARC", "DEC--ARC"

    # Define the NFI WCS
    nfi1_shape = [2048, 2048]
    nfi1_wcs = WCS(naxis=2)
    nfi1_wcs.wcs.crpix = nfi1_shape[1] / 2 - 0.5, nfi1_shape[0] / 2 - 0.5
    nfi1_wcs.wcs.crval = get_sun_ra_dec(time_obs + time_delta)
    nfi1_wcs.wcs.cdelt = 0.01, 0.01
    nfi1_wcs.wcs.ctype = "RA---TAN", "DEC--TAN"

    # Mask data to define the field of view
    mask = define_mask(shape=(4096, 4096), distance_value=0.155)

    # Read data and assemble into PUNCHData object
    pdata = assemble_punchdata(input_tb, input_pb, nfi1_wcs, product_code='PAN', product_level='3', mask=mask)

    reprojected_data = np.zeros((2, 2048, 2048), dtype=pdata.data.dtype)

    for i in np.arange(2):
        reprojected_data[i, :, :] = reproject.reproject_adaptive((pdata.data[i, :, :], mosaic_wcs), nfi1_wcs,
                                                                 (2048, 2048),
                                                                 roundtrip_coords=False, return_footprint=False,
                                                                 kernel='Gaussian', boundary_mode='ignore')

    uncert = (np.sqrt(reprojected_data) / np.sqrt(reprojected_data).max() * 255).astype('uint8')

    meta = NormalizedMetadata.load_template('PAN', '3')
    tstring_start = time_obs.strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_end = (time_obs + time_delta).strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_avg = (time_obs + time_delta / 2).strftime('%Y-%m-%dT%H:%M:%S.000')
    meta['DATE-OBS'] = tstring_start
    meta['DATE-BEG'] = tstring_start
    meta['DATE-END'] = tstring_end
    meta['DATE-AVG'] = tstring_avg
    meta['DATE'] = (time_obs + time_delta + timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%S.000')
    nfi1_wcs = add_stokes_axis_to_wcs(nfi1_wcs, 0)
    outdata = PUNCHData(data=reprojected_data, wcs=nfi1_wcs, meta=meta, uncertainty=uncert)

    # Update required metadata
    tstring_start = time_obs.strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_end = (time_obs + time_delta).strftime('%Y-%m-%dT%H:%M:%S.000')
    tstring_avg = (time_obs + time_delta / 2).strftime('%Y-%m-%dT%H:%M:%S.000')

    pdata.meta['DATE-OBS'] = tstring_start
    pdata.meta['DATE-BEG'] = tstring_start
    pdata.meta['DATE-END'] = tstring_end
    pdata.meta['DATE-AVG'] = tstring_avg
    pdata.meta['DATE'] = (time_obs + time_delta + timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M:%S.000')

    # Write out
    outdata.write(path_output + pdata.filename_base + '.fits')


def generate_l3_all(datadir='/Users/jhughes/Desktop/data/GAMERA Mosaic data TB and PB jan 2024 /'):
    """Generate all level 3 synthetic data"""

    # Set file output path
    outdir = datadir + 'synthetic_L3/'

    # Parse list of model data
    print(datadir + 'synthetic_cme/*_TB*.fits')
    files_tb = glob.glob(datadir + 'synthetic_cme/*_TB*.fits')
    files_pb = glob.glob(datadir + 'synthetic_cme/*_PB*.fits')
    files_tb.sort()
    files_pb.sort()
    print(len(files_tb), len(files_pb))

    # Stack and repeat these data for testing - about 25 times to get around 5 days of data
    files_tb = np.tile(files_tb, 25)
    files_pb = np.tile(files_pb, 25)

    # Set the overall start time for synthetic data
    # Note the timing for data products - 32 minutes / low noise ; 8 minutes / clear ; 4 minutes / polarized
    time_start = datetime(2023, 7, 4, 0, 0, 0)

    # Generate a corresponding set of observation times for polarized trefoil / NFI data
    time_delta = timedelta(minutes=4)
    times_obs = np.arange(len(files_tb)) * time_delta + time_start

    # Generate a corresponding set of observation times for low-noise mosaic / NFI data
    time_delta_ln = timedelta(minutes=32)

    # Run individual generators
    for i, (file_tb, file_pb, time_obs) in enumerate(zip(files_tb, files_pb, times_obs)):
        rotation_stage = i % 8
        # print(str(rotation_stage) + ' - PTM / PNN - ' + time_obs.strftime('%Y-%m-%dT%H:%M:%S.000'))
        generate_l3_ptm(file_tb, file_pb, outdir, time_obs, time_delta, rotation_stage)
        generate_l3_pnn(file_tb, file_pb, outdir, time_obs, time_delta)

        if rotation_stage == 0:
            # print(str(rotation_stage) + ' - PAM / PAN - ' + time_obs.strftime('%Y-%m-%dT%H:%M:%S.000'))
            generate_l3_pam(file_tb, file_pb, outdir, time_obs, time_delta_ln)
            generate_l3_pan(file_tb, file_pb, outdir, time_obs, time_delta_ln)


if __name__ == "__main__":
    generate_l3_all()
