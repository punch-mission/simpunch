"""Utility functions."""
import astropy.time
import astropy.units as u
import numpy as np
from astropy.io import fits
from ndcube import NDCube
from punchbowl.data.wcs import get_p_angle
from sunpy.coordinates import sun
from sunpy.coordinates.ephemeris import get_earth


def update_spacecraft_location(input_data: NDCube, time_obs: astropy.time.Time) -> NDCube:
    """Update the spacecraft location metadata."""
    input_data.meta["GEOD_LAT"] = 0.
    input_data.meta["GEOD_LON"] = 0.
    input_data.meta["GEOD_ALT"] = 0.

    coord = get_earth(time_obs)
    coord.observer = "earth"

    # S/C Heliographic Stonyhurst
    input_data.meta["HGLN_OBS"] = coord.heliographic_stonyhurst.lon.value
    input_data.meta["HGLT_OBS"] = coord.heliographic_stonyhurst.lat.value

    # S/C Heliographic Carrington
    input_data.meta["CRLN_OBS"] = coord.heliographic_carrington.lon.value
    input_data.meta["CRLT_OBS"] = coord.heliographic_carrington.lat.value

    input_data.meta["DSUN_OBS"] = sun.earth_distance(time_obs).to(u.m).value

    # S/C Heliocentric Earth Ecliptic
    input_data.meta["HEEX_OBS"] = coord.heliocentricearthecliptic.cartesian.x.to(u.m).value
    input_data.meta["HEEY_OBS"] = coord.heliocentricearthecliptic.cartesian.y.to(u.m).value
    input_data.meta["HEEZ_OBS"] = coord.heliocentricearthecliptic.cartesian.z.to(u.m).value

    # S/C Heliocentric Inertial
    input_data.meta["HCIX_OBS"] = coord.heliocentricinertial.cartesian.x.to(u.m).value
    input_data.meta["HCIY_OBS"] = coord.heliocentricinertial.cartesian.y.to(u.m).value
    input_data.meta["HCIZ_OBS"] = coord.heliocentricinertial.cartesian.z.to(u.m).value

    # S/C Heliocentric Earth Equatorial
    input_data.meta["HEQX_OBS"] = (coord.heliographic_stonyhurst.cartesian.x.value * u.AU).to(u.m).value
    input_data.meta["HEQY_OBS"] = (coord.heliographic_stonyhurst.cartesian.y.value * u.AU).to(u.m).value
    input_data.meta["HEQZ_OBS"] = (coord.heliographic_stonyhurst.cartesian.z.value * u.AU).to(u.m).value

    input_data.meta["SOLAR_EP"] = get_p_angle(time_obs).to(u.deg).value
    input_data.meta["CAR_ROT"] = float(sun.carrington_rotation_number(time_obs))

    return input_data


def write_array_to_fits(path: str, image: np.ndarray, overwrite: bool = True) -> None:
    """Write an array to a FITS file using compression."""
    hdu_data = fits.CompImageHDU(data=image, name="Primary data array")
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu_data])
    hdul.writeto(path, overwrite=overwrite, checksum=True)
    hdul.close()


def generate_stray_light(shape: tuple, instrument:str="WFI")->tuple[np.ndarray, np.ndarray]:
    """
    Generate stray light arrays for B and pB channels for WFI and NFI instruments.

    Parameters
    ----------
    - shape: tuple, the shape of the output array (height, width).
    - instrument: str, either 'WFI' or 'NFI' specifying the formula to use.

    Returns
    -------
    - strayarray_B: 2D numpy array, intensity for B channel.
    - strayarray_pB: 2D numpy array, intensity for pB channel.
    """
    strayarray_b = np.zeros(shape)
    strayarray_pb = np.zeros(shape)

    y, x = np.indices(shape)

    if instrument == "WFI":
        center = (1024, 0)
        a, b, c = [-11.22425753, -0.03322594, 0.78295454]  # from empirical model using STEREO data
        x -= center[1]
        y -= center[0]
        r = np.sqrt(x ** 2 + y ** 2)
        r = (r * (160 / shape[0])) + 20
        def intensity_func(r:float, a:float, b:float, c:float)->float:
            return a + b * r ** c
    elif instrument == "NFI":
        center = (1024, 1024)
        a, b, c = [3796.09, -1399.18, 0.0003]   # from empirical model using STEREO data
        x -= center[1]
        y -= center[0]
        r = np.sqrt(x ** 2 + y ** 2)
        r_threshold = 150
        r = (r * (32 / center[0])) * (r > r_threshold)
        def intensity_func(r:float, a:float, b:float, c:float)->float:
            return a + b * np.exp(r ** c)
    else:
        msg = "Instrument must be 'WFI' or 'NFI'"
        raise ValueError(msg)

    # Calculate intensity for B channel
    intensity_b = intensity_func(r, a, b, c) - 2
    strayarray_b[:, :] = 10 ** intensity_b

    # Calculate intensity for pB channel (2 orders of magnitude less than B)
    intensity_pb = intensity_func(r, a - 2, b, c) - 2
    strayarray_pb[:, :] = 10 ** intensity_pb

    strayarray_b[~np.isfinite(strayarray_b)] = 0
    strayarray_pb[~np.isfinite(strayarray_pb)] = 0

    return strayarray_b, strayarray_pb
