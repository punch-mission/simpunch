import astropy.units as u
from sunpy.coordinates import sun
from sunpy.coordinates.ephemeris import get_earth


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
