import astropy.units as u
import numpy as np
from astropy.coordinates import GCRS, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import frames


def test_helio_celestial_wcs() -> None:
    """Test WCS conversions."""
    header = fits.Header.fromtextfile("simpunch/tests/example_header.txt")

    wcs_helio = WCS(header)
    wcs_celestial = WCS(header, key="A")

    date_obs = Time(header["DATE-OBS"], format="isot", scale="utc")
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    npoints = 20
    input_coords = np.stack([
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int),
                             np.ones(npoints, dtype=int)], axis=1)

    points_celestial = wcs_celestial.all_pix2world(input_coords, 0)
    points_helio = wcs_helio.all_pix2world(input_coords, 0)

    output_coords = []
    for _c_pix, c_celestial, _c_helio in zip(input_coords, points_celestial, points_helio, strict=False):
        skycoord_celestial = SkyCoord(c_celestial[0] * u.deg, c_celestial[1] * u.deg,
                                      frame=GCRS,
                                      obstime=date_obs,
                                      observer=test_gcrs,
                                      obsgeoloc=test_gcrs.cartesian,
                                      obsgeovel=test_gcrs.velocity.to_cartesian(),
                                      distance=test_gcrs.hcrs.distance,
                                      )

        intermediate = skycoord_celestial.transform_to(frames.Helioprojective)
        output_coords.append(wcs_helio.all_world2pix(intermediate.data.lon.to(u.deg).value,
                                                     intermediate.data.lat.to(u.deg).value, 2, 0))

    output_coords = np.array(output_coords)
    distances = np.linalg.norm(input_coords - output_coords, axis=1)

    assert np.nanmean(distances) < 0.1  # noqa: PLR2004
