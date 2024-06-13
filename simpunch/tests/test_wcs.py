import os
from datetime import datetime

import astropy
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import GCRS, ICRS, EarthLocation, SkyCoord, get_sun
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.wcs import WCS
from ndcube import NDCube
from numpy.linalg import inv
from pytest import fixture
from sunpy import sun
from sunpy.coordinates import frames

from punchbowl.data import (
    History,
    HistoryEntry,
    MetaField,
    NormalizedMetadata,
    PUNCHData,
    calculate_helio_wcs_from_celestial,
    load_spacecraft_def,
    load_trefoil_wcs,
)


def test_helio_celestial_wcs():
    filename = '/Users/clowder/data/punch/synthetic_L3_testing/PUNCH_L3_PAM_20240620000000.fits'

    with fits.open(filename) as hdul:
        data = hdul[1].data
        header = hdul[1].header

    wcs_helio = WCS(header)
    wcs_celestial = WCS(header, key='A')

    date_obs = Time(header['DATE-OBS'], format='isot', scale='utc')
    test_loc = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    test_gcrs = SkyCoord(test_loc.get_gcrs(date_obs))

    npoints = 20
    input_coords = np.stack([
                             np.linspace(0, 4096, npoints).astype(int),
                             np.linspace(0, 4096, npoints).astype(int),
                             np.ones(npoints, dtype=int),], axis=1)

    points_celestial = wcs_celestial.all_pix2world(input_coords, 0)
    points_helio = wcs_helio.all_pix2world(input_coords, 0)

    output_coords = []
    for c_pix, c_celestial, c_helio in zip(input_coords, points_celestial, points_helio):
        skycoord_celestial = SkyCoord(c_celestial[0] * u.deg, c_celestial[1] * u.deg,
                                      frame=GCRS,
                                      obstime=date_obs,
                                      observer=test_gcrs,
                                      obsgeoloc=test_gcrs.cartesian,
                                      obsgeovel=test_gcrs.velocity.to_cartesian(),
                                      distance=test_gcrs.hcrs.distance
                                      )

        intermediate = skycoord_celestial.transform_to(frames.Helioprojective)
        output_coords.append(wcs_helio.all_world2pix(intermediate.data.lon.to(u.deg).value,
                                                     intermediate.data.lat.to(u.deg).value, 2, 0))

    output_coords = np.array(output_coords)
    distances = np.linalg.norm(input_coords - output_coords, axis=1)

    assert np.nanmean(distances) < 0.1
