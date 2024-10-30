"""Functions for starfield generation."""
import os

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, NoConvergence

THIS_DIR = os.path.dirname(__file__)
HIPPARCOS_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat"


def load_catalog(
        catalog_path: str = os.path.join(THIS_DIR, "data/hip_main.dat"),
        url: str = HIPPARCOS_URL,
) -> pd.DataFrame:
    """Load the Hipparcos catalog."""
    column_names = (
        "Catalog", "HIP", "Proxy", "RAhms", "DEdms", "Vmag",
        "VarFlag", "r_Vmag", "RAdeg", "DEdeg", "AstroRef", "Plx", "pmRA",
        "pmDE", "e_RAdeg", "e_DEdeg", "e_Plx", "e_pmRA", "e_pmDE", "DE:RA",
        "Plx:RA", "Plx:DE", "pmRA:RA", "pmRA:DE", "pmRA:Plx", "pmDE:RA",
        "pmDE:DE", "pmDE:Plx", "pmDE:pmRA", "F1", "F2", "---", "BTmag",
        "e_BTmag", "VTmag", "e_VTmag", "m_BTmag", "B-V", "e_B-V", "r_B-V",
        "V-I", "e_V-I", "r_V-I", "CombMag", "Hpmag", "e_Hpmag", "Hpscat",
        "o_Hpmag", "m_Hpmag", "Hpmax", "HPmin", "Period", "HvarType",
        "moreVar", "morePhoto", "CCDM", "n_CCDM", "Nsys", "Ncomp",
        "MultFlag", "Source", "Qual", "m_HIP", "theta", "rho", "e_rho",
        "dHp", "e_dHp", "Survey", "Chart", "Notes", "HD", "BD", "CoD",
        "CPD", "(V-I)red", "SpType", "r_SpType",
    )

    if not os.path.exists(catalog_path):
        response = requests.get(url)
        response.raise_for_status()
        with open(catalog_path, "wb") as file:
            file.write(response.content)

    return pd.read_csv(catalog_path, sep="|", names=column_names, usecols=["HIP", "Vmag", "RAdeg", "DEdeg"],
                     na_values=["     ", "       ", "        ", "            "])


def load_raw_hipparcos_catalog(
        catalog_path: str = os.path.join(THIS_DIR, "data/hip_main.dat"),
        url: str = HIPPARCOS_URL,
) -> pd.DataFrame:
    """Download hipparcos catalog from website.

    Parameters
    ----------
    catalog_path : str
        path to the Hipparcos catalog
    url : str
        url to the Hipparcos catalog for retrieval

    Returns
    -------
    pd.DataFrame
        loaded catalog with selected columns
    """
    column_names = (
        "Catalog",
        "HIP",
        "Proxy",
        "RAhms",
        "DEdms",
        "Vmag",
        "VarFlag",
        "r_Vmag",
        "RAdeg",
        "DEdeg",
        "AstroRef",
        "Plx",
        "pmRA",
        "pmDE",
        "e_RAdeg",
        "e_DEdeg",
        "e_Plx",
        "e_pmRA",
        "e_pmDE",
        "DE:RA",
        "Plx:RA",
        "Plx:DE",
        "pmRA:RA",
        "pmRA:DE",
        "pmRA:Plx",
        "pmDE:RA",
        "pmDE:DE",
        "pmDE:Plx",
        "pmDE:pmRA",
        "F1",
        "F2",
        "---",
        "BTmag",
        "e_BTmag",
        "VTmag",
        "e_VTmag",
        "m_BTmag",
        "B-V",
        "e_B-V",
        "r_B-V",
        "V-I",
        "e_V-I",
        "r_V-I",
        "CombMag",
        "Hpmag",
        "e_Hpmag",
        "Hpscat",
        "o_Hpmag",
        "m_Hpmag",
        "Hpmax",
        "HPmin",
        "Period",
        "HvarType",
        "moreVar",
        "morePhoto",
        "CCDM",
        "n_CCDM",
        "Nsys",
        "Ncomp",
        "MultFlag",
        "Source",
        "Qual",
        "m_HIP",
        "theta",
        "rho",
        "e_rho",
        "dHp",
        "e_dHp",
        "Survey",
        "Chart",
        "Notes",
        "HD",
        "BD",
        "CoD",
        "CPD",
        "(V-I)red",
        "SpType",
        "r_SpType",
    )

    if not os.path.exists(catalog_path):
        response = requests.get(url)
        response.raise_for_status()
        with open(catalog_path, "wb") as file:
            file.write(response.content)

    catalog_df = pd.read_csv(
        catalog_path,
        sep="|",
        names=column_names,
        usecols=["HIP", "Vmag", "RAdeg", "DEdeg", "Plx"],
        na_values=["     ", "       ", "        ", "            "],
    )
    catalog_df["distance"] = 1000 / catalog_df["Plx"]
    catalog_df = catalog_df[catalog_df["distance"] > 0]
    return catalog_df.iloc[np.argsort(catalog_df["Vmag"])]


def filter_for_visible_stars(
    catalog: pd.DataFrame,
    dimmest_magnitude: float = 6,
    ) -> pd.DataFrame:
    """Filter to only include stars brighter than a given magnitude.

    Parameters
    ----------
    catalog : pd.DataFrame
        a catalog data frame

    dimmest_magnitude : float
        the dimmest magnitude to keep

    Returns
    -------
    pd.DataFrame
        a catalog with stars dimmer than the `dimmest_magnitude` removed
    """
    return catalog[catalog["Vmag"] < dimmest_magnitude]


def find_catalog_in_image(
    catalog: pd.DataFrame,
    wcs: WCS,
    image_shape: (int, int),
    mode: str = "all",
    ) -> np.ndarray:
    """Convert the RA/DEC catalog into pixel coordinates using the provided WCS.

    Parameters
    ----------
    catalog : pd.DataFrame
        a catalog dataframe
    wcs : WCS
        the world coordinate system of a given image
    image_shape: (int, int)
        the shape of the image array associated with the WCS,
        used to only consider stars with coordinates in image
    mode : str
        either "all" or "wcs",
        see
        <https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord.to_pixel>

    Returns
    -------
    np.ndarray
        pixel coordinates of stars in catalog that are present in the image
    """
    try:
        xs, ys = SkyCoord(
            ra=np.array(catalog["RAdeg"]) * u.degree,
            dec=np.array(catalog["DEdeg"]) * u.degree,
            distance=np.array(catalog["distance"]) * u.parsec,
        ).to_pixel(wcs, mode=mode)
    except NoConvergence as e:
        xs, ys = e.best_solution[:, 0], e.best_solution[:, 1]
    bounds_mask = (xs >= 0) * (xs < image_shape[0]) * (ys >= 0) * (ys < image_shape[1])
    reduced_catalog = catalog[bounds_mask].copy()
    reduced_catalog["x_pix"] = xs[bounds_mask]
    reduced_catalog["y_pix"] = ys[bounds_mask]
    return reduced_catalog
