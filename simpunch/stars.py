import os

import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, NoConvergence
from numpy.fft import fft2, ifft2, ifftshift
from regularizepsf import ArrayCorrector

THIS_DIR = os.path.dirname(__file__)
HIPPARCOS_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat"


def load_catalog(
        catalog_path: str = os.path.join(THIS_DIR, "data/hip_main.dat"),
        url: str = HIPPARCOS_URL
):
    column_names = (
        'Catalog', 'HIP', 'Proxy', 'RAhms', 'DEdms', 'Vmag',
        'VarFlag', 'r_Vmag', 'RAdeg', 'DEdeg', 'AstroRef', 'Plx', 'pmRA',
        'pmDE', 'e_RAdeg', 'e_DEdeg', 'e_Plx', 'e_pmRA', 'e_pmDE', 'DE:RA',
        'Plx:RA', 'Plx:DE', 'pmRA:RA', 'pmRA:DE', 'pmRA:Plx', 'pmDE:RA',
        'pmDE:DE', 'pmDE:Plx', 'pmDE:pmRA', 'F1', 'F2', '---', 'BTmag',
        'e_BTmag', 'VTmag', 'e_VTmag', 'm_BTmag', 'B-V', 'e_B-V', 'r_B-V',
        'V-I', 'e_V-I', 'r_V-I', 'CombMag', 'Hpmag', 'e_Hpmag', 'Hpscat',
        'o_Hpmag', 'm_Hpmag', 'Hpmax', 'HPmin', 'Period', 'HvarType',
        'moreVar', 'morePhoto', 'CCDM', 'n_CCDM', 'Nsys', 'Ncomp',
        'MultFlag', 'Source', 'Qual', 'm_HIP', 'theta', 'rho', 'e_rho',
        'dHp', 'e_dHp', 'Survey', 'Chart', 'Notes', 'HD', 'BD', 'CoD',
        'CPD', '(V-I)red', 'SpType', 'r_SpType',
    )

    if not os.path.exists(catalog_path):
        response = requests.get(url)
        response.raise_for_status()
        with open(catalog_path, 'wb') as file:
            file.write(response.content)

    return pd.read_csv(catalog_path, sep="|", names=column_names, usecols=['HIP', 'Vmag', 'RAdeg', 'DEdeg'],
                     na_values=['     ', '       ', '        ', '            '])


def load_raw_hipparcos_catalog(
        catalog_path: str = os.path.join(THIS_DIR, "data/hip_main.dat"),
        url: str = HIPPARCOS_URL
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
        with open(catalog_path, 'wb') as file:
            file.write(response.content)

    df = pd.read_csv(
        catalog_path,
        sep="|",
        names=column_names,
        usecols=["HIP", "Vmag", "RAdeg", "DEdeg", "Plx"],
        na_values=["     ", "       ", "        ", "            "],
    )
    df["distance"] = 1000 / df["Plx"]
    df = df[df["distance"] > 0]
    return df.iloc[np.argsort(df["Vmag"])]


def filter_for_visible_stars(
    catalog: pd.DataFrame,
    dimmest_magnitude: float = 6
    ) -> pd.DataFrame:
    """Filters to only include stars brighter than a given magnitude

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
    mode: str = "all"
    ) -> np.ndarray:
    """Using the provided WCS converts the RA/DEC catalog into pixel coordinates

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
        print(np.array(catalog["RAdeg"]))
        print(np.array(catalog["DEdeg"]))
        xs, ys = SkyCoord(
            ra=np.array(catalog["RAdeg"]) * u.degree,
            dec=np.array(catalog["DEdeg"]) * u.degree,
            distance=np.array(catalog["distance"]) * u.parsec,
        ).to_pixel(wcs, mode=mode)
    except NoConvergence as e:
        xs, ys = e.best_solution[:, 0], e.best_solution[:, 1]
    bounds_mask = (0 <= xs) * (xs < image_shape[0]) * (0 <= ys) * (ys < image_shape[1])
    reduced_catalog = catalog[bounds_mask].copy()
    reduced_catalog["x_pix"] = xs[bounds_mask]
    reduced_catalog['y_pix'] = ys[bounds_mask]
    return reduced_catalog


def make_fake_stars_for_wfi(my_wcs,
                            array_corrector_path,
                            reference_mag=-1.44,
                            reference_flux=100_000,
                            psf_shape=(400, 400)):
    # load the catalog and compute fluxes
    catalog = load_catalog()
    catalog['flux'] = reference_flux * np.power(10, -0.4 * (catalog['Vmag'] - reference_mag))

    # determine which stars are in the FOV
    world_coords = my_wcs.wcs_world2pix(np.stack([catalog['RAdeg'], catalog['DEdeg']], axis=-1) * u.degree, 0)
    valid = np.where((0 <= world_coords[:, 0]) * (world_coords[:, 0] < 2048)
                     * (0 <= world_coords[:, 1]) * (world_coords[:, 1] < 2048))
    fluxes = catalog.flux.iloc[valid[0]]
    pixel_coords = world_coords[valid].astype(int)

    # Make a fake image and create point sources with those fluxes
    fake_image = np.zeros((2048, 2048))
    fake_image[pixel_coords[:, 1], pixel_coords[:, 0]] = fluxes

    # TODO: replace this portion of the code with corrector.simulate_observation when that gets added to regularizepsf
    # Load the PSF model and apply it
    corrector = ArrayCorrector.load(array_corrector_path)
    pad_shape = psf_shape
    img_shape = fake_image.shape

    xarr, yarr = np.meshgrid(np.arange(psf_shape[0]), np.arange(psf_shape[1]))
    apodization_window = np.sin((xarr + 0.5) * (np.pi / psf_shape[0])) * np.sin((yarr + 0.5) * (np.pi / psf_shape[1]))

    img_p = np.pad(fake_image, psf_shape, mode='constant')

    observation_synthetic = np.zeros(img_shape)
    observation_synthetic_p = np.pad(observation_synthetic, pad_shape)

    def get_img_i(x, y):
        xs, xe, ys, ye = x + psf_shape[0], x + 2 * psf_shape[0], y + psf_shape[1], y + 2 * psf_shape[1]
        return img_p[xs:xe, ys:ye]

    def set_synthetic_p(x, y, image):
        xs, xe, ys, ye = x + psf_shape[0], x + 2 * psf_shape[0], y + psf_shape[1], y + 2 * psf_shape[1]
        observation_synthetic_p[xs:xe, ys:ye] = np.nansum([image, observation_synthetic_p[xs:xe, ys:ye]], axis=0)

    for (x, y), psf_i in corrector._evaluations.items():
        img_i = get_img_i(x, y)
        out_i = np.real(ifftshift(ifft2(fft2(img_i * apodization_window) * fft2(psf_i)))) * apodization_window
        set_synthetic_p(x, y, out_i)

    return observation_synthetic_p[psf_shape[0]:img_shape[0] + psf_shape[0],
                  psf_shape[1]:img_shape[1] + psf_shape[1]]
