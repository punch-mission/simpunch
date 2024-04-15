import os

import astropy.units as u
import numpy as np
import pandas as pd
from numpy.fft import fft2, ifft2, ifftshift
from regularizepsf import ArrayCorrector

THIS_DIR = os.path.dirname(__file__)


def load_catalog(catalog_path=os.path.join(THIS_DIR, "data/hip_main.dat.txt")):
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
    return pd.read_csv(catalog_path, sep="|", names=column_names, usecols=['HIP', 'Vmag', 'RAdeg', 'DEdeg'],
                     na_values=['     ', '       ', '        ', '            '])


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
