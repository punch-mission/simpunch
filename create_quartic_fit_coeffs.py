# ruff: noqa
from datetime import datetime

from astropy.io.fits import CompImageHDU, HDUList, ImageHDU, PrimaryHDU
from astropy.wcs import WCS
from astropy.wcs.docstrings import naxis
from ndcube import NDCube
from punchbowl.data import NormalizedMetadata, write_ndcube_to_fits
from punchbowl.data.io import load_ndcube_from_fits
from punchbowl.level1.quartic_fit import create_constant_quartic_coefficients

# backward
wfi_vignetting_model_path = "./build_3_review_files/PUNCH_L1_GM1_20240817174727_v2.fits"
nfi_vignetting_model_path = "./build_3_review_files/PUNCH_L1_GM4_20240819045110_v1.fits"

wfi_vignette = load_ndcube_from_fits(wfi_vignetting_model_path).data[...] + 1E-8
nfi_vignette = load_ndcube_from_fits(nfi_vignetting_model_path).data[...] + 1E-8

wfi_quartic = create_constant_quartic_coefficients((2048, 2048))
nfi_quartic = create_constant_quartic_coefficients((2048, 2048))

wfi_quartic[-2, :, :] = wfi_vignette
nfi_quartic[-2, :, :] = nfi_vignette

meta = NormalizedMetadata.load_template("FQ1", "1")
meta['DATE-OBS'] = str(datetime.now())

wfi_cube = NDCube(data=wfi_quartic, meta=meta, wcs=WCS(naxis=3))
nfi_cube = NDCube(data=nfi_quartic, meta=meta, wcs=WCS(naxis=3))

write_ndcube_to_fits(wfi_cube, "wfi_quartic_backward_coeffs.fits")
write_ndcube_to_fits(nfi_cube, "nfi_quartic_backward_coeffs.fits")

# forward
wfi_vignetting_model_path = "./build_3_review_files/PUNCH_L1_GM1_20240817174727_v2.fits"
nfi_vignetting_model_path = "./build_3_review_files/PUNCH_L1_GM4_20240819045110_v1.fits"

wfi_vignette = load_ndcube_from_fits(wfi_vignetting_model_path).data[...] + 1E-8
nfi_vignette = load_ndcube_from_fits(nfi_vignetting_model_path).data[...] + 1E-8

wfi_quartic = create_constant_quartic_coefficients((2048, 2048))
nfi_quartic = create_constant_quartic_coefficients((2048, 2048))

wfi_quartic[-2, :, :] = 1/wfi_vignette
nfi_quartic[-2, :, :] = 1/nfi_vignette

meta = NormalizedMetadata.load_template("FQ1", "1")
meta['DATE-OBS'] = str(datetime.now())

wfi_cube = NDCube(data=wfi_quartic, meta=meta, wcs=WCS(naxis=3))
nfi_cube = NDCube(data=nfi_quartic, meta=meta, wcs=WCS(naxis=3))

write_ndcube_to_fits(wfi_cube, "wfi_quartic_coeffs.fits")
write_ndcube_to_fits(nfi_cube, "nfi_quartic_coeffs.fits")
