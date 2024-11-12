# ruff: noqa
from astropy.io.fits import CompImageHDU, HDUList, ImageHDU, PrimaryHDU
from punchbowl.data.io import load_ndcube_from_fits
from punchbowl.level1.quartic_fit import create_constant_quartic_coefficients

wfi_vignetting_model_path = "./build_3_review_files/PUNCH_L1_GM1_20240817174727_v2.fits"
nfi_vignetting_model_path = "./build_3_review_files/PUNCH_L1_GM4_20240819045110_v1.fits"

wfi_vignette = load_ndcube_from_fits(wfi_vignetting_model_path).data[...] + 1E-8
nfi_vignette = load_ndcube_from_fits(nfi_vignetting_model_path).data[...] + 1E-8

wfi_quartic = create_constant_quartic_coefficients((2048, 2048)).T
nfi_quartic = create_constant_quartic_coefficients((2048, 2048)).T

wfi_quartic[-2, :, :] /= wfi_vignette
nfi_quartic[-2, :, :] /= nfi_vignette

HDUList([PrimaryHDU(), CompImageHDU(data=wfi_quartic)]).writeto("wfi_quartic_coeffs.fits")
HDUList([PrimaryHDU(), CompImageHDU(data=nfi_quartic)]).writeto("nfi_quartic_coeffs.fits")
