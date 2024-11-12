# ruff: noqa

from astropy.wcs import WCS
from ndcube import NDCube
from punchbowl.data import write_ndcube_to_fits
from punchbowl.data.io import load_ndcube_from_fits
from punchbowl.level1.quartic_fit import create_constant_quartic_coefficients

wfi_vignetting_model_path = "./build_3_review_files/PUNCH_L1_GM1_20240817174727_v2.fits"
nfi_vignetting_model_path = "./build_3_review_files/PUNCH_L1_GM4_20240819045110_v1.fits"

wfi_vignette = load_ndcube_from_fits(wfi_vignetting_model_path)
nfi_vignette = load_ndcube_from_fits(nfi_vignetting_model_path)

wfi_quartic = create_constant_quartic_coefficients((2048, 2048))
nfi_quartic = create_constant_quartic_coefficients((2048, 2048))

wfi_quartic[:, :, -2] /= wfi_vignette.data
nfi_quartic[:, :, -2] /= nfi_vignette.data


wfi_cube = NDCube(data=wfi_quartic, meta={"DSATVAL": 1E8}, wcs=WCS(naxis=3))
nfi_cube = NDCube(data=nfi_quartic, meta={"DSATVAL": 1E8}, wcs=WCS(naxis=5))

write_ndcube_to_fits(wfi_cube, "wfi_quartic_coeffs.fits", skip_stats=True)
write_ndcube_to_fits(nfi_cube, "nfi_quartic_coeffs.fits", skip_stats=True)
