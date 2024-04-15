import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from punchbowl.data import PUNCHData

from simpunch.stars import make_fake_stars_for_wfi


def make_fake_level0(path_to_input, array_corrector_path, path_to_save):
    with fits.open(path_to_input) as hdul:
        test_header = hdul[0].header
    test_wcs = WCS(test_header)

    fake_star_data = make_fake_stars_for_wfi(test_wcs, array_corrector_path)

    my_data = PUNCHData(data=fake_star_data, wcs=test_wcs, uncertainty=np.zeros_like(fake_star_data))
    my_data.write(path_to_save)


if __name__ == "__main__":
    make_fake_level0("/Users/jhughes/Nextcloud/23103_PUNCH_Data/SOC_Data/PUNCH_WFI_EM_Starfield_campaign2_night2_phase3/calibrated/campaign2_night2_phase3_calibrated_000.new",
                     "/Users/jhughes/Desktop/projects/PUNCH/psf_paper/paper-variable-point-spread-functions/scripts/punch_array_corrector.h5",
                     "../fake_data.fits")
