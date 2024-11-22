"""Run the entire pipeline backward."""
import glob
import os
import shutil
from datetime import datetime

import numpy as np
from asyncpg.pgproto.pgproto import timedelta
from prefect import flow

from simpunch.level0 import generate_l0_all
from simpunch.level1 import generate_l1_all
from simpunch.level2 import generate_l2_all
from simpunch.level3 import generate_l3_all, generate_l3_all_fixed


@flow(log_prints=True)
def generate_flow(gamera_directory: str,
                  output_directory: str,
                  forward_psf_model_path: str,
                  backward_psf_model_path: str,
                  wfi_quartic_backward_model_path: str,
                  nfi_quartic_backward_model_path: str,
                  wfi_quartic_model_path: str,
                  nfi_quartic_model_path: str,
                  num_repeats: int = 1,
                  start_time: datetime | None = None,
                  transient_probability: float = 0.03,
                  shift_pointing: bool = False,
                  generate_new: bool = True,
                  update_database: bool = True) -> None:
    """Generate all the products in the reverse pipeline."""
    if start_time is None:
        start_time = datetime.now() # noqa: DTZ005

    if generate_new:
        time_delta = timedelta(days=0.25)
        files_tb = sorted(glob.glob(gamera_directory + "/synthetic_cme/*_TB.fits"))
        files_pb = sorted(glob.glob(gamera_directory + "/synthetic_cme/*_PB.fits"))

        previous_month = np.linspace(1, -30, int(timedelta(days=30)/time_delta)) * time_delta + start_time
        generate_l3_all_fixed(gamera_directory, previous_month, files_pb[0], files_tb[0])

        next_month = np.linspace(1, 30, int(timedelta(days=30)/time_delta)) * time_delta + start_time
        generate_l3_all_fixed(gamera_directory, next_month, files_pb[-1], files_tb[-1])

        generate_l3_all(gamera_directory, output_directory, start_time, num_repeats=num_repeats)
        generate_l2_all(gamera_directory, output_directory)
        generate_l1_all(gamera_directory, output_directory)
        generate_l0_all(gamera_directory,
                        output_directory,
                        backward_psf_model_path,
                        wfi_quartic_backward_model_path,
                        nfi_quartic_backward_model_path,
                        shift_pointing=shift_pointing,
                        transient_probability=transient_probability)

        model_time = start_time - timedelta(days=35)
        model_time_str = model_time.strftime("%Y%m%d%H%M%S")

        # duplicate the psf model to all required versions
        for type_code in ["RM", "RZ", "RP", "RC"]:
            for obs_code in ["1", "2", "3", "4"]:
                new_name = 	f"PUNCH_L1_{type_code}{obs_code}_{model_time_str}_v1.fits"
                shutil.copy(forward_psf_model_path, os.path.join(output_directory, f"synthetic_l0/{new_name}"))

        # duplicate the quartic model
        type_code = "FQ"
        for obs_code in ["1", "2", "3"]:
            new_name = 	f"PUNCH_L1_{type_code}{obs_code}_{model_time_str}_v1.fits"
            shutil.copy(wfi_quartic_model_path, os.path.join(output_directory, f"synthetic_l0/{new_name}"))
        obs_code = "4"
        new_name = f"PUNCH_L1_{type_code}{obs_code}_{model_time_str}_v1.fits"
        shutil.copy(nfi_quartic_model_path, os.path.join(output_directory, f"synthetic_l0/{new_name}"))

    if update_database:
        from punchpipe import __version__
        from punchpipe.control.db import File
        from punchpipe.control.util import get_database_session
        db_session = get_database_session()
        for file_path in sorted(glob.glob(os.path.join(output_directory, "synthetic_l0/*v[0-9].fits")),
                                key=lambda s: os.path.basename(s)[13:27]):
            file_name = os.path.basename(file_path)
            level = file_name[7]
            file_type = file_name[9:11]
            observatory = file_name[11]
            year = file_name[13:17]
            month = file_name[17:19]
            day = file_name[19:21]
            hour = file_name[21:23]
            minute = file_name[23:25]
            second = file_name[25:27]
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            version = file_name.split(".fits")[0].split("_")[-1][1:]

            output_dir = os.path.join(output_directory, level, file_type+observatory, year, month, day)
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy(file_path, os.path.join(output_dir, file_name))

            db_entry = File(
                level=level,
                file_type=file_type,
                observatory=observatory,
                file_version=version,
                software_version=__version__,
                date_obs=dt,
                polarization=file_type[1],
                state="created",
            )
            db_session.add(db_entry)
            db_session.commit()
