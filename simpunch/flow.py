"""Run the entire pipeline backward."""
from datetime import datetime, timedelta
from typing import List

import numpy as np
from prefect import flow

from simpunch.level0 import generate_l0_pmzp, generate_l0_cr
from simpunch.level1 import generate_l1_pmzp, generate_l1_cr
from simpunch.level2 import generate_l2_ptm, generate_l2_ctm
from simpunch.level3 import generate_l3_ptm, generate_l3_ctm


@flow
def generate_flow(file_tb: str,
                  file_pb: str,
                  out_dir: str,
                  start_time: datetime,
                  backward_psf_model_path: str,
                  wfi_quartic_backward_model_path: str,
                  nfi_quartic_backward_model_path: str,
                  transient_probability: float = 0.03,
                  shift_pointing: bool = False) -> List[str]:
    """Generate all the products in the reverse pipeline."""
    i = int(file_tb.split("_")[6][4:])
    rotation_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    rotation_stage = rotation_indices[i % 8]
    print(i, rotation_stage)
    time_obs = start_time + timedelta(minutes=i*4)
    l3_ptm = generate_l3_ptm(file_tb, file_pb, out_dir, time_obs, timedelta(minutes=4), rotation_stage)
    l3_ctm = generate_l3_ctm(file_tb, out_dir, time_obs, timedelta(minutes=4), rotation_stage)
    l2_ptm = generate_l2_ptm(l3_ptm, out_dir)
    l2_ctm = generate_l2_ctm(l3_ctm, out_dir)

    l1_polarized = []
    l1_clear = []
    for spacecraft in ["1", "2", "3", "4"]:
        l1_polarized.extend(generate_l1_pmzp(l2_ptm, out_dir, rotation_stage, spacecraft))
        l1_clear.append(generate_l1_cr(l2_ctm, out_dir, rotation_stage, spacecraft))

    l0_pmzp = []
    for filename in l1_polarized:
        l0_pmzp.append(generate_l0_pmzp(filename, out_dir, backward_psf_model_path,
                                               wfi_quartic_backward_model_path, nfi_quartic_backward_model_path,
                                               transient_probability, shift_pointing))

    l0_cr = []
    for filename in l1_clear:
        l0_cr.append(generate_l0_cr(filename, out_dir, backward_psf_model_path,
                                               wfi_quartic_backward_model_path, nfi_quartic_backward_model_path,
                                               transient_probability, shift_pointing))

    return [l3_ptm, l3_ctm, l2_ptm, l2_ctm] + l1_polarized + l1_clear + l0_pmzp + l0_cr