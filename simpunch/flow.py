"""Run the entire pipeline backward."""
import os
from datetime import datetime, timedelta

import numpy as np
from prefect import flow
from prefect_dask import DaskTaskRunner

from simpunch.level0 import generate_l0_cr, generate_l0_pmzp
from simpunch.level1 import generate_l1_cr, generate_l1_pmzp
from simpunch.level2 import generate_l2_ctm, generate_l2_ptm
from simpunch.level3 import generate_l3_ctm, generate_l3_ptm


@flow(task_runner=DaskTaskRunner(address="localhost:8786"))
def generate_flow(file_tb: str,
                  file_pb: str,
                  time_obs: datetime,
                  out_dir: str,
                  backward_psf_model_path: str,
                  wfi_quartic_backward_model_path: str,
                  nfi_quartic_backward_model_path: str,
                  transient_probability: float = 0.03,
                  shift_pointing: bool = False) -> list[str]:
    """Generate all the products in the reverse pipeline."""
    i = int(os.path.basename(file_tb).split("_")[4][4:])  # TODO: make less specific to the filename
    rotation_indices = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    rotation_stage: int = rotation_indices[i % 8]

    l3_ptm = generate_l3_ptm(file_tb, file_pb, out_dir, time_obs, timedelta(minutes=4), rotation_stage)
    l3_ctm = generate_l3_ctm(file_tb, out_dir, time_obs, timedelta(minutes=4), rotation_stage)
    l2_ptm = generate_l2_ptm(l3_ptm, out_dir)
    l2_ctm = generate_l2_ctm(l3_ctm, out_dir)

    l1_polarized = []
    l1_clear = []
    for spacecraft in ["1", "2", "3", "4"]:
        l1_polarized.append(generate_l1_pmzp.submit(l2_ptm, out_dir, rotation_stage, spacecraft))
        l1_clear.append(generate_l1_cr.submit(l2_ctm, out_dir, rotation_stage, spacecraft))
    l1_polarized = [entry.result() for entry in l1_polarized]  # this is a list of lists
    l1_polarized = [item for sublist in l1_polarized for item in sublist]  # so we flatten it
    l1_clear = [entry.result() for entry in l1_clear]

    l0_pmzp = []
    for filename in l1_polarized:
        l0_pmzp.append(generate_l0_pmzp.submit(filename, out_dir, backward_psf_model_path,  # noqa: PERF401
                                               wfi_quartic_backward_model_path, nfi_quartic_backward_model_path,
                                                   transient_probability, shift_pointing))
    l0_pmzp = [entry.result() for entry in l0_pmzp]

    l0_cr = []
    for filename in l1_clear:
        l0_cr.append(generate_l0_cr.submit(filename, out_dir, backward_psf_model_path,  # noqa: PERF401
                                               wfi_quartic_backward_model_path, nfi_quartic_backward_model_path,
                                               transient_probability, shift_pointing))
    l0_cr = [entry.result() for entry in l0_cr]

    return [l3_ptm, l3_ctm, l2_ptm, l2_ctm] + l1_polarized + l1_clear + l0_pmzp + l0_cr  # noqa: RUF005
