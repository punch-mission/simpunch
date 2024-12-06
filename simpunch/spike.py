"""Tools for generating realistic spikes."""
import os
import random
from glob import glob

import numpy as np

TESTDATA_DIR = os.path.dirname(__file__)
HIT_MAP_DIR = os.path.join(TESTDATA_DIR, "data/hit_maps/")
SPIKE_SCALING_MEAN = 2**16-5_000 # brightness of the average spike
SPIKE_SCALING_STD = SPIKE_SCALING_MEAN * 0.01 # width of the normal distribution for generating spike brightness
SPIKE_FREQUENCY = 40 * 49  # number of spikes in an image

def read_hit_map(path: str) -> np.ndarray:
    """Read in a hit map file and make an image of the spikes."""
    with open(path) as file:
        line = file.readline()
    num_spikes, width, height = map(int, line.split())
    table =  np.genfromtxt(path, skip_header=1)
    image = np.zeros((width, height), dtype=float)
    image[table[:, 0].astype(int), table[:, 1].astype(int)] = table[:, 2] / np.percentile(table[:, 2], 50)
    return image

def load_spike_library() -> np.ndarray:
    """Load all the spike images as a 3D numpy array."""
    paths = glob(HIT_MAP_DIR + "*.hmp")
    return np.array([read_hit_map(path) for path in paths])

def generate_spike_image(
        image_shape: (int, int),
        spike_frequency: int = SPIKE_FREQUENCY,
        spike_scaling_mean: float = SPIKE_SCALING_MEAN,
        spike_scaling_std: float = SPIKE_SCALING_STD,
        max_spike: int = 2**16,
        patch_size: int = 50,
        rotate: bool = True,
        transpose: bool = True,
        library: np.ndarray | None = None) -> np.ndarray:
    """Generate a realistic spike image."""
    if library is None:
        library = load_spike_library()

    spike_target = np.random.choice(image_shape[0] * image_shape[1], spike_frequency)
    spike_target_x, spike_target_y = np.unravel_index(spike_target, image_shape)

    spike_source = np.random.choice(np.prod(library.shape), spike_frequency)
    spike_source_i, spike_source_j, spike_source_k = np.unravel_index(spike_source, library.shape)

    spike_values = np.random.normal(spike_scaling_mean, spike_scaling_std, spike_frequency)

    spike_image = np.zeros(image_shape)

    for spike_value, i, j, k, x, y in zip(spike_values,
                                          spike_source_i, spike_source_j, spike_source_k,
                                          spike_target_x, spike_target_y, strict=False):
        patch = library[i, j:j+patch_size, k:k+patch_size] * spike_value
        pad_amount = tuple([(0, patch_size-s) for s in patch.shape])
        patch = np.pad(patch, pad_amount, mode="constant", constant_values=0)

        if transpose and random.random() < 0.5:  # noqa: PLR2004
            patch = patch.T
        if rotate:
            n = int(random.random() // 0.25)
            patch = np.rot90(patch, k=n)

        target_shape = spike_image[x:x+patch_size, y:y+patch_size].shape
        spike_image[x:x+patch_size, y:y+patch_size] = patch[:target_shape[0], :target_shape[1]]

    return np.clip(spike_image, 0, max_spike)
