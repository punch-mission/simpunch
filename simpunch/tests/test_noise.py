import numpy as np

from simpunch.noise import generate_noise


def test_noise_generation() -> None:
    """Verify noise generates."""
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2**16)
    noise_arr = generate_noise(arr)

    assert noise_arr.shape == arr.shape
