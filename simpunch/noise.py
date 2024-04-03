import numpy as np


def generate_noise(
    data: np.ndarray,
    bias_level: float = 100,
    dark_level: float = 55.81,
    gain: float = 4.3,
    read_noise_level: float = 17,
    bitrate_signal: int = 16,
) -> np.ndarray:
    """
    Generates noise based on an input data array, with specified noise parameters

    Parameters
    ----------
    data
        input data array (n x n)
    bias_level
        ccd bias level
    dark_level
        ccd dark level
    gain
        ccd gain
    read_noise_level
        ccd read noise level
    bitrate_signal
        desired ccd data bit level

    Returns
    -------
    np.ndarray
        computed noise array corresponding to input data and ccd/noise parameters

    """

    # Generate a copy of the input signal
    data_signal = np.copy(data)

    # Convert / scale data
    # Think of this as the raw signal input into the camera
    data = np.interp(
        data_signal,
        (np.min(data_signal), np.max(data_signal)),
        (0, 2**bitrate_signal - 1),
    )
    data = data.astype("long")

    # Add bias level and clip pixels to avoid overflow
    data = np.clip(data + bias_level, 0, 2**bitrate_signal - 1)

    # Photon / shot noise generation
    data_photon = data_signal * gain  # DN to photoelectrons
    sigma_photon = np.sqrt(data_photon)  # Converting sigma of this
    sigma = sigma_photon / gain  # Converting back to DN
    noise_photon = np.random.normal(scale=sigma)

    # Dark noise generation
    noise_level = dark_level * gain
    noise_dark = np.random.poisson(lam=noise_level, size=data.shape) / gain

    # Read noise generation
    noise_read = np.random.normal(scale=read_noise_level, size=data.shape)
    noise_read = noise_read / gain  # Convert back to DN

    # And then add noise terms directly
    return noise_photon + noise_dark + noise_read
