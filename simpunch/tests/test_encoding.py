import numpy as np
from simpunch.encoding import encode, decode, decode_top


def test_encoding():
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2**16)

    encoded_arr = encode(arr, frombits=16, tobits=10)

    assert encoded_arr.shape == arr.shape
    assert np.max(encoded_arr) <= 2**10


def test_decoding():
    arr_dim = 2048
    arr = np.random.random([arr_dim, arr_dim]) * (2**10)

    decoded_arr = decode(arr, frombits=10, tobits=16)

    assert decoded_arr.shape == arr.shape
    assert np.max(decoded_arr) <= 2**16


def test_encode_decode_circular():
    arr_dim = 2048
    ccd_gain = 1.0 / 4.3  # DN/electron
    ccd_offset = 100  # DN
    ccd_fixedsigma = 17  # DN

    original_arr = (np.random.random([arr_dim, arr_dim]) * (2**16)).astype(int)

    encoded_arr = encode(original_arr, 16, 10)
    decoded_arr = decode_top(encoded_arr, frombits=16, tobits=10, ccd_gain=ccd_gain, ccd_offset=ccd_offset, ccd_fixedsigma=ccd_fixedsigma)

    assert np.all(np.abs(original_arr - decoded_arr) <= 200)  # np.allclose(original_arr, decoded_arr)
