import numpy as np
from simpunch.encoding import encode, decode


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
    original_arr = (np.random.random([arr_dim, arr_dim]) * (2**16)).astype(int)

    encoded_arr = encode(original_arr, frombits=16, tobits=16)
    decoded_arr = decode(encoded_arr, frombits=16, tobits=16)

    assert np.allclose(original_arr, decoded_arr)
