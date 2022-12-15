import numpy as np

# TODO: move decode into punchbowl and leave encode here

# TODO: move to be parameter for necessary functions instead of global
int_type = np.int32


def decode(data, from_bits, to_bits, ccd_gain, ccd_bias, ccd_read_noise):
    """

    Parameters
    ----------
    data
    from_bits
    to_bits
    ccd_gain
    ccd_bias
    ccd_read_noise

    Returns
    -------

    """
    table = gen_decode_table(from_bits, to_bits, ccd_gain, ccd_bias, ccd_read_noise)
    return decode_by_table(data, table)


def encode(data, from_bits, to_bits):
    """
    
    Parameters
    ----------
    data
    from_bits
    to_bits

    Returns
    -------

    """
    data = np.round(data).astype(int_type).clip(0, None)
    factor = np.array(2 ** (2*to_bits - from_bits))   
    data_scaled_by_factor = np.round(data * factor).astype(int_type) 
    return np.floor(np.sqrt(data_scaled_by_factor)).astype(int_type)


def decode_simple(data, from_bits, to_bits):
    """Performs a simple decoding using the naive squaring strategy

    Parameters
    ----------
    data
    from_bits
    to_bits

    Returns
    -------

    """
    data = np.round(data).astype(int_type).clip(0, None) 
    factor = 2.0**(2*to_bits - from_bits)
    return np.round(np.square(data)/factor).astype(int_type)


def noise_pdf(val, gain, offset, read_noise_level, n_sigma=5, n_steps=10000):
    electrons = np.clip((val-offset)/gain, 1, None)              # Use camera calibration to get an e- count
    poisson_sigma = np.sqrt(electrons) * gain                    # Shot noise, converted back to DN
    sigma = np.sqrt(poisson_sigma ** 2 + read_noise_level ** 2)  # Total sigma is quadrature sum of fixed & shot
    dn_steps = np.arange(-n_sigma*sigma, n_sigma*sigma, sigma*n_sigma*2/n_steps) 
    normal = np.exp(-dn_steps*dn_steps/sigma/sigma/2)  # Explicitly calculate the Gaussian/normal PDF at each step
    normal = normal / np.sum(normal) # Easier to normalize numerically than to account for missing tails
    return val+dn_steps, normal


def mean_b_offset(sval, from_bits, to_bits, ccd_gain, ccd_offset, ccd_read_noise):
    naive_decoded_value = decode_simple(sval, from_bits, to_bits)

    # Generate distribution around naive value
    (vals, weights) = noise_pdf(naive_decoded_value, ccd_gain, ccd_offset, ccd_read_noise)

    # Ignore values below the offset -- which break the noise model
    weights = weights * (vals >= ccd_offset)
    if np.sum(weights) < 0.95:
        return 0

    weights = weights / np.sum(weights)
    svals = encode(vals, from_bits, to_bits)  # Encode the entire value distribution
    dcvals = decode_simple(svals, from_bits, to_bits)  # Decode the entire value distribution to find the net offset
    ev = np.sum(dcvals*weights)  # Expected value of the entire distribution
    return ev-naive_decoded_value  # Return ΔB.


def decode_corrected(sval, from_bits, to_bits, ccd_gain, ccd_offset, ccd_read_noise):
    s1p = decode_simple(sval + 1, from_bits, to_bits)
    s1n = decode_simple(sval - 1, from_bits, to_bits)
    width = (s1p-s1n)/4
    fixed_sigma = np.sqrt(ccd_read_noise ** 2 + width ** 2)
    of = mean_b_offset(sval, from_bits, to_bits, ccd_gain, ccd_offset, fixed_sigma)
    return decode_simple(sval, from_bits, to_bits)-of


def gen_decode_table(from_bits, to_bits, ccd_gain, ccd_offset, ccd_read_noise):
    output = np.zeros(2**to_bits)
    for i in range(0, 2**to_bits):
        output[i] = decode_corrected(i, from_bits, to_bits, ccd_gain, ccd_offset, ccd_read_noise)
    return output


def decode_by_table(s, table):
    s = np.round(s).astype(int_type).clip(0, table.shape[0])
    return table[s]

