import numpy as np

int_type = np.int32

def decode_top(data, frombits, tobits, ccd_gain, ccd_offset, ccd_fixedsigma):
    table = gen_decode_table(frombits, tobits, ccd_gain, ccd_offset, ccd_fixedsigma)
    np.save("my_table.npy", table)
    return decode_bytable(data, table)

def encode(source,frombits,tobits):
    source = np.round(source).astype(int_type).clip(0, None)
    ibits = tobits*2                                            # Intermediate step is multiplication to exactly 2x target bit value
    factor = np.array(2**(ibits-frombits))                      # force a 1-element numpy array
    s2 = np.round(source * factor).astype(int_type)             # source*factor is normally an int anyhow but force integer arithmetic
    return np.floor(np.sqrt(s2)).astype(int_type)               # force nearest-integer square root

def decode(source,frombits,tobits):
    source = np.round(source).astype(int_type).clip(0, None)     # Force integer arithmetic and nonnegative values
    ibits = tobits*2                                            # Calculate factor as in encode above.
    factor = 2.0**(ibits-frombits)
    s2 = source*source                                          # Square the rounded square root
    return np.round(s2/factor).astype(int_type)                 # nearest-integer division of the square

def noise_pdf(val, gain, offset, fixedsigma, n_sigma=5, n_steps=10000):
    electrons = np.clip( (val-offset)/gain, 1, None )           # Use camera calibration to get an e- count
    poisson_sigma = np.sqrt(electrons) * gain                   # Shot noise, converted back to DN
    sigma = np.sqrt( poisson_sigma**2 + fixedsigma**2 )         # Total sigma is quadrature sum of fixed & shot
    step=sigma*n_sigma*2/n_steps                                # Step through a range in DN value -n_sigma to n_sigma in n_steps
    dn_steps = np.arange(-n_sigma*sigma, n_sigma*sigma, step)   # Explicitly enumerate the step values
    normal = np.exp(- dn_steps*dn_steps /sigma/sigma /2 )       # Explicitly calculate the Gaussian/normal PDF at each step
    normal = normal / np.sum(normal)                            # Easier to normalize numerically than to account for missing tails
    return ( val+dn_steps, normal )

def mean_b_offset(sval, frombits, tobits, ccd_gain, ccd_offset, ccd_fixedsigma):
    val = decode(sval, frombits, tobits)                                     # Find the "naive" decoded value
    (vals, weights) = noise_pdf( val, ccd_gain, ccd_offset, ccd_fixedsigma ) # Generate a distribution around that naive value
    weights = weights * (vals >= ccd_offset)                                 # Ignore values below the offset -- which break the noise model
    if(np.sum(weights) < 0.95):                                              # At or below the ccd offset, just return no delta at all.
        return 0
    weights = weights / np.sum(weights)
    svals = encode(vals, frombits, tobits)                                   # Encode the entire value distribution
    dcvals = decode(svals, frombits, tobits)                                 # Decode the entire value distribution to find the net offset
    ev = np.sum(dcvals*weights)                                              # Expected value of the entire distribution
    return ev-val                                                            # Return ΔB.

def decode_corrected(sval, frombits, tobits, ccd_gain, ccd_offset, ccd_fixedsigma):
    s1p = decode(sval+1,frombits,tobits)
    s1n = decode(sval-1,frombits,tobits)
    width = (s1p-s1n)/4
    fixed_sigma = np.sqrt(ccd_fixedsigma**2 + width**2)
    of = mean_b_offset(sval, frombits, tobits, ccd_gain, ccd_offset, fixed_sigma)
    return decode(sval,frombits,tobits)-of

def gen_decode_table(frombits, tobits, ccd_gain, ccd_offset, ccd_fixedsigma):
    output = np.zeros(2**tobits)
    for i in range(0, 2**tobits):
        output[i] = decode_corrected(i, frombits, tobits, ccd_gain, ccd_offset, ccd_fixedsigma)
    return output

def decode_bytable(s,table):
    s = np.round(s).astype(int_type).clip(0, table.shape[0])
    return table[s]

