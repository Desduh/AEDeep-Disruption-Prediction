# pmodel_generation.py
import numpy as np
import pandas as pd

# XP-model adapted from Meneveau & Sreenevasan, 1987 & Malara et al., 2016
def pmodel(noValues=1024, p=0.3, slope=[]):
    noOrders = int(np.ceil(np.log2(noValues)))
    noValuesGenerated = 2**noOrders

    dx = np.array([1.0])
    for _ in range(noOrders):
        dx = next_step_1d(dx, p)

    if (slope):
        # This part is complex and not used in the main logic, keeping it for completeness
        fourierCoeff = fractal_spectrum_1d(noValues, slope/2)
        meanVal = np.mean(dx)
        stdy = np.std(dx)
        x = np.fft.ifft(dx - meanVal)
        phase = np.angle(x)
        x = fourierCoeff*np.exp(1j*phase)
        x = np.fft.fft(x).real
        x *= stdy/np.std(x)
        x += meanVal
    else:
        x = dx

    return x[0:noValues]

def next_step_1d(dx, p):
    y2 = np.zeros(dx.size*2)
    sign = np.random.rand(1, dx.size) - 0.5
    sign /= np.abs(sign)
    y2[0:2*dx.size:2] = dx + sign*(1-2*p)*dx
    y2[1:2*dx.size+1:2] = dx - sign*(1-2*p)*dx
    return y2

def fractal_spectrum_1d(noValues, slope):
    ori_vector_size = noValues
    ori_half_size = ori_vector_size//2
    a = np.zeros(ori_vector_size)
    for t2 in range(ori_half_size):
        index = t2
        t4 = 1 + ori_vector_size - t2
        if (t4 >= ori_vector_size):
            t4 = t2
        coeff = (index + 1)**slope
        a[t2] = coeff
        a[t4] = coeff
    a[1] = 0
    return a

def generate_and_prepare_series(length, p_value, peak_threshold, norm_percentile=99.0):
    """
    Generates a p-model series and prepares it for training/evaluation.
    
    Returns:
        pd.DataFrame with 'raw', 'normalized', and 'peak' columns.
    """
    # Generate raw series
    raw_series = pmodel(length, p_value) + 0.01

    # Normalize the series. Using a high percentile for the max value
    # makes the normalization more robust to single extreme outliers.
    max_val = np.percentile(raw_series, norm_percentile)
    normalized_series = raw_series / (max_val + 1e-6)
    
    # Create DataFrame
    df = pd.DataFrame({
        'raw': raw_series,
        'normalized': normalized_series
    })

    # Label peaks based on the normalized series and threshold
    df['peak'] = (df['normalized'] > peak_threshold).astype(int)
    
    return df

def create_sequences(df, window_size, lookahead):
    """
    Creates sequences and labels for the LSTM model.
    A label is 1 if a peak occurs within the lookahead period.
    """
    X, y = [], []
    signal = df['normalized'].values
    labels = df['peak'].values

    for i in range(len(signal) - window_size - lookahead + 1):
        X.append(signal[i : i + window_size])
        # The label is 1 if any peak exists in the future window
        y.append(int(np.any(labels[i + window_size : i + window_size + lookahead])))

    return np.array(X)[..., np.newaxis], np.array(y)