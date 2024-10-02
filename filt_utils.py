import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from math import ceil#, abs
import pandas as pd
from scipy.signal import butter, sosfilt, firls, filtfilt, lfilter

def plot_filter_freq_response(coeffs, freqs, nyquist, ideal, filt_type='firls'):
    """
    Plot the frequency response of a FIR filter.

    Parameters:
    coeffs: Filter coefficients (from firls or butter).
    freqs: Frequency points.
    nyquist: Nyquist frequency.
    ideal: Ideal frequency response.
    filt_type: Type of filter ('firls' or 'butter').

    Only 'firls' is currently implemented.
    """
    plt.figure()  # Optional: Add figsize=(10,7) for custom figure size

    if filt_type == 'firls':
        # Plot ideal frequency response
        plt.plot(freqs * nyquist, ideal, 'r', label='Ideal Response')

        # Compute and normalize FFT of filter kernel
        fft_filtkern = abs(fft(coeffs))
        fft_filtkern /= np.max(fft_filtkern)  # Normalize to 1 for visualization

        # Calculate normalized amplitude and frequency range
        normalized_amplitude = fft_filtkern[:int(ceil(len(fft_filtkern) / 2))]
        hz_filtkern = np.linspace(0, nyquist, len(normalized_amplitude))

        # Plot actual frequency response
        plt.plot(hz_filtkern, normalized_amplitude, 'b', label='Actual Response')

        # Labeling the plot
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Filter Frequency Response')
        plt.legend()

        plt.show()

    else:
        # Placeholder for other filter types
        print("Only 'firls' filter type is currently supported.")


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Create a Butterworth bandpass filter.
    
    Parameters:
    lowcut: Lower cutoff frequency.
    highcut: Higher cutoff frequency.
    fs: Sampling frequency.
    order: Filter order. Default is 5.

    Returns:
    Second-order sections representation of the filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def apply_butter_bandpass(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to the data.
    
    Parameters:
    data: Input data.
    lowcut: Lower cutoff frequency.
    highcut: Higher cutoff frequency.
    fs: Sampling frequency.
    order: Filter order.

    Returns:
    Filtered data.
    """
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data, axis=0)
    return y

def apply_firls_bandpass(data, lowcut, highcut, srate, trans_width=0.10, num_taps=251, plot_freq_response=True):
    """
    Create and apply an FIR least squares bandpass filter to the data.
    
    Parameters:
    data: Input data.
    lowcut: Lower cutoff frequency.
    highcut: Higher cutoff frequency.
    srate: Sampling rate.
    trans_width: Transition width.
    num_taps: Number of filter taps.
    plot_freq_response: Whether to plot the frequency response.

    Returns:
    Filtered data.
    """
    nyquist = srate / 2
    freqs = np.array([0, (1 - trans_width) * lowcut, lowcut, highcut, (1 + trans_width) * highcut, nyquist]) / nyquist
    ideal = np.array([0, 0, 1, 1, 0, 0])
    filterweights = firls(num_taps, freqs, ideal)
    filtered_data = filtfilt(filterweights, 1, data, axis=0)

    if plot_freq_response:
        plot_filter_freq_response(filterweights, freqs, nyquist, ideal, filt_type='firls')

    return filtered_data

def apply_bandpass_filter(data_df, lowcut, highcut, srate, filt_type='butter'):
    """
    Apply a specified bandpass filter to a DataFrame and return the filtered data.

    Parameters:
    data_df: Input data in a DataFrame.
    lowcut: Lower cutoff frequency.
    highcut: Higher cutoff frequency.
    srate: Sampling rate.
    filt_type: Type of filter ('butter' or 'firls').

    Returns:
    DataFrame with filtered data.
    """
    # Convert to numpy and handle NaNs
    data = data_df.apply(lambda x: np.nan_to_num(x, nan=np.nanmean(x))).values

    # Apply the selected filter
    if filt_type == 'butter':
        res = apply_butter_bandpass(data, lowcut, highcut, srate)
    elif filt_type == 'firls':
        res = apply_firls_bandpass(data, lowcut, highcut, srate)
    else:
        raise ValueError("Invalid filter type. Choose 'butter' or 'firls'.")

    # Convert result back to DataFrame with original column names
    filtered_df = pd.DataFrame(res, columns=data_df.columns)
    return filtered_df

########################################################################################################################
# low-pass and high-pass (probs will only use lowpass for our purposes)
########################################################################################################################

# change to use sos response. 
def butter_filter(data, cutoff_freq=20, sampling_rate=200, order=4, filter_type='low'):
    """
    Apply a Butterworth filter (lowpass or highpass) to the data.

    Parameters:
    data: Input data.
    cutoff_freq: Cutoff frequency.
    sampling_rate: Sampling rate of the data.
    order: Order of the filter.
    filter_type: Type of filter ('low' for lowpass, 'high' for highpass).

    Returns:
    Filtered data.
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

# low_filtered_eeg_df = apply_filter(eeg_df, cutoff_freq=5, srate=200, filter_type='low')
def apply_filter(data_df, cutoff_freq=20, srate=200, filter_type='low'):
    """
    Apply a Butterworth lowpass or highpass filter to a DataFrame.

    Parameters:
    data_df: DataFrame containing the data to be filtered.
    cutoff_freq: Cutoff frequency for the filter.
    srate: Sampling rate of the data.
    filter_type: Type of filter ('low' for lowpass, 'high' for highpass).

    Returns:
    DataFrame containing the filtered data.
    """
    # Convert DataFrame to numpy array and handle NaNs
    data = data_df.apply(lambda x: np.nan_to_num(x, nan=np.nanmean(x))).values

    # Apply the filter
    res = butter_filter(data, cutoff_freq=cutoff_freq, sampling_rate=srate, order=4, filter_type=filter_type)

    # Convert result back to DataFrame with original column names
    filtered_df = pd.DataFrame(res, columns=data_df.columns)
    return filtered_df
