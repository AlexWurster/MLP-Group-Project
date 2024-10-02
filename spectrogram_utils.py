import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


########################################################################################
## 4 values to check: nperseg, noverlap, fix_colorbar_range, ref_power
########################################################################################


# check default setting for ref_power
def convert_pwr_to_db(power_spec, ref_power=None):
    if ref_power is None:
        # ref_power = np.max(power_spec)
        ref_power = np.mean(power_spec)
        print(f"ref_power = {ref_power}") # ref_power = 23356.943359375
    with np.errstate(divide='ignore'):  # Ignore log(0) warnings
        spec_db = 10 * np.log10(power_spec / ref_power)

    return spec_db

# for given spectrogram parquet files only
def plot_spectrogram_parquet(partitioned_dfs, convert_to_db=False, figsize=(15, 9), fix_colorbar_range=True, cmap='jet'):

    # for subplot layout
    n = len(partitioned_dfs) # partitioned into 4 spectrogram images. 
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # in case of multiple rows and cols

    for i, (chain_name, df) in enumerate(partitioned_dfs.items()):
        # convert df to numpy array and transpose
        spec_vals = df.to_numpy().T
        print(f"spec_vals.shape = {spec_vals.shape}")
        # spec_vals.shape = (100, 312)
        # spec_vals.shape = (100, 312)
        # spec_vals.shape = (100, 312)
        # spec_vals.shape = (100, 312)

        # time and frequency axes
        t_ = np.linspace(0, 100, num=spec_vals.shape[1])
        f_ = np.linspace(0, 100, num=spec_vals.shape[0])

        if convert_to_db:
            spec_vals = convert_pwr_to_db(spec_vals)

        # Plot
        ax = axes[i]
        # shading: or 'flat', 'nearest', 'auto'
        cax = ax.pcolormesh(t_, f_, spec_vals, vmin=-20, vmax=20, shading='gouraud', cmap=cmap) if fix_colorbar_range else ax.pcolormesh(t_, f_, spec_vals, shading='gouraud', cmap=cmap)
        fig.colorbar(cax, ax=ax)
        ax.set_title(chain_name)
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Time')

    # adjust layout
    plt.tight_layout()
    plt.show()


# nperseg and noverlap values are non-trivial. might have to check. 
def get_spectrogram_from_eeg(eeg, srate=200, nperseg=None, noverlap=None, ref_power=None, width=None):

    if not nperseg:
        nperseg = 256 # int(0.5 * srate)  # window length as 0.5 seconds, scipy default is 256
    
    # Compute the power spectral density
    f, t, Sxx = signal.spectrogram(eeg, fs=srate, window='hann', nperseg=nperseg, noverlap=noverlap, axis=-1)

    return f, t, Sxx
    
# def plot_spectrogram_from_eeg(f, t, spectrogram, electrodes, eeg_file_id, convert_to_db=True, figsize=(10, 6), fix_colorbar_range=True, cmap='jet'):

#     '''
#     spectrogram: np array of dim (freq, time) -> we don't have to transpose
#     '''

#     plt.figure(figsize=figsize)

#     # t_ = np.linspace(0, 100, num=spectrogram.shape[1])
#     # f_ = np.linspace(0, 100, num=spectrogram.shape[0])

#     if convert_to_db:
#         spec_vals = convert_pwr_to_db(spectrogram)

#     # Plot
#     # ax = axes[i]
#     # shading: or 'flat', 'nearest', 'auto'
#     plt.pcolormesh(t, f, spec_vals, vmin=-20, vmax=20, shading='gouraud', cmap=cmap) if fix_colorbar_range else plt.pcolormesh(t, f, spec_vals, shading='gouraud', cmap=cmap)
    
#     bar_label = 'Magnitude (dB)' if convert_to_db else 'Linear Scale'
    
#     plt.colorbar(label=bar_label)
#     plt.title(f"Spectrogram ({electrodes}) from /train_eegs/{eeg_file_id}.parquet")
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')

#     # adjust layout
#     # plt.tight_layout()
#     plt.show()


def plot_spectrogram_from_eeg(f, t, spectrogram, electrodes, eeg_file_id, convert_to_db=True, figsize=(10, 6), fix_colorbar_range=True, cmap='jet', time_regions=None, show_plot=True, save_path=None):
    '''
    Plot a spectrogram with vertical bars at specified time regions.
    
    Parameters:
    - f: Frequency bins array
    - t: Time bins array
    - spectrogram: Spectrogram data array of dimensions (freq, time)
    - electrodes: Name or list of electrodes
    - eeg_file_id: Identifier for the EEG file
    - convert_to_db: Whether to convert spectrogram to dB scale
    - figsize: Tuple specifying figure size
    - fix_colorbar_range: Whether to fix the colorbar range
    - cmap: Colormap for the spectrogram
    - time_regions: List of start times to draw vertical bars (approximate, function will find the closest matching time in 't')
    '''
    
    plt.figure(figsize=figsize)

    if convert_to_db:
        # Assuming convert_pwr_to_db is defined elsewhere
        spec_vals = convert_pwr_to_db(spectrogram)
    else:
        spec_vals = spectrogram

    # Plot the spectrogram
    plt.pcolormesh(t, f, spec_vals, vmin=-20, vmax=20, shading='gouraud', cmap=cmap) if fix_colorbar_range else plt.pcolormesh(t, f, spec_vals, shading='gouraud', cmap=cmap)
    
    # Setup colorbar and labels
    bar_label = 'Magnitude (dB)' if convert_to_db else 'Linear Scale'
    plt.colorbar(label=bar_label)
    plt.title(f"Spectrogram ({electrodes}) from /train_eegs/{eeg_file_id}.parquet")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    # Find the closest values in 't' to the specified 'time_regions' and draw vertical bars
    if time_regions:
        for time_region in time_regions:
            closest_time = t[np.argmin(np.abs(t - time_region))]
            plt.axvline(x=closest_time, color='red', linestyle='--', linewidth=4)

    if show_plot:
        plt.show()

    # Save the plot as an SVG image if save_path is provided
    # Note that save_path is a filename not a directory !!
    if save_path:
        plt.savefig(save_path, format='svg')
        print(f"Spectrogram generated from EEG saved as {save_path} !!")
    

