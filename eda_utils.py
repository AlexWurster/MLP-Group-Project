import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_eeg_stacks(df, eeg_file_id, eeg_df_label, mode='bdb_montage', vertical_shift=180, additional_shift=300, figsize_x=100):
    """
    Plots EEG data stacks from a pandas df.

    """
    # Set figure size based on figsize_x and a fixed aspect ratio
    plt.figure(figsize=(figsize_x, figsize_x // 3))

    # Generating time vector for x-axis
    time_vector = np.arange(0, len(df))

    # Initialize cumulative vertical shift
    cumulative_shift = 0

    # Reverse the column order for plotting
    columns = df.columns[::-1].to_list()
    print(f"Columns: {columns}")

    # Determine grouping based on mode
    grouping = 4 if mode == 'bdb_montage' else 5

    # Plot each EEG data series
    for idx, column in enumerate(columns):
        # Adjust each series by its minimum value and apply cumulative shift
        adjusted_series = df[column] - df[column].min() + cumulative_shift
        plt.plot(time_vector, adjusted_series, color='black')

        # Annotate the start of each EEG line
        start_point = adjusted_series.iloc[0]
        plt.annotate(column, xy=(0, start_point), xytext=(5, 0), 
                     textcoords="offset points", fontsize=figsize_x // 2)

        # Update cumulative shift
        cumulative_shift += vertical_shift

        # Apply additional shift after each group
        if (idx + 1) % grouping == 0:
            cumulative_shift += additional_shift

    # Adding axis labels and title
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(f'EEG Time Series - ID: {eeg_file_id}, Label: {eeg_df_label}', fontsize=70)

    # Display the plot
    plt.show()


def handle_nans(df):
    """
    Replace NaN values in the DataFrame with the mean of their respective columns.
    If a column is entirely NaNs, all its values are set to 0.

    """
    df_filled = df.copy()  # Create a copy to avoid modifying the original DataFrame
    column_means = df.mean(skipna=True, numeric_only=True)  # Mean of each column, ignoring NaNs
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):  # Apply only to numeric columns

            nan_ratio = df[col].isna().mean()
            print(f"NaN ratio for column '{col}': {nan_ratio}")

            if nan_ratio < 1:  # Check if the column is not entirely NaNs
                df_filled[col] = df[col].fillna(column_means[col])
            else:
                df_filled[col] = 0  # If all values are NaNs, set entire column to 0

    return df_filled


def apply_referencing(eeg_df, mode='bdb_montage'):
    '''
    Apply referencing to an EEG dataframe based on the specified mode.

    Parameters:
    eeg_df (pandas.DataFrame): EEG data.
    mode (str): Referencing mode - 'bdb_montage', 'chain_avg', or 'global_avg'.

    Returns:
    pandas.DataFrame: Referenced EEG data.
    '''

    electrodes = {
        'LL': ['Fp1', 'F7', 'T3', 'T5', 'O1'],
        'LP': ['Fp1', 'F3', 'C3', 'P3', 'O1'],
        'RP': ['Fp2', 'F8', 'T4', 'T6', 'O2'],
        'RR': ['Fp2', 'F4', 'C4', 'P4', 'O2']
    }

    # Initialize an empty DataFrame to store results
    res_df = pd.DataFrame()

    if mode == 'bdb_montage':
        for chain, electrodes_list in electrodes.items():
            for i in range(len(electrodes_list) - 1):
                diff_col_name = f"{electrodes_list[i]} - {electrodes_list[i+1]}"
                res_df[diff_col_name] = eeg_df[electrodes_list[i]] - eeg_df[electrodes_list[i+1]]

    elif mode == 'chain_avg': # this one doesn't make much sense btw. 
        for chain, electrodes_list in electrodes.items():
            chain_avg = eeg_df[electrodes_list].mean(axis=1)
            for electrode in electrodes_list:
                res_df[f"{electrode} - avg({chain})"] = eeg_df[electrode] - chain_avg

    elif mode == 'global_avg':
        global_avg = eeg_df.select_dtypes(include='number').mean(axis=1) # this should have (n-by-1) dimension. 
        for col in eeg_df.columns:
            res_df[f"{col} - global_avg"] = eeg_df[col] - global_avg

    return res_df

## From Deotte's discussion: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010

# i don't think ive tested this one. 
def get_eeg_spectrogram_pair(train_df, eeg_df, spectrogram_df, row_idx):

    row = train_df.iloc[row_idx]

    eeg_offset = int( row.eeg_label_offset_seconds )
    # 50 seconds from the offset (srate: 200Hz)
    eeg_segment = eeg_df.iloc[eeg_offset*200:(eeg_offset+50)*200]

    spec_offset = int( row.spectrogram_label_offset_seconds )
    # 10 mins (600 secs) from the offset
    spectrogram_segment = spectrogram_df.loc[(spectrogram_df.time>=spec_offset)
                        &(spectrogram_df.time<spec_offset+600)]
    
    return eeg_segment, spectrogram_segment


# GET_ROW = 0
# EEG_PATH = 'train_eegs/'
# SPEC_PATH = 'train_spectrograms/'

# train = pd.read_csv('train.csv')
# row = train.iloc[GET_ROW]

# eeg = pd.read_parquet(f'{EEG_PATH}{row.eeg_id}.parquet')
# eeg_offset = int( row.eeg_label_offset_seconds )
# eeg = eeg.iloc[eeg_offset*200:(eeg_offset+50)*200]

# spectrogram = pd.read_parquet(f'{SPEC_PATH}{row.spectrogram_id}.parquet')
# spec_offset = int( row.spectrogram_label_offset_seconds )
# spectrogram = spectrogram.loc[(spectrogram.time>=spec_offset)
#                      &(spectrogram.time<spec_offset+600)]
