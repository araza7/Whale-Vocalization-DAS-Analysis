import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, medfilt2d, stft
import sys

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    Applies a zero-phase Butterworth band-pass filter to the data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    # Apply filter along the time axis (axis=1)
    return signal.filtfilt(b, a, data, axis=1)


if __name__ == '__main__':
    # --- 1. Load Data ---
    # load the MAT file.
    mat_file_path = '/kaggle/input/dataset/20200627_052441_ch08751_to_ch10000_whale_raw_L160s.mat'
    mat = scipy.io.loadmat(mat_file_path)


    # --- 2. Extract and Prepare Variables ---
    print("Extracting variables from the MAT file structure...")
    
    # data: The main (channels, samples) array
    data_raw = mat['data']
    
    # x2_time_s: Has shape (1, 103217). .flatten() converts it to a 1D array.
    time_vector = mat['x2_time_s'].flatten()
    
    # x1_position_m: Has shape (1, 1250). .flatten() converts it to a 1D array.
    distance_vector_m = mat['x1_position_m'].flatten()
    
    # info_sampling_frequency_Hz: Has shape (1,1). .item() extracts the single value.
    fs = float(mat['info_sampling_frequency_Hz'].item())
    
    print(f"Data loaded. Shape: {data_raw.shape}, Fs: {fs:.2f} Hz, Duration: {time_vector[-1]:.2f} s")

    # --- 3. Select Channels to Plot ---
    # We will select three channels to compare for clear visualization.
    num_channels_in_slice = data_raw.shape[0]
    channels_to_plot_indices = [
        int(num_channels_in_slice * 0.1),  # Channel at 10%
        int(num_channels_in_slice * 0.5),  # Middle channel
        int(1188),  # Channel at 90%
    ]
    print(f"\nSelected channel indices for plotting: {channels_to_plot_indices}")

    # --- 4. Filter the Data ---
    # Raw DAS data is noisy. We filter to see the whale calls (~15-18 Hz) clearly.
    print("Applying band-pass filter [40-60 Hz] to isolate signals...")
    filtered_data = bandpass_filter(data_raw, fs, 40, 60.0)
    
    # --- 5. Create the Plot with Vertical Offsets ---
    fig, ax = plt.subplots(figsize=(16, 8))

    # Calculate an automatic offset based on the data's variance for nice spacing
    # This prevents the plots from overlapping.
    offset_scale = np.std(filtered_data) * 10
    
    # Loop through the selected channels and plot each one with an offset
    for i, channel_index in enumerate(channels_to_plot_indices):
        # The vertical offset for this trace
        vertical_offset = i * offset_scale
        
        # Get the filtered strain data for this specific channel
        strain_timeseries = filtered_data[channel_index, :] + vertical_offset
        
        # Get the distance of this channel for the legend
        distance_km = distance_vector_m[channel_index] / 1000.0
        
        # Plot the strain data vs. time
        label = f'Channel Index {channel_index} ({distance_km:.2f} km)'
        ax.plot(time_vector, strain_timeseries, label=label)

    # --- 6. Finalize the Plot ---
    ax.set_title('Filtered Strain vs. Time for Selected Channels (with Vertical Offset)', fontsize=16)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Filtered Strain Amplitude (Offset for Clarity)', fontsize=12)
    
    ax.set_yticks([]) 
    
    ax.legend(title='Channel Location', fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(time_vector[0], time_vector[-1])
    
    plt.tight_layout()
    plt.show()

