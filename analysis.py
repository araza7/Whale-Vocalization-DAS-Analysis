import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.signal as signal
from scipy.signal import butter, filtfilt, stft, medfilt2d
import matplotlib.gridspec as gridspec


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
    # Using the structure you provided to load the MAT file.
    try:
        mat_file_path = '/kaggle/input/dataset/20200627_052441_ch08751_to_ch10000_whale_raw_L160s.mat'
        mat = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"ERROR: The file '{mat_file_path}' was not found.")
        exit()

    # --- 2. Extract and Prepare Variables Based on Your File Structure ---
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
    
    # Since we added an offset, the y-axis ticks are not meaningful as absolute strain values.
    # We can remove them for clarity.
    ax.set_yticks([]) 
    
    ax.legend(title='Channel Location', fontsize=10, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(time_vector[0], time_vector[-1])
    
    plt.tight_layout()
    plt.show()


class DASData:
    """A class to load and hold DAS data and metadata from the .mat file."""
    def __init__(self, mat_file_path):
        print(f"Loading data from {mat_file_path}...")
        try:
            mat = loadmat(mat_file_path)
            self.strain = mat['data'].astype(np.float64)
            self.fs = float(mat['info_sampling_frequency_Hz'][0, 0])
            self.dt = float(mat['info_sample_interval_s'][0, 0])
            self.channel_dist_m = mat['x1_position_m'][0]
            self.time_s = mat['x2_time_s'][0]
            self.dist_km = self.channel_dist_m / 1000
            self.n_channels, self.n_samples = self.strain.shape
            print(f"Data loaded successfully. Range: {self.dist_km[0]:.2f} - {self.dist_km[-1]:.2f} km.")
        except (FileNotFoundError, KeyError) as e:
            print(f"ERROR: Could not load or parse the .mat file: {e}")
            exit()

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """Applies a Butterworth bandpass filter along the time axis."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

# --- REPLICATION FUNCTIONS ---

def Plot_figures(das):
    
    fig = plt.figure(figsize=(15, 10))
    # This GridSpec correctly mimics the layout of Figure 4 in the paper.
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[2.5, 1, 0.08, 0.08], height_ratios=[1, 1], wspace=0.4)
    axA = fig.add_subplot(gs[:, 0])      # Left large plot
    axB = fig.add_subplot(gs[0, 1])      # Top-right plot
    axC = fig.add_subplot(gs[1, 1])      # Bottom-right plot
    
    # Colorbars (rightmost column)
    cbar_axA = fig.add_subplot(gs[0, 2])
    cbar_axB = fig.add_subplot(gs[1, 2])
    cbar_axC = fig.add_subplot(gs[1, 3])
    
    fig.suptitle("DAS Analysis at 40.6km for 60s to 80s", fontsize=16)

    # --- Panel A: t-x Spatio-Temporal Plot (Method §4.2) ---
    # This processing is correct as per the paper's methods.
    filtered_data = bandpass_filter(das.strain, das.fs, 30, 43)
    conditioned_data = np.abs(medfilt2d(filtered_data, kernel_size=3))
    row_med, col_med = np.median(conditioned_data, axis=1, keepdims=True), np.median(conditioned_data, axis=0, keepdims=True)
    conditioned_data = conditioned_data - row_med - col_med + np.median(conditioned_data)
    vmin, vmax = np.percentile(conditioned_data, [5, 99])
    imA = axA.imshow(conditioned_data, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
               extent=[das.time_s[0], das.time_s[-1], das.dist_km[0], das.dist_km[-1]])
    fig.colorbar(imA, cax=cbar_axA, label='Filtered Strain (a.u.)')
    axA.set_title("A: Spatio-Temporal (t-x) ")
    axA.set_xlabel("Time (s)"), axA.set_ylabel("Distance (km)")

    # --- Panel B: f-x Spatio-Spectral Plot (Method §4.3) ---
    # This processing is also correct, and it's this plot that reveals the true signals.
    data_win = bandpass_filter(das.strain, das.fs, 5, 110)
    f, _, Sxx = stft(data_win, das.fs, nperseg=256, noverlap=128, axis=1)
    fx_matrix = np.max(np.abs(Sxx), axis=2) - np.mean(np.abs(Sxx), axis=2)
    vmin, vmax = np.percentile(fx_matrix, [5, 99.5])
    imB =  axB.imshow(fx_matrix, aspect='auto', origin='lower', cmap='viridis', vmin=vmax*0.1, vmax=vmax,
               extent=[f[0], f[-1], das.dist_km[0], das.dist_km[-1]])
    imB = fig.colorbar(imB, ax=axB, label='Processed Strain (dB)')
    axB.set_title("B: Spatio-Spectral (f-x) ")
    axB.set_xlabel("Frequency (Hz)"), axB.set_ylabel("Distance (km)")
    axB.set_xlim(10, 100)

    # --- Panel C: Representative Spectrogram at the CORRECT Location ---
   
    # We select the location of the brightest yellow feature, which is ~40.6 km.
    prominent_signal_km = 40.52
    
    ch_idx = np.argmin(np.abs(das.dist_km - prominent_signal_km))
    # Method §4.1: Average 2 channels for noise reduction.
    signal = np.mean(das.strain[ch_idx-1:ch_idx+1, :], axis=0)
    
    f_spec, t_spec, Sxx_spec = stft(signal, das.fs, nperseg=512, noverlap=256)
    db_power = 10 * np.log10(np.abs(Sxx_spec)**2 + 1e-12)
    
    pcm = axC.pcolormesh(t_spec, f_spec, db_power, shading='gouraud', cmap='viridis',
                   vmin=np.percentile(db_power, 5), vmax=np.percentile(db_power, 95))
    fig.colorbar(pcm, cax=cbar_axC, label='dB power')
    
    # Update title to reflect the correct location.
    axC.set_title(f"C: Spectrogram at {das.dist_km[ch_idx]:.2f} km ")
    axC.set_xlabel("Time (s)"), axC.set_ylabel("Frequency (Hz)")
    axC.set_ylim(0, 100)
    
    # Add an annotation line connecting panel B and C
    axB.axhline(y=das.dist_km[ch_idx], color='r', linestyle='--')
    axB.text(100, das.dist_km[ch_idx], '-> To Panel C', color='r', ha='right', va='bottom', weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    """Main function to run the definitive, corrected replication script."""
    mat_file_path = '/kaggle/input/dataset/20200627_052441_ch08751_to_ch10000_whale_raw_L160s.mat'
    das = DASData(mat_file_path)
    
   
    Plot_figures(das)
    
 
#multiple channel data with 4.08m spacing between each channel from 35.75 km to 40.85 km
if __name__ == '__main__':
    main()
