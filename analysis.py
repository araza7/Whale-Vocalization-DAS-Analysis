import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, medfilt2d, stft
import sys


class DASData:
    """A class to load and hold DAS data and metadata from the .mat file."""
    def __init__(self, mat_file_path):
        print(f"Loading data from {mat_file_path}...")
        try:
            mat = loadmat(mat_file_path)
            # Load strain data and convert to float64 for precision
            self.strain = mat['data'].astype(np.float64)
            
            # Extract metadata
            self.fs = float(mat['info_sampling_frequency_Hz'][0, 0])
            self.dt = float(mat['info_sample_interval_s'][0, 0])
            
            # Spatial data conversion (meters to km)
            self.channel_dist_m = mat['x1_position_m'][0]
            self.time_s = mat['x2_time_s'][0]
            self.dist_km = self.channel_dist_m / 1000
            
            self.n_channels, self.n_samples = self.strain.shape
            print(f"Data loaded successfully. Range: {self.dist_km[0]:.2f} - {self.dist_km[-1]:.2f} km.")
        except (FileNotFoundError, KeyError) as e:
            print(f"ERROR: Could not load or parse the .mat file: {e}")
            sys.exit()

def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """Applies a Butterworth bandpass filter along the time axis."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

def Plot_figures(das):
    
    fig = plt.figure(figsize=(15, 10))
    
    # GridSpec layout: 
    # Column 0: Main Plot (Left)
    # Column 1: Sub Plots (Right)
    # Column 2: Colorbars for Sub Plots (Right edge)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[2.5, 1, 0.08], height_ratios=[1, 1], wspace=0.2, hspace=0.3)
    
    # main plot axes
    axA = fig.add_subplot(gs[:, 0])      # Left large plot
    axB = fig.add_subplot(gs[0, 1])      # Top-right plot
    axC = fig.add_subplot(gs[1, 1])      # Bottom-right plot
    
    # Explicit axes for right-side colorbars to prevent resizing issues
    cbar_axB = fig.add_subplot(gs[0, 2]) 
    cbar_axC = fig.add_subplot(gs[1, 2]) 
    
    fig.suptitle("DAS Analysis at 40.6km for 60s to 80s", fontsize=16)

    # --- Panel A: t-x Spatio-Temporal Plot ---
    # Filter and condition the data (remove common mode noise)
    filtered_data = bandpass_filter(das.strain, das.fs, 30, 43)
    conditioned_data = np.abs(medfilt2d(filtered_data, kernel_size=3))
    row_med, col_med = np.median(conditioned_data, axis=1, keepdims=True), np.median(conditioned_data, axis=0, keepdims=True)
    conditioned_data = conditioned_data - row_med - col_med + np.median(conditioned_data)
    
    vmin, vmax = np.percentile(conditioned_data, [5, 99])
    
    imA = axA.imshow(conditioned_data, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
               extent=[das.time_s[0], das.time_s[-1], das.dist_km[0], das.dist_km[-1]])
    
    # Attach colorbar directly to axA (letting matplotlib handle sizing for the main plot)
    fig.colorbar(imA, ax=axA, label='Filtered Strain (a.u.)', pad=0.02, aspect=30)
    
    axA.set_title("A: Spatio-Temporal (t-x) ")
    axA.set_xlabel("Time (s)"), axA.set_ylabel("Distance (km)")

    # --- Panel B: f-x Spatio-Spectral Plot ---
    # Compute STFT and aggregate over time
    data_win = bandpass_filter(das.strain, das.fs, 5, 110)
    f, _, Sxx = stft(data_win, das.fs, nperseg=256, noverlap=128, axis=1)
    fx_matrix = np.max(np.abs(Sxx), axis=2) - np.mean(np.abs(Sxx), axis=2)
    
    vmin, vmax_b = np.percentile(fx_matrix, [5, 99.5])
    
    imB = axB.imshow(fx_matrix, aspect='auto', origin='lower', cmap='viridis', vmin=vmax_b*0.1, vmax=vmax_b,
               extent=[f[0], f[-1], das.dist_km[0], das.dist_km[-1]])
    
    # Use the explicit colorbar axis (cbar_axB)
    fig.colorbar(imB, cax=cbar_axB, label='Processed Strain (dB)')
    
    axB.set_title("B: Spatio-Spectral (f-x) ")
    axB.set_xlabel("Frequency (Hz)"), axB.set_ylabel("Distance (km)")
    axB.set_xlim(10, 100)

    # --- Panel C: Spectrogram ---
    # Select channel at specific location and stack neighbors for SNR
    prominent_signal_km = 40.52
    ch_idx = np.argmin(np.abs(das.dist_km - prominent_signal_km))
    signal = np.mean(das.strain[ch_idx-1:ch_idx+1, :], axis=0)
    
    f_spec, t_spec, Sxx_spec = stft(signal, das.fs, nperseg=512, noverlap=256)
    db_power = 10 * np.log10(np.abs(Sxx_spec)**2 + 1e-12)
    
    pcm = axC.pcolormesh(t_spec, f_spec, db_power, shading='gouraud', cmap='viridis',
                   vmin=np.percentile(db_power, 5), vmax=np.percentile(db_power, 95))
    
    # Use the explicit colorbar axis (cbar_axC)
    fig.colorbar(pcm, cax=cbar_axC, label='dB power')
    
    axC.set_title(f"C: Spectrogram at {das.dist_km[ch_idx]:.2f} km ")
    axC.set_xlabel("Time (s)"), axC.set_ylabel("Frequency (Hz)")
    axC.set_ylim(0, 100)
    
    # visual reference connecting B and C
    axB.axhline(y=das.dist_km[ch_idx], color='r', linestyle='--')
    axB.text(100, das.dist_km[ch_idx], '-> To Panel C', color='r', ha='right', va='bottom', weight='bold')

    plt.show()





if __name__ == '__main__':
    main()
    mat_file_path = '/kaggle/input/dataset/20200627_052441_ch08751_to_ch10000_whale_raw_L160s.mat'
    das = DASData(mat_file_path)
    Plot_figures(das)
