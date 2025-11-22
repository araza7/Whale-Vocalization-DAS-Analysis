import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.signal as signal
from scipy.signal import butter, filtfilt, stft, medfilt2d
import matplotlib.gridspec as gridspec


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
    # processing as per the paper's methods.
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

    # --- Panel B: f-x Spatio-Spectral Plot (Method 4.3) ---
    # processing plot that reveals the true signals.
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

    # --- Representative Spectrogram at the specified Location ---
   
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

    #plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main():
    """Main function to run the definitive"""
    mat_file_path = '/kaggle/input/dataset/20200627_052441_ch08751_to_ch10000_whale_raw_L160s.mat'
    das = DASData(mat_file_path)
    
   
    Plot_figures(das)
    
 
#multiple channel data with 4.08m spacing between each channel from 35.75 km to 40.85 km
if __name__ == '__main__':
    main()
