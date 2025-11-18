# Analysis of Whale Vocalizations in a Distributed Acoustic Sensing (DAS) Dataset

## Project Overview

This project provides an in-depth analysis of a specific segment from a Distributed Acoustic Sensing (DAS) dataset, focusing on the characterization of whale vocalizations. The work is based on the methodologies presented in the paper **"Eavesdropping at the Speed of Light: Distributed Acoustic Sensing of Baleen Whales in the Arctic"** by Bouffaut et al. (2022).

The primary goal is to process and visualize the complex, multi-channel DAS data to identify and analyze acoustic events of interest. The Python script provided in this repository loads a 160-second raw data segment, applies filtering and signal processing techniques, and generates visualizations that replicate the analytical approach of the source paper.

The dataset was recorded by a 120 km fiber optic cable array in Svalbard, Norway. This analysis focuses on a specific segment spanning **35.7 km to 40.8 km**.

## Dataset

*   **Source:** Bouffaut, L., & Taweesintananon, K. (2022). Data from "Eavesdropping at the speed of light: Distributed acoustic sensing of baleen whales in the Arctic."
*   **File:** `20200627_052441_ch08751_to_ch10000_whale_raw_L160s.mat`
*   **Segment Length:** 160 seconds
*   **Spatial Coverage:** 35.7 km to 40.8 km of the DAS array
*   **Sampling Frequency:** 645.16 Hz
*   **Spatial Resolution:** Channels spaced approximately every 4.08 meters

## Methodology

The analysis is performed in a step-by-step manner, following the logical flow presented in the accompanying presentation. The Python script implements the following key steps:

1.  **Data Loading:** The `.mat` file containing the raw strain data, time vector, and distance information is loaded.
2.  **Exploratory Time Series Analysis:** A band-pass filter is applied to the raw data to isolate frequencies of interest (e.g., 40-60 Hz). Time-series plots for selected channels are generated to visualize the filtered strain and identify potential events.
3.  **Spatio-Temporal (t-x) Analysis:** The data is processed using a median conditioning filter to reduce noise and highlight coherent signals across multiple channels over time. This helps visualize how acoustic energy propagates along the fiber optic cable.
4.  **Spatio-Spectral (f-x) Analysis:** A Short-Time Fourier Transform (STFT) is applied to the data to compute its spectral content. By plotting the maximum energy for each frequency across all channels, we can identify which locations along the cable are receiving specific frequency bands.
5.  **Detailed Spectrogram Analysis:** A spectrogram is generated for a specific channel (at approximately 40.6 km) where a significant acoustic event was identified. This provides a detailed time-frequency representation of the whale vocalization.

## How to Run the Code

### Prerequisites

*   Python 3.x
*   NumPy
*   SciPy
*   Matplotlib

You can install the required libraries using pip:
```bash
pip install numpy scipy matplotlib
```

### Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/Whale-Vocalization-DAS-Analysis.git
    cd Whale-Vocalization-DAS-Analysis
    ```
2.  **Place the dataset:**
    Download the dataset (`20200627_052441_ch08751_to_ch10000_whale_raw_L160s.mat`) and place it in a directory (e.g., `/data/`). **Important:** Make sure to update the `mat_file_path` variable in the Python scripts to point to the correct location of your `.mat` file.

3.  **Run the analysis scripts:**
    Execute the Python scripts from your terminal:
    ```bash
    python analysis.py
    ```
    This will generate and display the plots corresponding to the different stages of the analysis.

## Expected Results

The scripts will produce a series of visualizations that correspond to the figures in the presentation:

*   **Exploratory Time Series Plot:** Shows the filtered strain over time for three selected channels, highlighting the presence of a significant event around the 60-80 second mark.
*   **Spatio-Temporal (t-x) Plot:** A 2D heatmap showing the filtered strain across both distance (y-axis) and time (x-axis), with vertical lines indicating acoustic events.
*   **Spatio-Spectral (f-x) Plot:** A 2D heatmap showing processed strain across distance (y-axis) and frequency (x-axis), revealing a concentration of energy around 40.6 km.
*   **Spectrogram:** A detailed time-frequency plot for the channel at 40.6 km, clearly showing the spectral characteristics of the identified event.

These combined plots provide a multi-modal view that connects the timing, location, and frequency content of the whale vocalizations.

## Acknowledgements

*   This analysis is based on the work of **Dr. Victor Espinosa**, **Dr. Hefeng Dong**, and the original paper's author, **Lea Bouffaut**.
*   The project was completed as part of the **Master Waves** program at **Universitat Politècnica de València (UPV Gandia)**.
*   **Key Reference:** Bouffaut, L., Taweesintananon, K., et al. (2022). *Eavesdropping at the speed of light: Distributed acoustic sensing of baleen whales in the Arctic*. Frontiers in Marine Science, 9. [https://doi.org/10.3389/fmars.2022.901348](https://doi.org/10.3389/fmars.2022.901348)
