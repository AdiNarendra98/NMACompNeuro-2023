import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import Preprocessing



def plot_target_cursors(alldat, max_len = None):
    
    num_participants = alldat.shape[1]
    for target_cursor in ["target", "cursor"]:
        f, axs = plt.subplots(1,3,figsize = (3*5,5))
        for i in range(num_participants): #Loop over participant
            dat = alldat[0][i] #Select participant data
            axs[0].plot(dat[target_cursor + "X"], dat[target_cursor + "Y"], alpha = 0.5, label = f"Participant {i}") #XY plot
            axs[0].set_xlabel("X")
            axs[0].set_ylabel("Y")
            axs[0].legend(loc = "lower right")
            for k, X_Y in enumerate(["X", "Y"]):
                axs[k+1].plot(dat[target_cursor + X_Y], alpha = 0.5, label = f"Participant {i}")
                axs[k+1].set_ylabel(X_Y)
                axs[k+1].set_xlabel("Timestamp")
                if not max_len is None:
                    axs[k+1].set_xlim(0,max_len)
        plt.show()
        
        
def plot_signals(data,window = None):
    time_window = [0, data.shape[0]] if window is None else window
    num_channels = data.shape[1]
    f, axs = plt.subplots(num_channels,1,figsize = (5,0.8*num_channels))
    for k, channel in enumerate(range(data.shape[1])):
        amp = [np.max(data[time_window[0]:time_window[1], channel]), np.min(data[time_window[0]:time_window[1], channel])]
        axs[k].plot(data[time_window[0]:time_window[1],channel], color = "k", linewidth = 0.2)
        axs[k].set_yticks([])
        axs[k].set_title(f"ch {k}, amplitude = {amp[1], amp[0]}", fontsize = 6)
        if k < num_channels -1:
            axs[k].set_xticks([])

    plt.tight_layout()
    plt.show()


def plot_freq_ranges(data_fft, indices, window):
    window_ind = np.where((indices > window[0]) & (indices < window[1]))[0]
    plt.plot(indices[window_ind], data_fft[window_ind])
    plt.show()
    
def plot_cumsum_results(input_data, max_freq = 30, num_windows = 60):
    
    
    preproc_fun = Preprocessing()
    frequency_windows = list(zip(np.linspace(0,max_freq,num_windows), np.linspace(max_freq/num_windows,max_freq + max_freq/num_windows,num_windows)))
    cumsum_results = np.zeros((len(frequency_windows),input_data.shape[1]))
    for k, frequency_window in enumerate(frequency_windows):
        preprocessed_data, freq_indices = preproc_fun.sel_freq_band(input_data, frequency_window)
        cumsum_results[k,:] = np.sum(np.abs(preprocessed_data), axis = 0)

    max_results = np.max(cumsum_results, axis = 0)
    max_results_ind = np.argmax(cumsum_results, axis = 0)
    plt.scatter(max_results_ind, max_results)
    plt.plot(cumsum_results, alpha = 0.2)
    plt.xticks(np.linspace(0,num_windows,10, dtype = np.int32), np.linspace(0,max_freq, 10, dtype = np.int32))
    plt.xlabel("Frequency")
    plt.ylabel("Cumulative amplitude")
    os.makedirs("../plots", exist_ok = True)
    plt.savefig("../plots/spectral_cumulative_amplitudes.png")
    plt.show()
            