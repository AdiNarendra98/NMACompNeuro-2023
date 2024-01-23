import os
import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt

class Preprocessing():
    '''Set of preprocessing functions, All input data should be 2D numpy arrays'''
    
    def check_input(self, input_data):
        '''Check that input data has 2 dimensions'''
        assert len(input_data.shape) == 2       
    
    def band_pass(self, input_data, lp_freq = 200, hp_freq = 0.15, fs = 1000, order = 5):
        '''Apply band pass filter to the data
        inputs:
            input_data : 2D numpy array (Timepoints x Channels),
            lp_freq : low_pass cut-off frequency (float),
            hp_freq : high_pass cut-off frequency (float),
            
        outputs:
            filtered_data : band-passed filtered data, 2D numpy array (Timepoints x Channels)'''
        
        # Check input dimensionality
        self.check_input(input_data)
        
        # Apply band pass filter
        butter = signal.butter(order,[hp_freq, lp_freq],"bp", fs = fs, output = "sos")
        filtered_data= signal.sosfilt(butter, input_data)
        return filtered_data
    
    def hilbert(self, input_data, fs = 1000, get_all = False):
        '''Apply hilbert function to the data, returns envelopp and instantaneous frequencies'''
        envelop = signal.hilbert(input_data, axis = 0)
        amplitude_envelope = np.abs(envelop)
        instantaneous_phase = np.unwrap(np.angle(envelop))
        instantaneous_frequency = (np.diff(instantaneous_phase, axis = 0) /
                           (2.0*np.pi) * fs)
        
        if get_all:
            return envelop, amplitude_envelope, instantaneous_phase, instantaneous_frequency
        else:
            return envelop
    
    def car(self, input_data):
        '''Computes common average reference'''
        return (input_data.T - np.mean(input_data, axis = 1)).T
    
    def sel_freq_band(self, input_data, frequency_windows):
        '''Select a specific frequency band in the data using hilbert functions.'''
        car_data = self.car(input_data)
        hilbert_envelop, amplitude_envelop, _, instantaneous_frequency = self.hilbert(car_data, get_all = True)
        indices = []
        hilbert_envelop_list = []
        for frequency_window in frequency_windows:
            indices.append(np.where((instantaneous_frequency > frequency_window[0]) & (instantaneous_frequency < frequency_window[1]))[0])
            hilbert_envelop_list.append(amplitude_envelop[indices[-1]])
            
        return hilbert_envelop_list, indices
    
    def spectogram(self, input_data, fs = 1000, save_plot = False):
        '''Compute spectral decomposition of the data with visualization feature.'''
        f_list =[]
        t_list = []
        if save_plot:
            f, axs = plt.subplots(input_data.shape[1],1, figsize = (8,12))
        for i in range(input_data.shape[1]):
            f, t, Sxx = signal.spectrogram(input_data[:,i], fs = 1000, return_onesided=False)
            f_list.append(f)
            t_list.append(t)
            if save_plot:
                axs[i].pcolormesh(t, fftshift(f), fftshift(Sxx, axes=0))
                axs[i].set_ylim(-50,50)
                axs[i].set_yticks([])
                axs[i].set_xticks([])
        
        if save_plot:
            os.makedirs("../plots", exist_ok=True)
            plt.savefig("../plots/spectogram.png")
            plt.close()
            
        f_list = np.array(f_list).T
        t_list = np.array(t_list).T
        
        return(f_list, t_list)