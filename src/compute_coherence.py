import os
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_data, load_pickle, save_pickle
from utils.plot_utils import plot_signals
from data_preprocessing import Preprocessing
from channel_selection import ChannelSelection
from regression_models import RegressionModels
from cosine_tuning  import CosineTuning
from mne_connectivity import spectral_connectivity_time
import pickle
#Import classes
preproc_fun = Preprocessing()
channel_sel = ChannelSelection()
regression_models = RegressionModels()
cosine_tuning = CosineTuning()


#Load data
alldat = load_data()

#Select one participant
dat = alldat[0,1]
freq_ranges = [[8,12], [18,24], [35,42], [42,70], [70,100], [100,140]]


for k,freq_range in enumerate(freq_ranges):
    
    # if os.path.exists(f"con_{k}.p"):
    #     con = load_pickle(f"con_{k}.p")
        
    # else:
    con = spectral_connectivity_time([dat["V"].T], freq_range, sfreq = 1000)
    save_pickle(con, f"con_{k}.p")

    con_data = con.get_data()
    
    for i in range(2):
        con_reshaped = con_data[0,:,i].reshape((dat["V"].shape[1],dat["V"].shape[1]))
        plt.imshow(con_reshaped)
        plt.title(str(freq_range))
        plt.xlabel("channel number")
        plt.ylabel("channel number")
        plt.colorbar()
        plt.savefig(f"coherence_{k}_{i}.png")
        plt.close()
