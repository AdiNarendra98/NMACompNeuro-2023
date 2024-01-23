import os
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_data
from data_preprocessing import Preprocessing
from channel_selection import ChannelSelection
from regression_models import RegressionModels
from cosine_tuning  import CosineTuning
import mne


#Import classes
preproc_fun = Preprocessing()
channel_sel = ChannelSelection()
regression_models = RegressionModels()
cosine_tuning = CosineTuning()


#Load data
alldat = load_data()

#Select one participant
dat = alldat[0,1]
locs = {f"{i}": dat["locs"][i] for i in range(dat["locs"].shape[0])}
montage = mne.channels.make_dig_montage(locs)
print(montage.get_positions()["coord_frame"])

