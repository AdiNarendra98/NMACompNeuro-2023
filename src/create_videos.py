import os
from matplotlib import pyplot as plt
import numpy as np
import sys
import moviepy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_data


#Load data
alldat = load_data()
dat = alldat[0,1]
target_cursor = "cursor"
track_size = 5
os.makedirs("../frames", exist_ok=True)
for k,timepoint in enumerate(range(track_size,dat[f"{target_cursor}X"].shape[0],30)):
    plt.scatter(dat[f"{target_cursor}X"][timepoint-track_size:timepoint], dat[f"{target_cursor}Y"][timepoint-track_size:timepoint], color = "black")
    plt.ylim(0,35000)
    plt.xlim(0,35000)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"../frames/frame_{k}.png")
    plt.close()
