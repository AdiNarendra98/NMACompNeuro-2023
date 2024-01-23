import numpy as np
import os, requests
from math import sqrt
from scipy.stats import pointbiserialr
import pickle
import argparse
import pandas as pd


def parse_arguments():
    '''Parse arguments for run_full_analysis.py '''
    parser = argparse.ArgumentParser(description="Run full analysis on the data.")

    parser.add_argument(
        "--feature",
        type=str,
        default = "cursorX",
        help="Name of the feature to predict."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=50,
        help="Range of window sizes (min and max)."
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=None,
        help="Number of points in window, if None num_points = window_size"
    )
    parser.add_argument(
        "--brodmann_area_number",
        type=int,
        default = 4,
        help="Number of Brodmann area."
    )
    parser.add_argument(
        "--frequency_band",
        type=float,
        nargs=2,
        default=(8,25),
        metavar=("min_frequency", "max_frequency"),
        help="Range of frequency to use (min and max)."
    )
    parser.add_argument(
        "--participant_ind",
        type=int,
        default=1,
        help="Participant index."
    )
    parser.add_argument(
        "--forc_dist",
        type=int,
        default=0,
        help="Forcasting distance in frame"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for data split"
    )

    return parser.parse_args()

def save_scores(train_score, test_score, args, datetime, filename = "search_results.csv"):
    '''Save and track scores gotten from different experiments'''
    data_dict = {
        "Feature": [args.feature],
        "Window_Size": [args.window_size],
        "Num_points": [args.num_points],
        "Brodmann_Area_Number": [args.brodmann_area_number],
        "Min_Frequency": [args.frequency_band[0]],
        "Max_Frequency": [args.frequency_band[1]],
        "Participant_Index": [args.participant_ind],
        "Forcasting_Distance": [args.forc_dist],
    }

    df = pd.DataFrame(data_dict, index = [datetime])
    df["train_score"] = round(train_score, 5)
    df["test_score"] = round(test_score, 5)
    
    if os.path.exists(filename):
        df_load = pd.read_csv(filename, index_col=0)
        df_new = pd.concat([df_load, df], axis = 0)
        
    else:
        df_new = df
        
    df_new.to_csv(filename)
    

def load_data(path_to_data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")):
    '''Load data from existing file in path_to_data or download from link'''
    fname = os.path.join(path_to_data, 'joystick_track.npz')
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    url = "https://osf.io/6jncm/download"

    if not os.path.isfile(fname):
        try:
            r = requests.get(url)
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
        else:
            if r.status_code != requests.codes.ok:
                print("!!! Failed to download data !!!")
            else:
                with open(fname, "wb") as fid:
                    fid.write(r.content)
                
    alldat = np.load(fname, allow_pickle=True)['dat']
    return alldat

def get_radius_angles(x_coords, y_coords):
    '''Get radius and angles from x and y coordinates by fitting a parametric model'''
    angle = np.arctan2(y_coords - np.mean(y_coords), x_coords - np.mean(x_coords)) * 180 / np.pi
    radius = np.linalg.norm([x_coords - np.mean(x_coords), y_coords - np.mean(y_coords)], axis = 0)
    return radius, angle

def get_label(feature, dat, max_offset = 0):
    '''Get label data to regress from selected feature, you can choose in 
    ["cursorX", "cursorY", "targetX", "targetY", "cursor_angle", "cursor_radius", "target_angle", "target_radius"]
    inputs:
        feature : feature to regress in list above
        dat: data loaded from one participant
        forc_dist: forcasting distance
    returns
        label: data of the feature to regress against'''
    
    # Select the feature    
    if feature in ["cursorX", "cursorY", "targetX", "targetY"]:
        feature_dat = dat[feature]
    elif feature in ["cursor_angle", "cursor_radius", "target_angle", "target_radius"]:
        data_type, ang_or_rad = feature.split("_")        
        radius_angle = get_radius_angles(dat[data_type + "X"], dat[data_type + "Y"])
        feature_dat = radius_angle[["radius", "angle"].index(ang_or_rad)]
    
    # Slide the feature with forcasting distance 
    return feature_dat[max_offset:]
    
def getMerit(subset, label):
    k = len(subset)

    # average feature-class correlation
    rcf_all = []
    for feature in subset.T:
        coeff = pointbiserialr(label[:,0], feature )
        rcf_all.append(abs( coeff.correlation))
    rcf = np.mean( rcf_all )
    
    # average feature-feature correlation
    # corr = df[subset].corr()
    corr = np.corrcoef(subset.T)
    corr[np.tril_indices_from(corr)] = np.nan
    rff = np.nanmean(np.abs(corr))
    return (k * rcf) / sqrt(k + k * (k-1) * rff)

def getBestFeature(subset, label):
    best_value = -1
    best_feature = ''
    for i, feature in enumerate(subset.T):
        coeff = pointbiserialr( label[:,0], feature )
        abs_coeff = abs(coeff.correlation)
        if abs_coeff > best_value:
            best_value = abs_coeff
            best_feature = i

    return best_feature, best_value


def save_pickle(obj, filename):
    '''Save in pickle file'''
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    '''load from pickle file'''
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data 

def load_results(path_to_results = "search_results.csv"):
    df = pd.read_csv(path_to_results)
    return df

def load_all_results(path_to_res_dir = "results"):
    '''Load all results in a given folder'''
    all_dfs = pd.DataFrame()
    for filename in [os.path.join(path_to_res_dir, elem) for elem in os.listdir(path_to_res_dir)]:
        if filename.endswith(".csv"):
            df = load_results(filename)
            all_dfs = pd.concat([all_dfs, df], ignore_index=True, axis = 0)
    return all_dfs


class PriorityQueue:
    '''Class for CFS channel selection'''
    def  __init__(self):
        self.queue = []

    def isEmpty(self):
        return len(self.queue) == 0
    
    def push(self, item, priority):
        """
        item already in priority queue with smaller priority:
        -> update its priority
        item already in priority queue with higher priority:
        -> do nothing
        if item not in priority queue:
        -> push it
        """
        for index, (i, p) in enumerate(self.queue):
            if (set(i) == set(item)):
                if (p >= priority):
                    break
                del self.queue[index]
                self.queue.append( (item, priority) )
                break
        else:
            self.queue.append( (item, priority) )
        
    def pop(self):
        # return item with highest priority and remove it from queue
        max_idx = 0
        for index, (i, p) in enumerate(self.queue):
            if (self.queue[max_idx][1] < p):
                max_idx = index
        (item, priority) = self.queue[max_idx]
        del self.queue[max_idx]
        return (item, priority)







