import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_data, save_scores, parse_arguments, get_label
from data_preprocessing import Preprocessing
from channel_selection import ChannelSelection
from regression_models import RegressionModels
from cosine_tuning  import CosineTuning
from datetime import datetime

MAX_OFFSET = 50000

#Import classes
preproc_fun = Preprocessing()
channel_sel = ChannelSelection()
regression_models = RegressionModels()
cosine_tuning = CosineTuning()

verbose = False

def run_analysis(args):
    '''Run full analysis using a set of parameters'''

    #Load data
    alldat = load_data()

    feature = args.feature
    window_size = args.window_size
    num_points = args.num_points
    brodmann_area_number = args.brodmann_area_number
    frequency_band = args.frequency_band
    participant_ind = args.participant_ind
    forc_dist = args.forc_dist
    random_seed = args.random_seed

    assert forc_dist < MAX_OFFSET
    
    #Select one participant
    dat = alldat[0,participant_ind]
    # max_timepoint = np.min([alldat[0,i]["V"].shape[0] for i in range(alldat.shape[1])])   #TODO Set all participants to the same data length or fit sinusoidal curves

    label = get_label(feature, dat, MAX_OFFSET)

    #Preprocess the data
    if verbose:
        print("Preprocess the data...")
    preprocessed_data = preproc_fun.car(dat["V"])
    preprocessed_data = preproc_fun.band_pass(preprocessed_data, frequency_band[1],frequency_band[0], order = 5)
    if verbose:
        print("...Data preprocessed")

    # #Select channels
    if verbose:
        print("Select channels...")
    channel_sel_data = channel_sel.select_by_Brodmann(preprocessed_data, brodmann_area_number, dat["Brodmann_Area"])
    if channel_sel_data.shape[1] > 3:
        channel_sel_data, ev = channel_sel.channel_pca(channel_sel_data, num_components = 3, get_explained_variance = True)

    print(channel_sel_data.shape)
    # # #Run regression models
    if verbose:
        print("Run models...")
    X = regression_models.create_design_matrix(channel_sel_data, window_size = window_size, num_points = num_points, forc_dist = forc_dist, max_offset = MAX_OFFSET)
    print(X.shape, label.shape)
    # X_train,X_test, y_train, y_test = regression_models.temporal_split(X, dat[feature], test_size=0.2)
    X_train ,X_test, y_train, y_test = regression_models.temporal_split(X, label, test_size=0.2, seed = random_seed)
    y_pred, train_score, test_score, coeffs, intercept = regression_models.linear_regression(X_train, X_test, y_train, y_test, return_all=True)

    # Save score and save parameters
    save_scores(train_score, test_score, args, datetime.now().strftime("%m/%d/%Y,%H:%M:%S"), filename = "search_results.csv")
    
if __name__ == "__main__":
    
    args = parse_arguments()
    run_analysis(args)





