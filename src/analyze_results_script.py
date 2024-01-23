import os
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_data, load_results, load_all_results
from utils.plot_utils import *
from analysis import DataAnalysis


path_to_results = r"C:\Users\andyb\Documents\EPFL\Research Assistant 2023\NeuroMatch_academy\Neuromatch_Academy_Afrovenator\src\results"
all_results = load_all_results(path_to_results)
data_analysis = DataAnalysis()
path_to_figures = os.path.join(os.path.dirname(path_to_results), "figures")
os.makedirs(path_to_figures, exist_ok = True)


colormap = "plasma"

# Plot brodmann window_exploration
if True:
    feature = "cursorY"
    brodmann_areas = [2,3,6,8,9,22]
    sel_participant = 2
    freq = 8
    df = data_analysis.sel_feature(all_results, sel_participant, "Participant_Index")
    df = data_analysis.sel_feature(df, freq, "Min_Frequency")
    df = data_analysis.sel_feature(df, feature, "Feature")
    df = data_analysis.sel_features(df, brodmann_areas, "Brodmann_Area_Number")
    # df = data_analysis.get_positives(df, threshold = 0.05)
    df = data_analysis.sel_feature(df, 0, "Forcasting_Distance")
    df = data_analysis.set_to_zero(df)
    try:
        df_avg = data_analysis.avg_anly(df)
        print(df_avg["test_score"].max())
        # data_analysis.plot_feature_vs_score(df_avg, feature = "Brodmann_Area_Number", save = os.path.join(path_to_figures, f"brodmann_p_{sel_participant}_{freq}_{window_size}.png"))
        data_analysis.plot_multi_features_vs_score(df_avg, features = ["Window_Size", "Brodmann_Area_Number"], cm = colormap)
        data_analysis.plot_multi_features_vs_score(df_avg, features = ["Window_Size", "Brodmann_Area_Number"], save = os.path.join(path_to_figures, f"window_vs_brodmann_p_{sel_participant}_{feature}_{freq}.png"), cm = colormap)
    except:
        print("Did not work")
        print(df)
    
    
    
if True:
    # Get forecast plots
    sel_participant = 2
    freq = 8
    window_size = 2000
    feature = "cursorY"
    brodmann_areas = [2,3,6,8,9,22]


    df = data_analysis.sel_feature(all_results, sel_participant, "Participant_Index")
    df = data_analysis.sel_feature(df, feature, "Feature")
    df = data_analysis.sel_feature(df, freq, "Min_Frequency")
    df = data_analysis.sel_features(df, brodmann_areas, "Brodmann_Area_Number")
    df = data_analysis.sel_feature(df, window_size, "Window_Size")
    # df = data_analysis.sel_features(df, freqs, "Min_Frequency")
    df = data_analysis.set_to_zero(df)

    try:
        df_avg = data_analysis.avg_anly(df)
        print(df_avg["test_score"].max())
        df_avg.to_csv("treshold_final.csv")
        data_analysis.plot_multi_features_vs_score(df_avg, features = ["Forcasting_Distance", "Brodmann_Area_Number"], cm = colormap)
        data_analysis.plot_multi_features_vs_score(df_avg, features = ["Forcasting_Distance", "Brodmann_Area_Number"], save = os.path.join(path_to_figures, f"forecast_p_{sel_participant}_{freq}_{brodmann_areas}_{window_size}.png"), cm = colormap)
    except:
        print("Did not work :")
        print(df)