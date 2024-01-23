'''
Analysis of regression of ECoG Data to cursor position
'''

#@title Data retrieval
from matplotlib import rcParams
from matplotlib.cm import get_cmap
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from utils.utils import load_data, load_results
from utils.plot_utils import *
import pandas as pd 
from pandas import DataFrame as df
import itertools
from tqdm import tqdm


rcParams['figure.figsize'] = [20, 4]
rcParams['font.size'] = 15
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True

class DataAnalysis():
    def avg_anly(self, all_data):
        
        features = ['Feature', 'Window_Size', 'Num_points','Brodmann_Area_Number', 'Min_Frequency', 'Max_Frequency','Participant_Index', 'Forcasting_Distance']
        all_data = all_data[~np.isnan(all_data["Random_Seed"])]
        new_df = pd.DataFrame(columns = all_data.columns,)
        ref_df = all_data[features].drop_duplicates()
        for ref_index in tqdm(ref_df.index):
            group_score = []
            seed_temp_list = []
            for index in all_data.index:
                if all_data[features].loc[index].equals(ref_df.loc[ref_index]):
                    seed = all_data.loc[index]["Random_Seed"]
                    if not seed in seed_temp_list:
                        seed_temp_list.append(seed)
                        group_score.append(all_data.loc[index]["test_score"])
            
            if len(group_score):
                mean_group = np.nanmean(group_score)
                std_group = np.nanstd(group_score)
                new_line = all_data.loc[ref_index].copy()
                new_line["test_score"] = mean_group
                new_line["test_score_std"] = std_group
                new_line["num_trials"] = len(group_score)
                new_df = pd.concat([new_df, new_line], ignore_index = True, axis = 1)
        nan_data = new_df.T["Min_Frequency"].to_numpy().astype(np.float16)
        new_df = new_df.T[~np.isnan(nan_data)]
        return new_df
    
    def get_brain_score(self, df, dat):
        brodmann_list = df["Brodmann_Area_Number"].to_list()
        dat_brodmann = dat["Brodmann_Area"]
        brain_score = np.zeros(len(dat_brodmann))
        
        for i in range(len(df.index)): 
            brain_score[dat_brodmann.index(f"Brodmann area {int(brodmann_list[i])}")] = df.iloc[i]["test_score"]
        
        return brain_score

            
    def sel_features(self, data, features:list, feature_name):
        res = data.loc[data[feature_name].isin(features)]
        return res
    
    def sel_feature(self, data, feature, feature_name):
        res = data[data[feature_name] == feature]
        return res
    
    def set_to_zero(self, data):
        new_df = data.copy()
        indices = new_df["test_score"] < 0
        new_df.loc[new_df["test_score"] < 0, "test_score"] = 0
        return new_df
    
    def get_positives(self, data, threshold = 0.):
        return data[data["test_score"] > threshold]
    
    def plot_feature_vs_score(self, data, feature, save = None):
        ranges = np.sort(np.unique(data[feature].to_list()))
        score = []
        std = []
        for feature_val in ranges:
            data_feature = data[data[feature] == feature_val]
            score.append(data_feature["test_score"].mean())
            if "test_score_std" in data.columns:
                std.append(data_feature["test_score_std"].mean())

        if "test_score_std" in data.columns:
            plt.errorbar(ranges, score, std, marker='o', capsize = 2)
        else:
            plt.plot(ranges, score, marker = "o")
            
        plt.xlabel(" ".join(feature.split("_")))
        plt.ylabel("R2 score")
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
            plt.close()
            
    def plot_multi_features_vs_score(self, data, features:list, save = None, cm = "Spectral"):
        '''Plot feature1 against multidimensional feature2'''
        assert len(features) <= 2
        feature, second_feature = features
        ranges = np.sort(np.unique(data[feature].to_list()))
        second_ranges = np.sort(np.unique(data[second_feature].to_list()))
        score = []
        std = []
        for feature_val in ranges:
            data_feature = data[data[feature] == feature_val]
            temp = []
            temp_std = []
            for second_feature_val in second_ranges:
                data_feature_second = data_feature[data_feature[second_feature] == second_feature_val]
                temp.append(data_feature_second["test_score"].mean())
                if "test_score_std" in data.columns:
                    temp_std.append(data_feature_second["test_score_std"].mean())
            score.append(temp)
            std.append(temp_std)
        cmap = get_cmap(cm)
        colors = [cmap(i) for i in np.linspace(0,1,len(second_ranges))]
        f = plt.figure(figsize = (10,6))
        print(ranges)
        ranges = np.array(ranges)/1000
        if "test_score_std" in data.columns:
            for i in range(len(second_ranges)):
                plt.errorbar(ranges, np.array(score)[:,i], np.array(std)[:,i], marker='o', capsize = 2, label = second_feature + f" {int(second_ranges[i])}", color= colors[i])
        else:
            plt.plot(ranges, score, marker = "o")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(" ".join(feature.split("_")) + " (s)")
        plt.ylabel("R2 score")
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
            plt.close()
        
    def brodmann_anly(self, avg_results, brodmann_channels, pind, savePath=None):
    
    #     - best trail per channel 
    #     - score against number of electrodes per area 

    
        #seperate out positive test values for participant pind and add channels
        pos_results = avg_results[(avg_results['test_score'] > 0) & (avg_results['Participant_Index'] == pind )]
        # for each brodmann area find top scores window size
        brodm_results = pos_results[['Participant_Index', 'test_score', 'Window_Size', 'Brodmann_Area_Number']]
        # add number of electrodes per broman area --> map area to number electrodes
        idx_ba = f'Channel_Num_P_{pind}'
        brodmann_channels = brodmann_channels[['Brodmann_Area_Number', idx_ba ]]
        # left join to map number of channels for each brodmann area
        brodm_anly = brodm_results.merge(brodmann_channels, on='Brodmann_Area_Number', how='left')
        
        # Get average score per brodmann area
        # group data by brodmann area number and find index of max test score for each group
        idx_max_test_score = brodm_anly.groupby('Brodmann_Area_Number')['test_score'].idxmax()
        # get rows with max test score and corresponding window size 
        best_scores_data = brodm_anly.loc[idx_max_test_score, ['Brodmann_Area_Number', idx_ba, 'test_score', ]]

        # TODO: - plot: x = brodmann area, y= average score, number channels is how big marker size it (idx_ba is how you index channel numbers)
        plt.scatter(best_scores_data['Brodmann_Area_Number'], best_scores_data['test_score'], s=best_scores_data[idx_ba])
        plt.xlabel('Brodmann Area Number')
        plt.ylabel('Best Score')
        plt.title(f'Best Test Score per Brodmann Area: Participant {pind}')
        
    #     - add parameter to save (add path to save) or show images (save=none)
        # TODO: should i standardize the naming convention? With pind (participant index) and source of all data?
        if savePath is None:
            plt.show()
        else:
            plt.savefig(f'{savePath}/BA_vs_Score_P{pind}.png')

    def score_window_plot(self, avg_data, savePath=None):
         # - func for each window size plot against score 
    #     can change the data frame you pass into the function 
    #     do for each channel/bromann area/ freq band in diff colors
    
    # - for each forcasting window plot against score (reuse same function as above)
        pass
    
    

if __name__ == "__main__":
    # load the main data
    # load regression results
    path_to_results = "/Users/taimacrean/Git/Neuromatch_Academy_Afrovenator/data/search_brodmann.csv"
    all_results = load_results(path_to_results)

    #average results across each kernel
    avg_results = avg_anly(all_results)

    #load channel per bromann area df
    path_to_channels = "/Users/taimacrean/Git/Neuromatch_Academy_Afrovenator/data/num_electrodes.csv"
    brodmann_channels = load_results(path_to_channels)
    brodmann_channels.rename(columns={
    'Unnamed: 0': 'Brodmann_Area_Number',
    '0': 'Channel_Num_P_0',
    '1': 'Channel_Num_P_1',
    '2': 'Channel_Num_P_2',
    '3': 'Channel_Num_P_3'
    }, inplace=True)

    # analyze all participants in the file
    # TODO: change all results to avg_results once the function is done 
    for pind in np.unique(all_results['Participant_Index']):
        pass
    #     call the functions you want from the class
    #     - func take full data, average over same data across random seeds to get mean value and standard deviation (set negative values to zero first)
    #           (look at replicate data in the drive)
    #           this will be the data that gets passed in to every other function
    # 
    #     - brodmann analysis func 
    # - func for each window size plot against score 
    # - for each forcasting window plot against score (reuse same function as above)

   
    