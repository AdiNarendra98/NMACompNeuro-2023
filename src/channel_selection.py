import os
import numpy as np
from sklearn.decomposition import PCA
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import getMerit, getBestFeature, PriorityQueue

class ChannelSelection():
    '''Set of channel selection functions, All input data should be 2D numpy arrays'''
    
    def check_input(self, input_data):
        '''Check that input data has 2 dimensions'''
        assert len(input_data.shape) == 2
    
    def specific_channel(self, input_data, channel_sel):
        '''Select specific channel and return a 2D array
        inputs:
            input_data : 2D numpy array (Timepoints x Channels),
            channel_sel : channel index to select (int)
        return:
            2D numpy array (Timepoints x Selected channels)'''
            
        self.check_input(input_data)
        return(np.array([input_data[:,channel_sel]]).T)    
    
    def channel_pca(self, input_data, num_components = 16, get_explained_variance = False):
        '''Run PCA over channels
        inputs:
            input_data : 2D numpy array (Timepoints x Channels),
            num_components: number of principal components
        return:
            2D numpy array (Timepoints x Selected channels)'''
            
        self.check_input(input_data)
        pca_model = PCA(n_components=num_components)
        pca_V = pca_model.fit_transform(input_data)
        
        if get_explained_variance:
            return pca_V, np.sum(pca_model.explained_variance_ratio_)
        else:
            return pca_V
    def select_by_Brodmann(self, input_data, sel_brodmann, brodmann_indices):
        '''Select all channels in a specific Brodmann area'''
        
        channel_per_brodmann = input_data[:, np.array(brodmann_indices) == f"Brodmann area {sel_brodmann}"]
        
        
        if channel_per_brodmann.shape[1]:
            if len(channel_per_brodmann.shape) < 2:
                channel_per_brodmann = channel_per_brodmann[:, np.newaxis]
            return channel_per_brodmann
        else:
            raise ValueError("Brodmann area not referenced")
        
    def get_rectangles(kernel_size, array_size):
        pass
    
    def CFS(self, input_data, label, max_backtrack = 5):
        '''Run Correlation-based feature selection (based on https://johfischer.com/2021/08/06/correlation-based-feature-selection-in-python-from-scratch/)
        inputs:
            input_data : 2D numpy array (Timepoints x Channels),
            label: class to run the correlation against
            max_backtrack: maximum number of trial for the while loop
        return:
            2D numpy array (Timepoints x Selected channels)'''
        
        best_feature_ind, best_value = getBestFeature(input_data, label)
    
        # initialize queue
        queue = PriorityQueue()

        # push first tuple (subset, merit)
        queue.push([best_feature_ind], best_value)
        
        # list for visited nodes
        visited = []

        # counter for backtracks
        n_backtrack = 0
                
        # repeat until queue is empty
        # or the maximum number of backtracks is reached
        while not queue.isEmpty():
            # get element of queue with highest merit
            subset, priority = queue.pop()
            
            # check whether the priority of this subset
            # is higher than the current best subset
            if (priority < best_value):
                n_backtrack += 1
            else:
                best_value = priority
                best_subset = subset

            # goal condition
            if (n_backtrack == max_backtrack):
                break
            
            # iterate through all features and look of one can
            # increase the merit
            for feature_ind in range(input_data.shape[1]):
                temp_subset = subset + [feature_ind]
                
                # check if this subset has already been evaluated
                for node in visited:
                    if (set(node) == set(temp_subset)):
                        break
                # if not, ...
                else:
                    # mark it as visited
                    visited.append(temp_subset)
                    # compute merit
                    merit = getMerit(input_data[:, temp_subset], label)
                    # and push it to the queue
                    queue.push(temp_subset, merit)
                    
        return(input_data[:, best_subset])