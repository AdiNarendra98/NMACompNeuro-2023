import os
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm

class RegressionModels():
    
    def temporal_split(self, X,y, test_size = 0.2, seed = 42):
        '''Perform a train test split by taking the first (1-test_size) as training set and the rest as testing set
        inputs:
            X: Neural data (Timepoints, channels),
            y: feature to regress (Timepoints, 1),
            test_size: portion of the testing set'''
        np.random.seed(seed)
        test_start_index = np.random.randint(0,int((1-test_size)*len(X)))
        test_stop_index = test_start_index + int((1-test_size)*len(X))
        indices = np.arange(len(X))
        train_indices = np.append(indices[:test_start_index], indices[test_stop_index:])
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_start_index:test_stop_index]
        y_test = y[test_start_index:test_stop_index]
        
        return(X_train, X_test, y_train, y_test)
    

    def create_design_matrix(self, data, window_size = 10, num_points = None, forc_dist = 0, max_offset = None):
        if num_points is None:
            num_points = window_size        
        
        num_timepoints = data.shape[0]
        num_channels = data.shape[1]
        assert(window_size < num_timepoints)
        offset = window_size + forc_dist
        max_offset = offset if max_offset is None else max_offset
        windows = []
        for i in range(max_offset - offset, num_timepoints - offset):
            indices = np.linspace(i, i + window_size, num_points).astype(np.int32) #Get indices of the window
            windows.append(data[indices])
        X = np.array(windows).reshape((data.shape[0] - max_offset, num_points*num_channels))
        return X
    
    def linear_regression(self, X_train, X_test, y_train,y_test, return_all = False):
        '''Train a linear regression model on the train set and return predictions of the test set
        inputs:
            X_train : Neural data to train on (Timepoints, channels),
            X_test: Neural data to test on (Timepoints, channels),
            y_train: feature to regress for the training (Timepoints, 1),
            y_test: feature to regress for the testing (Timepoints, 1),
            return_all : Set to return train/test scores and weights'''
            
        reg = LinearRegression().fit(X_train,y_train)
        train_score = reg.score(X_train,y_train)
        coeffs = reg.coef_
        intercept = reg.intercept_

        y_pred = reg.predict(X_test)
        test_score = reg.score(X_test, y_test)
        
        if return_all:
            return y_pred, train_score, test_score, coeffs, intercept
        else: 
            return y_pred
        
    def ridge_regression(self, X_train, X_test, y_train,y_test, alpha, return_all = False):
        '''Train a ridge regression model on the train set and return predictions of the test set
        inputs:
            X_train : Neural data to train on (Timepoints, channels),
            X_test: Neural data to test on (Timepoints, channels),
            y_train: feature to regress for the training (Timepoints, 1),
            y_test: feature to regress for the testing (Timepoints, 1),
            alpha: alpha parameter of the Ridge regression,
            return_all : Set to return train/test scores and weights'''
            
        reg = Ridge().fit(X_train,y_train)
        train_score = reg.score(X_train,y_train)
        coeffs = reg.coef_
        intercept = reg.intercept_

        y_pred = reg.predict(X_test)
        test_score = reg.score(X_test, y_test)
        
        if return_all:
            return y_pred, train_score, test_score, coeffs, intercept
        else: 
            return y_pred
        