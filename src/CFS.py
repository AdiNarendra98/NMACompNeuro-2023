'''
    Correlation based feature selection following tutorial from: "https://johfischer.com/2021/08/06/correlation-based-feature-selection-in-python-from-scratch/"

    As both features and classes are continuous pearson correlation coefficent is used to comptue merit
        (Should classes be binary, point-biserial correlation coefficient should be used)
'''

import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import getMerit, getBestFeature, load_data


class PriorityQueue:
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




if __name__ ==  "__main__":
    
    alldat = load_data()
    dat = alldat[0,1]
    
    best_feature_ind, best_value = getBestFeature(dat["V"], dat["cursorX"])
    
    # initialize queue
    queue = PriorityQueue()

    # push first tuple (subset, merit)
    queue.push([best_feature_ind], best_value)
    
    # list for visited nodes
    visited = []

    # counter for backtracks
    n_backtrack = 0

    # limit of backtracks
    max_backtrack = 5
            
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
        for feature_ind in range(dat["V"].shape[1]):
            temp_subset = subset + [feature_ind]
            
            # check if this subset has already been evaluated
            for node in visited:
                if (set(node) == set(temp_subset)):
                    break
            # if not, ...
            else:
                # ... mark it as visited
                visited.append(temp_subset)
                # ... compute merit
                merit = getMerit(dat["V"][:, temp_subset], dat["cursorX"])
                # and push it to the queue
                queue.push(temp_subset, merit)

print(best_subset)