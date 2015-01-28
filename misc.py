# -*- coding: utf-8 -*-
"""

@author: toni
"""


import pandas as pd
import numpy as np

from scipy.sparse import lil_matrix, csr_matrix
from numpy.linalg import norm
from pylab import hist

from time import time

from scipy.sparse.linalg import lsqr

import sqlite3 as sqlite3





class Sim():
    
    @staticmethod
    def cos(x, y):
        return min(max(x.dot(y) / norm(x) / norm(y), -1), 1)
        
    @staticmethod
    def cos2(x, y):
        return min(max(np.dot(x, y) / norm(x) / norm(y), -1), 1)
    
    @staticmethod
    def pearson(x, y):
        return sum( (x - x.mean()) * (y - y.mean()) ) / np.sqrt( sum( (x - x.mean())**2 )) / np.sqrt( sum( (y - y.mean())**2 ))

    @staticmethod
    def rbf(x, y, sigma=1):
        return min(max(np.exp(-(norm(x-y)**2 / 2 / sigma**2)), -1), 1)


class Metric():
    
    @staticmethod
    def MSE(y, yhat):
        return np.sqrt(((y - yhat) ** 2).mean())
    
    @staticmethod
    def MAE(y, yhat):
        return np.sqrt((abs(y - yhat)).mean())



def splitData(x = None, q = [60, 20, 20]):
    """
    Split data on variable x to training, validation and testing sets.
    """
    
    q = np.cumsum(q)
    
    train = np.array(x <= np.percentile(x, q[0]))
    valid = np.array((x > np.percentile(x, q[0])) & (x <= np.percentile(x, q[1])))

    ret = (train, valid)
    
    if (len(q) == 3):
        test = np.array((x > np.percentile(x, q[1])) & (x <= np.percentile(x, q[2])))
        
        ret = (train, valid, test)
    
    return ret
    
    
            
def splitData2(df):
    """
    Split data on variable x to training and validation sets.
    
    **** Need to add support for test set splits.
    
    """
    
    timestamp = pd.DataFrame(df["timestamp"].groupby(df["user id"]).agg(np.percentile, 80), columns = ["timestamp"])
    timestamp["user id"] = np.array(timestamp.index)
    timestamp.reset_index(drop = True)

    train = np.zeros(df.shape[0], dtype = bool)
    valid = np.zeros(df.shape[0], dtype = bool)

    for us in df["user id"].unique():
        t = timestamp.ix[timestamp["user id"] == us, "timestamp"]
        train = train | np.array((df["user id"] == us) & (df["timestamp"] < int(t)))
        valid = valid | np.array((df["user id"] == us) & (df["timestamp"] >= int(t)))

    return train, valid

