# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:29:29 2023

@author: Nakano_Lab
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 14:01:36 2023

@author: Nakano_Lab
"""

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from   matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
import matplotlib
import pandas as pd

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
plt.rcParams['mathtext.default'] = 'regular'

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(22)

font_legend = FontProperties()
font_legend.set_family('serif')
font_legend.set_name('Times New Roman')
font_legend.set_size(14)

plt.close('all')

w = 9
h = 6
label_font_size = 20

sn.set_theme(style="whitegrid",font='Times New Roman',font_scale=1.5)

matplotlib.rcParams['pdf.fonttype'] = 42

# =============================================================================
# distance calculation
# =============================================================================
def sq_euc(s1, s2):
    '''
    Calculate squared euclidean distance.

    Parameters
    ----------
    s1 : size 1 * m * n. 
         where m is the number of variables, n is the timesteps.
    s2 : size 1 * m * n. 
         where m is the number of variables, n is the timesteps.

    Returns
    -------
    dist: Squared euclidean distance

    '''
    
    dist = ((s1 - s2) ** 2)
    return dist.flatten().sum()


# =============================================================================
# multivariate dynamic time warping
# https://github.com/ali-javed/dynamic-time-warping
# =============================================================================
def dtw_d(s1, s2, w):
    '''
    Calculate distance using multivariate dynamic time warping.

    Parameters
    ----------
    s1 : size 1 * m * n. 
         where m is the number of variables, n is the timesteps.
    s2 : size 1 * m * n. 
         where m is the number of variables, n is the timesteps.
    w  : window parameter, percent of size and is between 0 and 1. 
         0 is euclidean distance while 1 is maximum window size.

    Returns
    -------
    dist: resulting distance.

    '''

    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    
    s1_shape = np.shape(s1)
    s2_shape = np.shape(s2)
    
    if w < 0 or w > 1:
        print("Error: W should be between 0 and 1")
        return False
    if s1_shape[0] > 1 or s2_shape[0] > 1:
        print("Error: Please check input dimensions")
        return False
    if s1_shape[1] != s2_shape[1]:
        print("Error: Please check input dimensions. Number of variables not consistent.")
        return False
    if s1_shape[2] != s2_shape[2]:
        print("Warning: Length of time series not equal")
        
    # if window size is zero, it is plain euclidean distance
    if w == 0:
        dist = np.sqrt(sq_euc(s1, s2))
        return dist

    # get absolute window size
    w = int(np.ceil(w * s1_shape[2]))

    # adapt window size
    w = int(max(w, abs(s1_shape[2] - s2_shape[2])));
        
    # initilize    
    DTW = {}
    for i in range(-1, s1_shape[2]):
        for j in range(-1, s2_shape[2]):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    # print('entering dtw')
    for i in range(s1_shape[2]):
        for j in range(max(0, i - w), min(s2_shape[2], i + w)):
            #squared euc distance
            dist = sq_euc(s1[0,:,i], s2[0,:,j])
            #find optimal path
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], 
                                     DTW[(i,     j - 1)], 
                                     DTW[(i - 1, j - 1)])

    dist = np.sqrt(DTW[s1_shape[2] - 1, s2_shape[2] - 1])
    
    return dist

# ==================================================================================================
# k-means clustering
# https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670
# ==================================================================================================
# import seaborn as sns
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
import random

# centers = 5
# X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
# X_train = StandardScaler().fit_transform(X_train)
# sns.scatterplot(x = [X[0] for X in X_train],
#                 y = [X[1] for X in X_train],
#                 hue     = true_labels,
#                 palette = "deep",
#                 legend  = None)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

def euclidean(point, data):
    '''
    Euclidean distance between point & data.

    Parameters
    ----------
    point : (1, m, n)
    data  : (N, 1, m, n)

    Returns
    -------
    dist : (N,)

    '''
    dist = []
    for i in range(len(data)):
        dist.append(dtw_d(point, data[i], 1))
    return dist

# =============================================================================
# k-means clustering with dtw
# =============================================================================
class KMeans_DTW:
    def __init__(self, n_clusters=8, max_iter=10):
        '''
        Initialization of parameters.
        
        Parameters
        ----------
        n_clusters : Number of clusters. The default is 8.
        max_iter :   Number of iterations. The default is 10.

        Returns
        -------
        None.

        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X_train):
        '''
        Training of the DTW-based Kmeans clustering.
        
        Parameters
        ----------
        X_train : size N * 1 * m * n
                  where N is the number of samples, m is the number of variables,
                  n is the length of the time series

        Returns
        -------
        None.

        '''
        # Randomly select centroid start points, uniformly distributed across the domain of the dataset
        
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [random.uniform(min_, max_) for _ in range(self.n_clusters)]
        
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized with probabilities proportional to their distances to the first
        
        # Pick a random point from train data for first centroid
        print('Init ... ...')
        self.centroids = [random.choice(X_train)]
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            # print([euclidean(centroid, X_train) for centroid in self.centroids])
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            # print(dists)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]
        
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            print('Iteration: ', iteration)
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
                
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                # Catch any np.nans, resulting from a centroid having no points
                if np.isnan(centroid).any():  
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
    
    def evaluate(self, X):
        '''
        Classification of the input data.

        Parameters
        ----------
        X : size N * 1 * m * n
            where N is the number of samples, m is the number of variables,
            n is the length of the time series

        Returns
        -------
        centroids : list. (1, m, n)
        centroid_idxs : int

        '''
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs

# =============================================================================
# 
# =============================================================================
import pickle

with open("TakeoverQuality_.pkl", "rb") as fp:     # Unpickling
    TakeoverQualityData = pickle.load(fp) 

def get_Data_Spd_Acc_Data(Data, TOR):
    '''
    Extract the data from the dataset.

    Parameters
    ----------
    Data : list of lists
    TOR  : str

    Returns
    -------
    AccData   : list of lists. Acceleration
    SpdData   : list of lists. Speed
    SteerData : list of lists. Steering wheel angle
    YawData   : list of lists. Yaw rate

    '''
    print('TOR: ', TOR)
    AccData    = [] 
    SpdData    = []
    SteerData  = []
    YawData    = []
    # RollData   = []
    # PitchData  = []
    # DisToCoL   = []
    # TurnSignal = []
    
    for i in range(len(Data)):
        if Data[i][1] == TOR:
            # print(Data[i][0], Data[i][2].shape)
            AccData.append(Data[i][2]['Acceleration'].values.tolist())
            SpdData.append(Data[i][2]['Speed'].values.tolist())
            SteerData.append(Data[i][2]['SteeringWheel'].values.tolist())
            YawData.append(Data[i][2]['Yaw'].values.tolist())
            # RollData.append(Data[i][2]['Roll'].values.tolist())
            # PitchData.append(Data[i][2]['Pitch'].values.tolist())
            # DisToCoL.append(Data[i][2]['DistanceToCoL'].values.tolist())
            # TurnSignal.append(Data[i][2]['TurnSignal'].values.tolist())
    
    return AccData, SpdData, SteerData, YawData
           # RollData, PitchData, DisToCoL, TurnSignal

# =============================================================================
# Test the algorithm
# =============================================================================
# AccData, SpdData, SteerData, YawData, _, _, _, TurnSignal = get_Data_Spd_Acc_Data(Data, 'TOR6', 5)

# X_train = []
# for i in range(len(AccData)):
#     s = []
#     temp = []
#     temp.append((AccData[i] - np.array(AccData).min())/(np.array(AccData).max() - np.array(AccData).min()))
#     temp.append((SpdData[i] - np.array(SpdData).min())/(np.array(SpdData).max() - np.array(SpdData).min()))
#     temp.append((SteerData[i] - np.array(SteerData).min())/(np.array(SteerData).max() - np.array(SteerData).min()))
#     temp.append((YawData[i] - np.array(YawData).min())/(np.array(YawData).max() - np.array(YawData).min()))
#     s.append(temp)
#     X_train.append(np.array(s))

# centers = 3
# kmeans = KMeans_DTW(n_clusters=centers)
# kmeans.fit(X_train)
# class_centers, classification = kmeans.evaluate(X_train)
# print('classes: ', classification)

# TOR1 : 0:  14  1:  21  2:  13 or 0:  13  1:  21  2:  14 or 0:  15  1:  20  2:  13 or 0:  13  1:  20  2:  15
## [1, 0, 0, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 1, 2, 1, 2, 2, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 2, 2, 0, 0, 1, 0, 1, 1, 2, 2, 1],
## [1, 2, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 2, 1, 1, 2, 0, 1, 1, 1, 0, 1, 0, 0, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 0, 0, 2, 2, 1, 2, 1, 1, 0, 0, 1],
#- [0, 1, 1, 1, 0, 0, 0, 2, 1, 0, 1, 0, 0, 1, 1, 0, 1, 2, 1, 1, 0, 2, 0, 2, 2, 1, 1, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 0, 1, 0, 0, 2, 2, 1] 
#- [2, 1, 1, 1, 2, 2, 2, 0, 1, 2, 1, 2, 2, 1, 1, 2, 1, 0, 1, 1, 2, 0, 2, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 2, 2, 0, 0, 1]

# TOR2 : 0:  30  1:  3   2:  15 or 0:  3   1:  29  2:  16 or 0:  20  1:  18  2:  10 or 0:  3   1:  29  2:  16 or 0:  18  1:  7   2:  23
#        0:  30  1:  3   2:  15 or 0:  30  1:  15  2:  3
## [0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 2, 1, 2, 0, 2, 2, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2],
#- [1, 1, 0, 1, 2, 2, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2]
#  [1, 0, 2, 0, 0, 1, 0, 0, 1, 0, 2, 2, 2, 0, 2, 1, 2, 0, 0, 1, 1, 0, 0, 2, 0, 2, 1, 2, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 2, 1, 1, 0, 0, 0, 1, 0, 1] 
#- [1, 1, 0, 1, 2, 2, 1, 1, 1, 2, 2, 0, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2]
#  [2, 0, 1, 0, 2, 2, 0, 0, 2, 2, 1, 1, 2, 0, 1, 2, 1, 0, 0, 2, 2, 0, 0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 2, 2, 1, 2, 2, 0, 0, 0, 2, 0, 2]
## [0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 2, 1, 2, 0, 2, 2, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2] 
## [0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 1, 2, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1]

# TOR3 : 0:  28  1:  11  2:  9  or 0:  8   1:  15  2:  25 or 0:  8   1:  37  2:  3  or 0:  15  1:  8   2:  25 or 0:  25  1:  8   2:  15
#        0:  9   1:  28  2:  11
#  [2, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 2, 1, 0, 0, 0, 1, 2, 1] 
## [0, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 0, 1, 2, 0, 0, 2, 1, 2, 0, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 0, 1, 1, 0, 1, 2, 2, 2, 1, 0, 1]
#  [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 0, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1] 
## [1, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 1, 0, 2, 1, 1, 2, 0, 2, 1, 0, 2, 0, 2, 2, 2, 2, 0, 0, 2, 1, 0, 0, 1, 0, 2, 2, 2, 0, 1, 0]
## [1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 2, 0, 1, 1, 0, 2, 0, 1, 2, 0, 2, 0, 0, 0, 0, 2, 2, 0, 1, 2, 2, 1, 2, 0, 0, 0, 2, 1, 2]
#  [0, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 1, 1, 0, 0, 1, 2, 1, 0, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 0, 2, 1, 0, 2, 1, 1, 1, 2, 0, 2] 
 
# TOR4 : 0:  26  1:  10  2:  12 or 0:  9   1:  13  2:  26 or 0:  26  1:  13  2:  9  or 0:  13  1:  9   2:  26 or 0:  9   1:  26  2:  13 
#- [1, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 1, 2, 0, 1, 2, 0, 2, 2, 0, 0, 0, 1, 0, 2, 2, 0, 2, 0],
## [0, 2, 2, 2, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1, 0, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 0, 2, 1, 1, 0, 1, 2]
## [2, 0, 0, 0, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 2, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 2, 0, 1, 1, 2, 1, 0] 
## [1, 2, 2, 2, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 2, 2, 1, 2, 0, 2, 2, 2, 2, 0, 1, 0, 2, 2, 0, 2, 0, 0, 0, 2, 2, 1, 2, 0, 0, 1, 0, 2]
##  [0, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 0, 1, 2, 1, 1, 1, 1, 2, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 1, 0, 1, 2, 2, 0, 2, 1] 

# TOR5 : 0:  11  1:  15  2:  22 or 0:  13  1:  15  2:  20 or 0:  15  1:  19  2:  14 or 0:  17  1:  15  2:  16 or 0:  17  1:  16  2:  15
#        0:  7   1:  12  2:  29 or 0:  15  1:  22  2:  11 or 0:  30  1:  12  2:  6  or 0:  17  1:  16  2:  15
#* [2, 2, 2, 2, 0, 0, 1, 0, 2, 1, 2, 2, 0, 2, 2, 2, 1, 2, 1, 2, 0, 2, 2, 1, 1, 2, 0, 2, 0, 1, 1, 1, 2, 0, 1, 2, 1, 1, 0, 2, 2, 1, 2, 0, 1, 2, 1, 0]
#  [2, 2, 2, 0, 0, 0, 1, 0, 2, 1, 2, 2, 0, 2, 2, 2, 1, 0, 1, 2, 0, 2, 2, 1, 1, 2, 0, 2, 0, 1, 1, 1, 2, 0, 1, 2, 1, 1, 0, 2, 2, 1, 2, 0, 1, 2, 1, 0] 
#  [1, 1, 1, 2, 2, 2, 0, 2, 1, 0, 1, 1, 2, 1, 1, 1, 0, 2, 0, 1, 2, 1, 1, 0, 0, 1, 2, 1, 2, 0, 0, 0, 1, 2, 0, 1, 0, 0, 2, 1, 1, 0, 1, 2, 0, 2, 0, 2] 
## [0, 2, 2, 0, 0, 0, 1, 0, 2, 1, 2, 2, 0, 2, 2, 2, 1, 0, 1, 2, 0, 2, 0, 1, 1, 2, 0, 2, 0, 1, 1, 1, 2, 0, 1, 2, 1, 1, 0, 2, 0, 1, 2, 0, 1, 0, 1, 0]
## [0, 1, 1, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 1, 1, 1, 2, 0, 2, 1, 0, 1, 0, 2, 2, 1, 0, 1, 0, 2, 2, 2, 1, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1, 0, 2, 0, 2, 0]
#  [2, 2, 2, 2, 2, 2, 1, 2, 2, 0, 2, 2, 1, 2, 2, 2, 1, 2, 0, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 0, 2, 2, 0, 2, 0, 0, 1, 2, 2, 1, 2, 1, 0, 2, 1, 2]
#* [1, 1, 1, 1, 2, 2, 0, 2, 1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 0, 1, 2, 1, 1, 0, 0, 1, 2, 1, 2, 0, 0, 0, 1, 2, 0, 1, 0, 0, 2, 1, 1, 0, 1, 2, 0, 1, 0, 2]
#  [0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 2, 0, 0, 2, 0, 2, 2, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
## [0, 1, 1, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 1, 1, 1, 2, 0, 2, 1, 0, 1, 0, 2, 2, 1, 0, 1, 0, 2, 2, 2, 1, 0, 2, 1, 2, 2, 0, 1, 0, 2, 1, 0, 2, 0, 2, 0]

# TOR6 : 0:  27  1:  5   2:  16 or 0:  11  1:  11  2:  26 or 0:  26  1:  17  2:  5 or  0:  3   1:  27  2:  18 or 0:  27  1:  11  2:  10
#        0:  17  1:  3   2:  28 or 0:  11  1:  10  2:  27 or 0:  17  1:  26  2:  5 or  0:  26  1:  17  2:  5
#- [0, 2, 0, 2, 0, 0, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 2, 1, 0, 2, 2, 0, 0, 2]     
#  [2, 1, 2, 1, 2, 2, 1, 0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 0, 2, 0, 1, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 0]
## [0, 1, 0, 1, 0, 0, 1, 1, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 1, 2, 0, 1, 1, 0, 0, 1]
#  [1, 2, 1, 2, 1, 1, 2, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 2, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 0, 1, 2, 2, 1, 1, 2]
#* [0, 2, 0, 2, 0, 0, 2, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1]
#  [2, 0, 2, 2, 2, 2, 0, 0, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 1, 0, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0, 1, 2, 0, 0, 2, 2, 0]
#* [2, 1, 2, 1, 2, 2, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 2, 2, 2, 2, 1, 2, 2, 0, 0, 2, 2, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 0]
## [1, 0, 1, 0, 1, 1, 0, 0, 2, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 2, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 0, 1, 0, 2, 1, 0, 0, 1, 1, 0]
## [0, 1, 0, 1, 0, 0, 1, 1, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 1, 2, 0, 1, 1, 0, 0, 1]
    
# TOR7 : 0:  16  1:  13  2:  19 or 0:  13  1:  16  2:  19 or 0:  13  1:  18  2:  17 or 0:  17  1:  19  2:  12
## [2, 0, 2, 2, 0, 1, 1, 0, 1, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, 1, 2, 2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 0, 2, 2, 0, 2, 0, 1, 2, 2, 1, 1, 0, 0, 0, 1, 1, 2]
## [2, 1, 2, 2, 1, 0, 0, 1, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 0, 2, 2, 1, 0, 0, 2, 0, 2, 1, 1, 0, 1, 2, 2, 1, 2, 1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 0, 2]
#- [1, 2, 1, 1, 2, 0, 0, 2, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 2, 0, 1, 1, 2, 0, 0, 1, 0, 1, 2, 2, 0, 2, 1, 1, 2, 1, 2, 0, 0, 1, 0, 0, 2, 2, 2, 0, 2, 1]
#- [1, 0, 1, 1, 0, 2, 2, 0, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 2, 1, 1, 0, 2, 2, 1, 2, 1, 0, 0, 2, 0, 1, 1, 0, 1, 0, 2, 1, 1, 2, 2, 0, 0, 0, 2, 0, 1]

# TOR8 : 0:  10  1:  20  2:  18 or 0:  14  1:  8   2:  26 or 0:  17  1:  21  2:  10 or 0:  21  1:  17  2:  10
#- [2, 1, 2, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 0, 1, 0, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 1, 2, 2, 0, 1, 1, 2, 1, 2, 0, 1, 2, 0, 2, 1, 1, 2, 1, 1, 1]    
#  [2, 0, 2, 2, 0, 2, 2, 1, 1, 0, 2, 2, 0, 2, 2, 1, 0, 0, 0, 2, 2, 2, 2, 1, 1, 2, 2, 2, 0, 2, 2, 0, 2, 2, 2, 0, 2, 1, 0, 2, 0, 2, 2, 0, 2, 1, 1, 0]
## [1, 0, 1, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 1, 1, 2, 0, 2, 2, 1, 1, 1, 1, 2, 0, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 2, 0, 1, 2, 1, 0, 0, 1, 0, 0, 0] 
## [0, 1, 0, 1, 1, 1, 1, 2, 2, 2, 1, 0, 1, 0, 0, 2, 1, 2, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 2, 1, 0, 2, 0, 1, 1, 0, 1, 1, 1] 
  
# TOR9 : 0:  11  1:  32  2:  5  or 0:  14  1:  16  2:  18 or 0:  5   1:  32  2:  11
## [1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 2, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 2, 0, 1, 1, 0, 1, 1, 1, 1, 1]
## [1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 2, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 2, 0, 1, 1, 0, 1, 1, 1, 1, 1] 
#  [2, 2, 0, 2, 0, 1, 2, 2, 1, 1, 0, 0, 1, 2, 0, 1, 0, 2, 2, 1, 1, 0, 0, 1, 1, 0, 1, 0, 2, 1, 1, 2, 1, 0, 2, 1, 2, 2, 0, 0, 2, 1, 0, 2, 2, 2, 2, 1]
## [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 0, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 0, 2, 1, 1, 2, 1, 1, 1, 1, 1]

# TOR10: 0:  28  1:  3   2:  16 or 0:  3   1:  28  2:  16 or 0:  36  1:  10  2:  1  or 0:  28  1:  3   2:  16
## [2, 2, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 0]
## [2, 2, 1, 1, 2, 1, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 0, 1, 1, 1, 1, 2, 2, 1, 2, 1] 
#  [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0]
## [2, 2, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 0]

# TOR11: 0:  19  1:  9   2:  19 or 0:  19  1:  9   2:  19 or 0:  9   1:  19  2:  19 or 0:  19  1:  19  2:  9
## [0, 2, 2, 0, 1, 1, 2, 0, 2, 2, 2, 1, 1, 2, 0, 0, 2, 0, 2, 0, 1, 0, 0, 0, 2, 0, 2, 2, 2, 0, 1, 2, 1, 2, 2, 1, 0, 2, 2, 0, 0, 0, 0, 1, 2, 0, 0] 
## [0, 2, 2, 0, 1, 1, 2, 0, 2, 2, 2, 1, 1, 2, 0, 0, 2, 0, 2, 0, 1, 0, 0, 0, 2, 0, 2, 2, 2, 0, 1, 2, 1, 2, 2, 1, 0, 2, 2, 0, 0, 0, 0, 1, 2, 0, 0] 
## [1, 2, 2, 1, 0, 0, 2, 1, 2, 2, 2, 0, 0, 2, 1, 1, 2, 1, 2, 1, 0, 1, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 0, 2, 2, 0, 1, 2, 2, 1, 1, 1, 1, 0, 2, 1, 1]
## [1, 0, 0, 1, 2, 2, 0, 1, 0, 0, 0, 2, 2, 0, 1, 1, 0, 1, 0, 1, 2, 1, 1, 1, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 0, 2, 1, 0, 0, 1, 1, 1, 1, 2, 0, 1, 1]

# TOR12: 0:  17  1:  21  2:  9  or 0:  14  1:  22  2:  11 or 0:  11  1:  22  2:  14 or 0:  11  1:  22  2:  14
#  [1, 1, 2, 0, 1, 1, 0, 1, 0, 2, 1, 1, 2, 2, 1, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 1, 2, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 2, 0, 0] 
## [1, 1, 0, 2, 0, 1, 2, 0, 2, 0, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 0, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 0, 1, 1, 2, 2, 0, 1, 2] 
## [1, 1, 2, 0, 2, 1, 0, 2, 0, 2, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 2, 1, 1, 0, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 1, 1, 2, 1, 1, 0, 0, 2, 1, 0] 
## [1, 1, 2, 0, 2, 1, 0, 2, 0, 2, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 2, 1, 1, 0, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 1, 1, 2, 1, 1, 0, 0, 2, 1, 0]

# classes = [[2, 0, 0, 0, 2, 1, 0, 2, 1, 2, 1, 1, 0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 2, 0, 2, 2, 2, 2, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0],         
#            [2, 1, 0, 1, 0, 1, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 2, 1, 2, 1, 0, 2, 0, 0, 2, 2, 0, 1, 0, 2, 2],
#            [2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 1, 0, 2, 2, 2, 2, 0, 1, 2, 1, 1, 0, 2, 2, 1, 0, 2, 0, 1, 1, 2, 2, 2, 0, 1, 2, 2, 2, 0, 2, 1, 2, 0],
#            [0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 0, 2],
#            [1, 1, 0, 2, 2, 2, 0, 2, 1, 2, 0, 0, 1, 0, 2, 0, 2, 1, 2, 2, 1, 2, 1, 1, 0, 0, 2, 1, 2, 0, 2, 0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1, 0, 0, 1, 1, 1],
#            [1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 0, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 0, 1, 1, 1, 0, 0, 1, 1, 2, 1, 2, 0, 1, 2, 1],
#           ]

classes = [[1, 0, 0, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 1, 2, 1, 2, 2, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 2, 2, 0, 0, 1, 0, 1, 1, 2, 2, 1], # TOR1
           [0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 2, 1, 2, 0, 2, 2, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2], # TOR2
           [0, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 0, 1, 2, 0, 0, 2, 1, 2, 0, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 0, 1, 1, 0, 1, 2, 2, 2, 1, 0, 1], # TOR3
           [0, 2, 2, 2, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1, 0, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 0, 2, 1, 1, 0, 1, 2], # TOR4
           [2, 2, 2, 2, 0, 0, 1, 0, 2, 1, 2, 2, 0, 2, 2, 2, 1, 2, 1, 2, 0, 2, 2, 1, 1, 2, 0, 2, 0, 1, 1, 1, 2, 0, 1, 2, 1, 1, 0, 2, 2, 1, 2, 0, 1, 2, 1, 0], # TOR5
           [0, 2, 0, 2, 0, 0, 2, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1], # TOR6
           [2, 0, 2, 2, 0, 1, 1, 0, 1, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, 1, 2, 2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 0, 2, 2, 0, 2, 0, 1, 2, 2, 1, 1, 0, 0, 0, 1, 1, 2], # TOR7
           [1, 0, 1, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 1, 1, 2, 0, 2, 2, 1, 1, 1, 1, 2, 0, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 2, 0, 1, 2, 1, 0, 0, 1, 0, 0, 0], # TOR8
           [1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 2, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 2, 0, 1, 1, 0, 1, 1, 1, 1, 1], # TOR9
           [2, 2, 0, 0, 2, 0, 0, 0, 2, 1, 0, 0, 2, 1, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 2, 0],    # TOR10
           [0, 2, 2, 0, 1, 1, 2, 0, 2, 2, 2, 1, 1, 2, 0, 0, 2, 0, 2, 0, 1, 0, 0, 0, 2, 0, 2, 2, 2, 0, 1, 2, 1, 2, 2, 1, 0, 2, 2, 0, 0, 0, 0, 1, 2, 0, 0],    # TOR11
           [1, 1, 0, 2, 0, 1, 2, 0, 2, 0, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 0, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 0, 1, 1, 2, 2, 0, 1, 2]]    # TOR12

classes_ = [[1, 0, 0, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 0, 1, 1, 0, 2, 1, 1, 1, 2, 1, 2, 2, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 2, 2, 0, 0, 1, 0, 1, 1, 2, 2, 1], # TOR1
            [2, 2, 0, 2, 2, 1, 2, 2, 2, 1, 1, 0, 1, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1], # TOR2
            [2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 1, 0, 2, 2, 0, 1, 0, 2, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 2, 1, 1, 2, 1, 0, 0, 0, 1, 2, 1], # TOR3
            [1, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 2, 1, 2, 0, 0, 2, 0, 2, 2, 2, 0, 0, 1, 0, 2, 2, 1, 2, 0], # TOR4
            # [0, 0, 0, 0, 1, 1, 2, 1, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 2, 0, 1, 0, 0, 2, 2, 0, 1, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 1, 0, 0, 2, 0, 1, 2, 0, 2, 1], # TOR5
            [1, 0, 0, 1, 1, 1, 2, 1, 0, 2, 0, 0, 1, 0, 0, 0, 2, 1, 2, 0, 1, 0, 1, 2, 2, 0, 1, 0, 1, 2, 2, 2, 0, 1, 2, 0, 2, 2, 1, 0, 1, 2, 0, 1, 2, 1, 2, 1],
            # [0, 2, 0, 2, 0, 0, 2, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 1], # TOR6
            [0, 1, 0, 1, 0, 0, 1, 1, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 1, 2, 0, 1, 1, 0, 0, 1],
            [0, 2, 0, 0, 2, 1, 1, 2, 1, 2, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 0, 0, 2, 1, 1, 0, 1, 0, 2, 2, 1, 2, 0, 0, 2, 0, 2, 1, 0, 0, 1, 1, 2, 2, 2, 1, 1, 0], # TOR7
            [0, 1, 0, 1, 1, 1, 1, 2, 2, 2, 1, 0, 1, 0, 0, 2, 1, 2, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 2, 1, 0, 2, 0, 1, 1, 0, 1, 1, 1], # TOR8
            [2, 2, 0, 2, 0, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 0, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 0, 2, 1, 2, 2, 0, 1, 2, 2, 1, 2, 2, 2, 2, 2], # TOR9
            [2, 2, 1, 1, 2, 1, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 0, 1, 1, 1, 1, 2, 2, 1, 2, 1],    # TOR10
            [1, 0, 0, 1, 2, 2, 0, 1, 0, 0, 0, 2, 2, 0, 1, 1, 0, 1, 0, 1, 2, 1, 1, 1, 0, 1, 0, 0, 0, 1, 2, 0, 2, 0, 0, 2, 1, 0, 0, 1, 1, 1, 1, 2, 0, 1, 1],    # TOR11
            [1, 1, 0, 2, 0, 1, 2, 0, 2, 0, 1, 1, 0, 0, 0, 0, 2, 2, 1, 1, 0, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 0, 1, 1, 2, 2, 0, 1, 2]]    # TOR12


color = ['k', 'g', 'r']

line  = ['-', '--', ':']

colorList = [['k', 'r', 'g'],
             ['g', 'k', 'r'],
             ['g', 'r', 'k'],
             ['r', 'g', 'k'],
             ['r', 'g', 'k'],
             ['k', 'r', 'g'],
             ['g', 'r', 'k'],
             ['r', 'k', 'g'],
             ['r', 'g', 'k'],
             ['r', 'k', 'g'],
             ['r', 'g', 'k'],
             ['k', 'r', 'g']             
            ]

lineList  = [['-', ':', '--'],
             ['--', '-', ':'],
             ['--', ':', '-'],
             [':', '--', '-'],
             [':', '--', '-'],
             ['-', ':', '--'],
             ['--', ':', '-'],
             [':', '-', '--'],
             [':', '--', '-'],
             [':', '-', '--'],
             [':', '--', '-'],
             ['-', ':', '--']            
            ]

TORList = ['TOR1', 'TOR2', 'TOR3', 'TOR4',  'TOR5',  'TOR6',
           'TOR7', 'TOR8', 'TOR9', 'TOR10', 'TOR11', 'TOR12']

def plot_classification_results(data, classes):
    '''
    Plot the results.

    Parameters
    ----------
    data    : size N * 1 * m * n
              where N is the number of samples, m is the number of variables,
              n is the length of the time series
    classes : list

    Returns
    -------
    None.

    '''
    AccData, SpdData, SteerData, YawData = get_Data_Spd_Acc_Data(data, TORList[0])
    time = np.linspace(0, len(AccData[0])//60, len(AccData[0]))
    print('\n')
    
    for tor in range(12):
        AccData = []
        SpdData = []
        SteerData = []
        YawData = []
        
        AccData, SpdData, SteerData, YawData = get_Data_Spd_Acc_Data(data, TORList[tor])
        
        classification = classes[tor]
        
        fig = plt.figure()
        fig.tight_layout(pad=10.0)
        ax = fig.add_subplot(2,2,1)
        ax.grid(0)
        labelS = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labelS:
            label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
  
        SpdData_ = [[],[],[]]
        for i in range(len(SpdData)):
            SpdData_[classification[i]].append(SpdData[i])                
            plt.plot(time, SpdData[i], color=colorList[tor][classification[i]], linestyle=lineList[tor][classification[i]], alpha=0.3)
         
        for i in range(3):
            SpdData_[i] = np.array(SpdData_[i])
            mean = SpdData_[i].sum(axis=0)/len(SpdData_[i])
            # print('Spd mean: ', mean.sum()/len(mean))
            # print('Spd min : ', mean.min())
            # print('Spd max : ', mean.max())
            plt.plot(time, mean, color=colorList[tor][i], linestyle=lineList[tor][i], linewidth=4)
       
        # ax.set_xlabel(r'Time $(s)$', font = font)
        ax.set_ylabel(r'Speed $(km/h)$', font = font)
        ax.set_ylim(top=105, bottom = -5)
        
        # plt.show()
        
        # plt.figure()
        ax = fig.add_subplot(2,2,2)
        ax.grid(0)
        labelS = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labelS:
            label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
        
        AccData_ = [[],[],[]]
        for i in range(len(AccData)):
            AccData_[classification[i]].append(AccData[i])   
            plt.plot(time, AccData[i], color=colorList[tor][classification[i]], linestyle=lineList[tor][classification[i]], alpha=0.3)
        
        for i in range(3):
            AccData_[i] = np.array(AccData_[i])
            mean = AccData_[i].sum(axis=0)/len(AccData_[i])
            # print('Acc mean: ', mean.sum()/len(mean))
            # print('Acc min : ', mean.min())
            # print('Acc max : ', mean.max())
            plt.plot(time, mean, color=colorList[tor][i], linestyle=lineList[tor][i], linewidth=4)
        
        # ax.set_xlabel(r'Time $(s)$', font = font)
        ax.set_ylabel(r'Acceleration $(m/s^2)$', font = font)
        ax.set_ylim(top=10.5)
        # plt.show()
        
        # plt.figure()
        ax = fig.add_subplot(2,2,3)
        ax.grid(0)
        labelS = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labelS:
            label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
        
        SteerData_ = [[],[],[]]
        for i in range(len(SteerData)):
            SteerData_[classification[i]].append(SteerData[i])
            plt.plot(time, SteerData[i], color=colorList[tor][classification[i]], linestyle=lineList[tor][classification[i]], alpha=0.3)
        
        for i in range(3):
            SteerData_[i] = np.array(SteerData_[i])
            mean = SteerData_[i].sum(axis=0)/len(SteerData_[i])
            # print('Steer mean: ', mean.sum()/len(mean))
            # print('Steer min : ', mean.min())
            # print('Steer max : ', mean.max())
            plt.plot(time, mean, color=colorList[tor][i], linestyle=lineList[tor][i], linewidth=4)
        
        ax.set_xlabel(r'Time $(s)$', font = font)
        ax.set_ylabel(r'Steering Wheel $(^\circ)$', font = font)
        ax.set_ylim(top=55, bottom = -65)
        # plt.show()
        
        # plt.figure()
        ax = fig.add_subplot(2,2,4)
        ax.grid(0)
        labelS = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labelS:
            label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
        
        YawData_ = [[],[],[]]
        for i in range(len(YawData)):
            YawData_[classification[i]].append(YawData[i])
            plt.plot(time, np.array(YawData[i])/np.pi*180, color=colorList[tor][classification[i]], linestyle=lineList[tor][classification[i]], alpha=0.3)
        
        for i in range(3):
            YawData_[i] = np.array(YawData_[i])
            mean = YawData_[i].sum(axis=0)/len(YawData_[i])/np.pi*180
            # print('Yaw mean: ', mean.sum()/len(mean))
            # print('Yaw min : ', mean.min())
            # print('Yaw max : ', mean.max())
            plt.plot(time, mean, color=colorList[tor][i], linestyle=lineList[tor][i], linewidth=4)
        
        ax.set_xlabel(r'Time $(s)$', font = font)
        ax.set_ylabel(r'Yaw Rate $(^\circ/s)$', font = font)
        ax.set_ylim(top=15, bottom = -15)
        
        fig.suptitle(TORList[tor], y=0.9, va='baseline')
        
        fig.set_size_inches(15, 8)
            
        name = TORList[tor] + '_' + str(len(AccData[0])//60) + '.pdf'
        plt.savefig(name, bbox_inches="tight")  
        
        plt.show()

def plot_classification_results_(data, classes):
    '''
    Plot the results.

    Parameters
    ----------
    data    : size N * 1 * m * n
              where N is the number of samples, m is the number of variables,
              n is the length of the time series
    classes : list

    Returns
    -------
    None.

    '''
    AccData, SpdData, SteerData, YawData = get_Data_Spd_Acc_Data(data, TORList[0])
    time = np.linspace(0, len(AccData[0])//60, len(AccData[0]))
    print('\n')
    
    for tor in range(12):
        AccData = []
        SpdData = []
        SteerData = []
        YawData = []
        
        AccData, SpdData, SteerData, YawData = get_Data_Spd_Acc_Data(data, TORList[tor])
        
        classification = classes[tor]
        
        fig = plt.figure()
        fig.tight_layout(pad=10.0)
        ax = fig.add_subplot(2,2,1)
        ax.grid(0)
        labelS = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labelS:
            label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
  
        SpdData_ = [[],[],[]]
        for i in range(len(SpdData)):
            SpdData_[classification[i]].append(SpdData[i])                
            plt.plot(time, SpdData[i], color=color[classification[i]], linestyle=line[classification[i]], alpha=0.3)
         
        for i in range(3):
            SpdData_[i] = np.array(SpdData_[i])
            mean = SpdData_[i].sum(axis=0)/len(SpdData_[i])
            # print('Spd mean: ', mean.sum()/len(mean))
            # print('Spd min : ', mean.min())
            # print('Spd max : ', mean.max())
            plt.plot(time, mean, color=color[i], linestyle=line[i], linewidth=4)
       
        # ax.set_xlabel(r'Time $(s)$', font = font)
        ax.set_ylabel(r'Speed $(km/h)$', font = font)
        ax.set_ylim(top=105, bottom = -5)
        
        # plt.show()
        
        # plt.figure()
        ax = fig.add_subplot(2,2,2)
        ax.grid(0)
        labelS = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labelS:
            label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
        
        AccData_ = [[],[],[]]
        for i in range(len(AccData)):
            AccData_[classification[i]].append(AccData[i])   
            plt.plot(time, AccData[i], color=color[classification[i]], linestyle=line[classification[i]], alpha=0.3)
        
        for i in range(3):
            AccData_[i] = np.array(AccData_[i])
            mean = AccData_[i].sum(axis=0)/len(AccData_[i])
            # print('Acc mean: ', mean.sum()/len(mean))
            # print('Acc min : ', mean.min())
            # print('Acc max : ', mean.max())
            plt.plot(time, mean, color=color[i], linestyle=line[i], linewidth=4)
        
        # ax.set_xlabel(r'Time $(s)$', font = font)
        ax.set_ylabel(r'Acceleration $(m/s^2)$', font = font)
        ax.set_ylim(top=10.5)
        # plt.show()
        
        # plt.figure()
        ax = fig.add_subplot(2,2,3)
        ax.grid(0)
        labelS = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labelS:
            label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
        
        SteerData_ = [[],[],[]]
        for i in range(len(SteerData)):
            SteerData_[classification[i]].append(SteerData[i])
            plt.plot(time, SteerData[i], color=color[classification[i]], linestyle=line[classification[i]], alpha=0.3)
        
        for i in range(3):
            SteerData_[i] = np.array(SteerData_[i])
            mean = SteerData_[i].sum(axis=0)/len(SteerData_[i])
            # print('Steer mean: ', mean.sum()/len(mean))
            # print('Steer min : ', mean.min())
            # print('Steer max : ', mean.max())
            plt.plot(time, mean, color=color[i], linestyle=line[i], linewidth=4)
        
        ax.set_xlabel(r'Time $(s)$', font = font)
        ax.set_ylabel(r'Steering Wheel $(^\circ)$', font = font)
        ax.set_ylim(top=55, bottom = -65)
        # plt.show()
        
        # plt.figure()
        ax = fig.add_subplot(2,2,4)
        ax.grid(0)
        labelS = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labelS:
            label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
        
        YawData_ = [[],[],[]]
        for i in range(len(YawData)):
            YawData_[classification[i]].append(YawData[i])
            plt.plot(time, np.array(YawData[i])/np.pi*180, color=color[classification[i]], linestyle=line[classification[i]], alpha=0.3)
        
        for i in range(3):
            YawData_[i] = np.array(YawData_[i])
            mean = YawData_[i].sum(axis=0)/len(YawData_[i])/np.pi*180
            # print('Yaw mean: ', mean.sum()/len(mean))
            # print('Yaw min : ', mean.min())
            # print('Yaw max : ', mean.max())
            plt.plot(time, mean, color=color[i], linestyle=line[i], linewidth=4)
        
        ax.set_xlabel(r'Time $(s)$', font = font)
        ax.set_ylabel(r'Yaw Rate $(^\circ/s)$', font = font)
        ax.set_ylim(top=15, bottom = -15)
        
        fig.suptitle(TORList[tor], y=0.9, va='baseline')
        
        fig.set_size_inches(15, 8)
            
        name = TORList[tor] + '_' + str(len(AccData[0])//60) + '.pdf'
        plt.savefig(name, bbox_inches="tight")  
        
        plt.show()


# =============================================================================
# 
# =============================================================================
def clustering_dtw(data, tor):
    '''
    Main function of processing the data.
    
    Parameters
    ----------
    data : size N * 1 * m * n
           where N is the number of samples, m is the number of variables,
           n is the length of the time series
    tor  : int (0-11)

    Returns
    -------
    None.

    '''
    AccData = []
    SpdData = []
    SteerData = []
    YawData = []

    AccData, SpdData, SteerData, YawData = get_Data_Spd_Acc_Data(data, TORList[tor])
    
    # classification = classes[tor * 3 + dom]
    time = np.linspace(0, len(AccData[0])//60, len(AccData[0]))
    X_train = []
    # print(len(AccData))
    for i in range(len(AccData)):
        s = []
        temp = []
        temp.append((AccData[i]   - np.array(AccData).min())  / (np.array(AccData).max()   - np.array(AccData).min()))
        temp.append((SpdData[i]   - np.array(SpdData).min())  / (np.array(SpdData).max()   - np.array(SpdData).min()))
        temp.append((SteerData[i] - np.array(SteerData).min())/ (np.array(SteerData).max() - np.array(SteerData).min()))
        temp.append((YawData[i]   - np.array(YawData).min())  / (np.array(YawData).max()   - np.array(YawData).min()))
        s.append(temp)
        X_train.append(np.array(s))
    
    centers = 3
    kmeans = KMeans_DTW(n_clusters=centers)
    kmeans.fit(X_train)
    class_centers, classification = kmeans.evaluate(X_train)
    print('classes: ', classification, 
          '0: ',  classification.count(0),
          '1: ',  classification.count(1),
          '2: ',  classification.count(2))
    
    color = ['k', 'r', 'g']
    line  = ['-', ':', '--']
    fig = plt.figure()
    fig.tight_layout(pad=5.0)
    ax = fig.add_subplot(2,2,1)
    ax.grid(0)
    labelS = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labelS:
        label.set_fontname('Times New Roman')
        label.set_fontsize(label_font_size)
    for i in range(len(SpdData)):
        plt.plot(time, SpdData[i], color=color[classification[i]], linestyle=line[classification[i]])
    # ax.set_xlabel(r'Time $(s)$', font = font)
    ax.set_ylabel(r'Speed $(km/h)$', font = font)
    
    # plt.show()
    
    # plt.figure()
    ax = fig.add_subplot(2,2,2)
    ax.grid(0)
    labelS = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labelS:
        label.set_fontname('Times New Roman')
        label.set_fontsize(label_font_size)
    for i in range(len(AccData)):
        plt.plot(time, AccData[i], color=color[classification[i]], linestyle=line[classification[i]])
    # ax.set_xlabel(r'Time $(s)$', font = font)
    ax.set_ylabel(r'Acceleration $(m/s^2)$', font = font)
    # plt.show()
    
    # plt.figure()
    ax = fig.add_subplot(2,2,3)
    ax.grid(0)
    labelS = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labelS:
        label.set_fontname('Times New Roman')
        label.set_fontsize(label_font_size)
    for i in range(len(SteerData)):
        plt.plot(time, SteerData[i], color=color[classification[i]], linestyle=line[classification[i]])
    ax.set_xlabel(r'Time $(s)$', font = font)
    ax.set_ylabel(r'Steering Wheel $(^\circ)$', font = font)
    # plt.show()
    
    # plt.figure()
    ax = fig.add_subplot(2,2,4)
    ax.grid(0)
    labelS = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labelS:
        label.set_fontname('Times New Roman')
        label.set_fontsize(label_font_size)
    for i in range(len(YawData)):
        plt.plot(time, np.array(YawData[i])/np.pi*180, color=color[classification[i]], linestyle=line[classification[i]])
    ax.set_xlabel(r'Time $(s)$', font = font)
    ax.set_ylabel(r'Yaw Rate $(^\circ/s)$', font = font)
    
    fig.set_size_inches(15, 8)
    
    plt.show()

# plot_classification_results_(TakeoverQualityData, classes_)

# for tor in range(2,3):
#     for dom in range(3):
#         clustering_dtw(Data_15, tor, dom)

# for tor in range(2):        
#     clustering_dtw(TakeoverQualityData, tor+4)

# clustering_dtw(TakeoverQualityData, 3)
# temp = []
# for tor in range(12):
#     temp = temp + classes_[tor]
# temp = pd.DataFrame(temp, columns=['DrivingStyle'])
# print(temp)
# temp.to_csv('DrivingStyles.csv')

for tor in range(12):
    print(color[0], ' ', classes[tor].count(0), ' ', color[1], ' ', classes[tor].count(1), ' ', color[2], ' ', classes[tor].count(2))
    