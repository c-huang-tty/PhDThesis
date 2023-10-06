# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 10:57:18 2023

@author: tjhua
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import random

from matplotlib.font_manager import FontProperties
import pickle

# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(16)

font_legend = FontProperties()
font_legend.set_family('serif')
font_legend.set_name('Times New Roman')
font_legend.set_size(12)

plt.close('all')

w = 9
h = 6
label_font_size = 14


data = pd.read_excel('Dataset.xlsx')
# data = data[data.columns[0:76]]
# data.drop(columns=['Number','Participant'],inplace=True)
# print(data)

target = ['Time','Crash','DrivingStyle']

categorical = ['Age', 'Gender', 'YearsDriving', 'ExperienceADS', 'HandInUse', 
               'Scenario', 'Scenario_C', 'DoM', 'TB', 'Weather', 'Road', 'EoR']

features1 = ['Age', 'Gender', 'YearsDriving', 'ExperienceADS',
            'Neuroticism', 'Extraversion', 'Openness', 'Agreeableness',
            'Conscientiousness', 'HandInUse', 'avgSpd', 'avgCorrectRate',
            'Scenario', 'Scenario_C', 'DoM', 'TB', 'Weather', 'Road',
            'Mental_Demand', 'Physical_Demand', 'Temproal_Demand', 'Predictability',
            'Criticality', 'Effort', 'Frustration', 'EoR']

features2 = ['CenterScreen_3s', 'Others_3s', 'LeftSide_3s', 'RightSide_3s',
            'EyelidOpeningMean_3s', 'EyelidOpeningStd_3s', 'EyelidOpeningAmp_3s',
            'PupilDiamterMean_3s', 'PupilDiamterStd_3s', 'PupilDiamterAmp_3s', 
            'NumFixations_3s', 'GazeEntropy_3s'
            ]
            
features3 = ['AccMax', 'SpdMean', 'SpdStd', 'SteerMean', 'SteerMax', 'YawMean', 'YawMax']

features4 = ['CenterScreen_6s', 'Others_6s', 'LeftSide_6s', 'RightSide_6s',
            'EyelidOpeningMean_6s', 'EyelidOpeningStd_6s', 'EyelidOpeningAmp_6s',
            'PupilDiamterMean_6s', 'PupilDiamterStd_6s', 'PupilDiamterAmp_6s', 
            'NumFixations_6s', 'GazeEntropy_6s']

features5 = ['CenterScreen_9s', 'Others_9s', 'LeftSide_9s', 'RightSide_9s',
            'EyelidOpeningMean_9s', 'EyelidOpeningStd_9s', 'EyelidOpeningAmp_9s',
            'PupilDiamterMean_9s', 'PupilDiamterStd_9s', 'PupilDiamterAmp_9s', 
            'NumFixations_9s', 'GazeEntropy_9s']

features6 = ['CenterScreen_12s', 'Others_12s', 'LeftSide_12s', 'RightSide_12s',
            'EyelidOpeningMean_12s', 'EyelidOpeningStd_12s', 'EyelidOpeningAmp_12s',
            'PupilDiamterMean_12s', 'PupilDiamterStd_12s', 'PupilDiamterAmp_12s', 
            'NumFixations_12s', 'GazeEntropy_12s']

features7 = ['CenterScreen_15s', 'Others_15s', 'LeftSide_15s', 'RightSide_15s',
            'EyelidOpeningMean_15s', 'EyelidOpeningStd_15s', 'EyelidOpeningAmp_15s',
            'PupilDiamterMean_15s', 'PupilDiamterStd_15s', 'PupilDiamterAmp_15s', 
            'NumFixations_15s', 'GazeEntropy_15s']

features = [features1, features2, features4, features5, features6, features7]

# X = data[features1 + features2 + features4 + features5 + features6 + features7 + target]
# X = X.dropna()
# y1 = X[target[0]]
# y2 = X[target[1]]
# y3 = X[target[2]]
# X = X.drop(columns=target)

# from sklearn.preprocessing import StandardScaler

# std = StandardScaler()
# X = pd.DataFrame(std.fit_transform(X),columns = X.columns)
# print(X)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(X, y1, test_size=0.2)

# Console 1
# x_train = pd.read_csv('x_train_648.csv')
# y_train = pd.read_csv('y_train_648.csv')
# x_test  = pd.read_csv('x_test_648.csv')
# y_test  = pd.read_csv('y_test_648.csv')

# Console 5
x_train = pd.read_csv('x_train_MC_503.csv')
y_train = pd.read_csv('y_train_MC_503.csv')
x_test  = pd.read_csv('x_test_MC_503.csv')
y_test  = pd.read_csv('y_test_MC_503.csv')

# Console 2
# x_train = pd.read_csv('x_train_HC_398.csv')
# y_train = pd.read_csv('y_train_HC_398.csv')
# x_test  = pd.read_csv('x_test_HC_398.csv')
# y_test  = pd.read_csv('y_test_HC_398.csv')

x_train_ = x_train[features1 + features2]
x_test_  = x_test[features1 + features2]
y_train  = y_train['Time']
y_test   = y_test['Time'] 

# ['Age', 'Gender', 'YearsDriving', 'ExperienceADS', 'Neuroticism',
#  'Extraversion', 'Openness', 'Agreeableness', 'Conscientiousness',
#  'HandInUse', 'avgSpd', 'avgCorrectRate', 'Scenario', 'Scenario_C',
#  'DoM', 'TB', 'Weather', 'Road', 'Mental_Demand', 'Physical_Demand',
#  'Temproal_Demand', 'Predictability', 'Criticality', 'Effort',
#  'Frustration', 'EoR', 'CenterScreen_3s', 'Others_3s', 'LeftSide_3s',
#  'RightSide_3s', 'EyelidOpeningMean_3s', 'EyelidOpeningStd_3s',
#  'EyelidOpeningAmp_3s', 'PupilDiamterMean_3s', 'PupilDiamterStd_3s',
#  'PupilDiamterAmp_3s', 'NumFixations_3s', 'GazeEntropy_3s']

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
# 38 categorical variables & 26 continuous variables
transformer = ColumnTransformer([
    ('Age', OneHotEncoder(),[0]),           # 3
    ('Gender', OneHotEncoder(),[1]),        # 2
    ('YearsDriving', OneHotEncoder(),[2]),  # 5
    ('ExperienceADS', OneHotEncoder(),[3]), # 3
    ('HandInUse', OneHotEncoder(),[9]),     # 2
    ('Scenario', OneHotEncoder(),[12]),     # 9
    ('Scenario_C', OneHotEncoder(),[13]),   # 2
    ('DoM', OneHotEncoder(),[14]),          # 3
    ('TB', OneHotEncoder(),[15]),           # 2
    ('Weather', OneHotEncoder(),[16]),      # 2
    ('Road', OneHotEncoder(),[17]),         # 3
    ('EoR', OneHotEncoder(),[25])           # 2
    ], remainder= MinMaxScaler())
x_train_onehot = transformer.fit_transform(x_train_)
x_test_onehot  = transformer.fit_transform(x_test_)

def model_train_test_xgboost(window_size):
    '''
    Train and test XGBoost regressor

    Parameters
    ----------
    window_size : int (0, 1, 2, 3, 4)

    Returns
    -------
    None.

    '''
    print('Window Size: ', window_size*3, ' s')
    print('-------------------------------------------------')
    
    x_train_ = x_train[features[0] + features[window_size]]
    x_test_  = x_test[features[0] + features[window_size]]

    model = XGBRegressor()
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    
    # define model evaluation method
    parameters = {'learning_rate': [.005, .01, .02], #so called `eta` value
                  'max_depth': [3, 4, 5],
                  # 'min_child_weight': [4],
                  # 'silent': [1],
                  'subsample': [0.7, 0.8, 0.9],
                  # 'colsample_bytree': [0.7, 0.8, 0.9],
                  'n_estimators': [400, 420, 440]
                 }
    
    model_grid = GridSearchCV(model,
                              parameters,
                              cv = cv,
                              n_jobs  = 5,
                              verbose = True,
                              scoring = 'neg_mean_absolute_error')
    
    model_grid.fit(x_train_, y_train)
    
    print('Best Score: ', model_grid.best_score_)
    print('Best Parameters: ', model_grid.best_params_)
    
    y_pred = model_grid.predict(x_test_)

    print('MAE: ', mean_absolute_error(y_test, y_pred))
    print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
    print('R2: ', r2_score(y_test, y_pred))
    print('-------------------------------------------------')

# for i in range(1):
#     model_train_test_xgboost(i + 1)

def xgboost_shap(pattern):
    '''
    Plot shap values

    Parameters
    ----------
    pattern : int
        0-glocal importance plot
        1-main effects
        2-interaction effects
        3-local importance plot.

    Returns
    -------
    None.

    '''
    print('Window Size: ', 3, ' s')
    print('-------------------------------------------------')
  
    model = XGBRegressor(max_depth=3, 
                         learning_rate=0.01, 
                         subsample=0.7, 
                         n_estimators=440,
                         eval_metric=mean_absolute_error)

    model.fit(x_train_, y_train)
    
    y_pred = model.predict(x_test_)
    
    print('MAE: ', mean_absolute_error(y_test, y_pred))
    print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
    print('R2: ', r2_score(y_test, y_pred))
    print('-------------------------------------------------')
    
    explainer = shap.TreeExplainer(model, x_train_)
    shap_values = explainer(x_train_)
    
    if pattern == 0:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        shap.plots.bar(shap_values, max_display = 38, show=False)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            # label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
        plt.savefig('../20230913/Figures/' + 'importance-bar.pdf', bbox_inches = "tight")
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        shap.plots.beeswarm(shap_values, max_display = 38, show=False)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            # label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
        plt.savefig('../20230913/Figures/' + 'importance-beeswarm.pdf', bbox_inches = "tight")
        plt.show()
       
    
    # shap.plots.bar(shap_values[1], max_display = 38)
    # shap.plots.waterfall(shap_values[1], max_display = 38)
    
    # shap.plots.scatter(shap_values[:, "CenterScreen_3s"], color=shap_values)
    # shap.plots.scatter(shap_values[:, "CenterScreen_3s"], color=shap_values[:, "DoM"])
    
    featureList = ['CenterScreen_3s', 'avgCorrectRate', 'Physical_Demand', 'Temproal_Demand',
                   'Neuroticism', 'Scenario', 'GazeEntropy_3s', 'Effort']
    labelName   = ['Percentage of Eyes on Road', 'Average Correction Rate', 'Physical Demand', 'Temporal Demand',
                   'Neuroticism', 'Scenario', 'Gaze Entropy', 'Effort']
    fileName    = ['CenterScreen', 'avgCorrectRate', 'PhysicalDemand', 'TemporalDemand',
                   'Neuroticism', 'Scenario', 'GazeEntropy', 'Effort']
    
    if pattern == 1:
        for i in range(8):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if i == 6:
                shap.plots.scatter(shap_values[:, featureList[i]], ax=ax, show=False, xmin=0.01)
            else:
                shap.plots.scatter(shap_values[:, featureList[i]], ax=ax, show=False)
            labels = ax.get_xticklabels() + ax.get_yticklabels()        
            for label in labels:
                # label.set_fontname('Times New Roman')
                label.set_fontsize(label_font_size)
            if i == 1:
                plt.xlim(0.92, 1.02) 
            ax.set_xlabel(labelName[i])
            ax.set_ylabel('SHAP Value')
            plt.savefig('../20230913/Figures/' + 'MainEffects_' + fileName[i] + '.pdf', bbox_inches = "tight")
            plt.show()
    
    if pattern == 2:
        for i in range(8):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if i == 6:
                shap.plots.scatter(shap_values[:, featureList[i]], ax=ax, color=shap_values, show=False, xmin=0.01)
            else:
                shap.plots.scatter(shap_values[:, featureList[i]], ax=ax, color=shap_values, show=False)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            for label in labels:
                # label.set_fontname('Times New Roman')
                label.set_fontsize(label_font_size)
            if i == 1:
                plt.xlim(0.92, 1.02) 
            ax.set_xlabel(labelName[i])
            ax.set_ylabel('SHAP Value')
            # ax2.set_ylabel(font=font)
            plt.savefig('../20230913/Figures/' + 'InteractionEffects_' + fileName[i] + '.pdf', bbox_inches = "tight")
            plt.show()
    
    # 182： 3.833
    # 234： 1.223
    # 222： 2.637
    localList = [234, 222, 182]
    fileNameLocal = ['short', 'medium', 'long']
    if pattern == 3:
        for i in range(3):
            # num = random.randint(0, 298)
            # print(num)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            shap.plots.waterfall(shap_values[localList[i]], max_display = 38, show=False)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            for label in labels:
                # label.set_fontname('Times New Roman')
                label.set_fontsize(label_font_size)
            plt.savefig('../20230913/Figures/' + 'LocalExplanation_' + fileNameLocal[i] + '.pdf', bbox_inches = "tight")
            plt.show()
        
xgboost_shap(2)

def model_train_test_linearRegression():
    '''
    Train and test linear regressor

    Returns
    -------
    None.

    '''
    from sklearn.linear_model import LinearRegression
    
    regressor = LinearRegression()
    regressor.fit(x_train_onehot, y_train)#predicting the test set results
    y_pred = regressor.predict(x_test_onehot)
    
    print('MAE: ', mean_absolute_error(y_test, y_pred))
    print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
    print('R2: ', r2_score(y_test, y_pred))
    print('-------------------------------------------------')

# model_train_test_linearRegression()

def model_train_test_kNN():
    '''
    Train and test k-nearest-neighbors regressor

    Returns
    -------
    None.

    '''
    from sklearn.neighbors import KNeighborsRegressor
    
    for n_neighbors in range(3,10):
        for weights in ["uniform", "distance"]:           
            regressor = KNeighborsRegressor(n_neighbors, weights=weights)
            regressor.fit(x_train_onehot, y_train)#predicting the test set results
            y_pred = regressor.predict(x_test_onehot)
                
            print('n_neighbors: ', n_neighbors, ' weights: ', weights)
            
            print('MAE: ', mean_absolute_error(y_test, y_pred))
            print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
            print('R2: ', r2_score(y_test, y_pred))
            print('-------------------------------------------------')
    
# model_train_test_kNN() 

def model_train_test_SVM():
    '''
    Train and test support vector machine regressor

    Returns
    -------
    None.

    '''
    from sklearn.svm import SVR
    
    for kernel in ['linear', 'poly', 'rbf‘]:
        for degree in [2, 3, 4, 5]:
            for C in [1, 10, 100]:
                regressor = SVR(kernel = kernel, degree = degree, C = C)
                regressor.fit(x_train_onehot, y_train)#predicting the test set results
                y_pred = regressor.predict(x_test_onehot)
                
                print('kernel: ', kernel, ' degree: ', degree, ' C: ', C)
    
                print('MAE: ', mean_absolute_error(y_test, y_pred))
                print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
                print('R2: ', r2_score(y_test, y_pred))
                print('-------------------------------------------------')

# model_train_test_SVM()

def model_train_test_DT():
    '''
    Train and test decision tree regressor

    Returns
    -------
    None.

    '''
    from sklearn.tree  import DecisionTreeRegressor
    
    for criterion in ['squared_error', 'absolute_error']:
        for splitter in ["best", "random"]:     
            for max_depth in [3, 4, 5, 6]:
                regressor = DecisionTreeRegressor(criterion = criterion, splitter = splitter, max_depth = max_depth)
                regressor.fit(x_train_onehot, y_train)#predicting the test set results
                y_pred = regressor.predict(x_test_onehot)
                    
                print('criterion: ', criterion, ' splitter: ', splitter, ' max_depth: ', max_depth)
                
                print('MAE: ', mean_absolute_error(y_test, y_pred))
                print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
                print('R2: ', r2_score(y_test, y_pred))
                print('-------------------------------------------------')

# model_train_test_DT()

def model_train_test_RF():
    '''
    Train and test random forest regressor

    Returns
    -------
    None.

    '''
    from sklearn.ensemble  import RandomForestRegressor
    
    for criterion in ['squared_error', 'absolute_error']:
        for n_estimators in [100, 200, 300, 400]:     
            for max_depth in [3, 4, 5, 6]:
                regressor = RandomForestRegressor(criterion = criterion, n_estimators = n_estimators, max_depth = max_depth)
                regressor.fit(x_train_onehot, y_train)#predicting the test set results
                y_pred = regressor.predict(x_test_onehot)
                    
                print('criterion: ', criterion, ' n_estimators: ', n_estimators, ' max_depth: ', max_depth)
                
                print('MAE: ', mean_absolute_error(y_test, y_pred))
                print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))
                print('R2: ', r2_score(y_test, y_pred))
                print('-------------------------------------------------')

# model_train_test_RF()