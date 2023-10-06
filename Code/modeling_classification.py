# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:13:47 2023

@author: Nakano_Lab
"""

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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

import warnings
warnings.filterwarnings("ignore")

data = pd.read_excel('Dataset.xlsx')
# data = data[data.columns[0:76]]
# data.drop(columns=['Number','Participant'],inplace=True)
# print(data)
target = ['Time','Crash','DrivingStyle']

categorical = ['Age', 'Gender', 'YearsDriving', 'ExperienceADS', 'HandInUse', 'Scenario', 'Scenario_C', 'DoM', 'TB', 'Weather', 'Road', 'EoR']

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

X = data[features1 + features2 + features4 + features5 + features6 + features7 + target]
X = X.dropna()
y1 = X[target[0]]
y2 = X[target[1]]
y3 = X[target[2]]
X = X.drop(columns=target)

# from sklearn.preprocessing import StandardScaler

# std = StandardScaler()
# X = pd.DataFrame(std.fit_transform(X),columns = X.columns)
# print(X)

# from imblearn.over_sampling import SMOTENC, SMOTE, ADASYN
# from imblearn.under_sampling import RandomUnderSampler
# oversample  = SMOTENC(random_state=42, categorical_features=[0,1,2,3,9,12,13,14,15,16,17,25])
# oversample  = SMOTENC(random_state=42, categorical_features=[0,1,2,3,9,12,13,14,15,16,17,25], sampling_strategy=0.5)
# undersample = RandomUnderSampler(random_state=42)
# X, y = oversample.fit_resample(X, y2)
# X, y = undersample.fit_resample(X, y)


# from sklearn.model_selection import train_test_split 
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# oversample = SMOTENC(random_state=42, categorical_features=[0,1,2,3,9,12,13,14,15,16,17,25], sampling_strategy=0.5)
# x_train, y_train = oversample.fit_resample(x_train, y_train)

# C_50per_926
# x_train = pd.read_csv('x_train_C_50per_926.csv')
# y_train = pd.read_csv('y_train_C_50per_926.csv')
# x_test  = pd.read_csv('x_test_C_50per_926.csv')
# y_test  = pd.read_csv('y_test_C_50per_926.csv')

# C_80per_963
x_train = pd.read_csv('x_train_C_80per_963.csv')
y_train = pd.read_csv('y_train_C_80per_963.csv')
x_test  = pd.read_csv('x_test_C_80per_963.csv')
y_test  = pd.read_csv('y_test_C_80per_963.csv')

# C_80per20per_938
# x_train = pd.read_csv('x_train_C_80per20per_938.csv')
# y_train = pd.read_csv('y_train_C_80per20per_938.csv')
# x_test  = pd.read_csv('x_test_C_80per20per_938.csv')
# y_test  = pd.read_csv('y_test_C_80per20per_938.csv')

# C_961
# x_train = pd.read_csv('x_train_C_961.csv')
# y_train = pd.read_csv('y_train_C_961.csv')
# x_test  = pd.read_csv('x_test_C_961.csv')
# y_test  = pd.read_csv('y_test_C_961.csv')

x_train_ = x_train[features1 + features2]
x_test_  = x_test[features1 + features2]
y_train  = y_train['Crash']
y_test   = y_test['Crash'] 


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


def model_train_test_xgboost_C(window_size):
    '''
    Train and test XGBoost classifier

    Parameters
    ----------
    window_size : int
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    print('Window Size: ', window_size*3, ' s')
    print('-------------------------------------------------')
    
    x_train_ = x_train[features[0] + features[window_size]]
    x_test_  = x_test[features[0] + features[window_size]]

    model = XGBClassifier(verbosity = 0, use_label_encoder=False)
    cv = RepeatedKFold(n_splits = 10, n_repeats = 5, random_state = 1)
    # define model evaluation method
    parameters = {'learning_rate': [.005, .01, .02], #so called `eta` value
                  'max_depth': [3, 4, 5],
                  # 'min_child_weight': [4],
                  # 'silent': [1],
                  'subsample': [0.7, 0.8, 0.9],
                  # 'colsample_bytree': [0.7, 0.8, 0.9],
                  'n_estimators': [300, 350, 400]
                  }
    
    model_grid = GridSearchCV(model,
                              parameters,
                              cv = cv,
                              n_jobs = 5,
                              verbose=True,
                              scoring='accuracy')
    
    model_grid.fit(x_train_, y_train)
    
    print('Best Score: ', model_grid.best_score_)
    print('Best Parameters: ', model_grid.best_params_)
    
    y_pred = model_grid.predict(x_test_)

    print('Accuracy: ',  accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred))
    print('Recall: ', recall_score(y_test, y_pred))
    print('F1 Score: ', f1_score(y_test, y_pred))
    print('-------------------------------------------------')

# for i in range(1):
#     model_train_test_xgboost_C(i + 1)

def xgboost_shap_C(pattern):
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
  
    model = XGBClassifier(max_depth=5, 
                         learning_rate=0.02, 
                         subsample=0.7, 
                         n_estimators=350,
                         eval_metric=accuracy_score)

    model.fit(x_train_, y_train)
    
    y_pred = model.predict(x_test_)
    
    print('Accuracy: ',  accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred))
    print('Recall: ', recall_score(y_test, y_pred))
    print('F1 Score: ', f1_score(y_test, y_pred))
    print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
    print('-------------------------------------------------')
    
    explainer = shap.TreeExplainer(model, x_train_)
    shap_values = explainer(x_train_)
    
    # shap.plots.bar(shap_values, max_display=38)
    # shap.plots.beeswarm(shap_values, max_display=38)
    
    # shap.plots.bar(shap_values[1], max_display=38)
    # shap.plots.waterfall(shap_values[1], max_display=38)
    
    # shap.plots.scatter(shap_values[:, "CenterScreen_3s"])
    # shap.plots.scatter(shap_values[:, "CenterScreen_3s"], color=shap_values)
    # shap.plots.scatter(shap_values[:, "CenterScreen_3s"], color=shap_values[:, "DoM"])
    
    if pattern == 0:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        shap.plots.bar(shap_values, max_display = 38, show=False)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            # label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
        plt.savefig('../20230913/Figures/' + 'importance-bar_C.pdf', bbox_inches = "tight")
        plt.show()
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        shap.plots.beeswarm(shap_values, max_display = 38, show=False)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        for label in labels:
            # label.set_fontname('Times New Roman')
            label.set_fontsize(label_font_size)
        plt.savefig('../20230913/Figures/' + 'importance-beeswarm_C.pdf', bbox_inches = "tight")
        plt.show()
    
    featureList = ['Predictability', 'Frustration', 'Scenario_C', 'Neuroticism',
                   'Road', 'Mental_Demand', 'Gender', 'Criticality', 'Effort', 'avgSpd']
    labelName   = ['Predictability', 'Frustration', 'Crticality of Scenario', 'Neuroticism',
                   'Road', 'Mental Demand', 'Gender', 'Criticality', 'Effort', 'Average Speed']
    fileName    = ['Predictability', 'Frustration', 'ScenarioCriticality', 'Neuroticism',
                   'Road', 'MentalDemand', 'Gender', 'Criticality', 'Effort', 'avgSpd']
    
    if pattern == 1:
        for i in range(10):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            # if i == 6:
            #     shap.plots.scatter(shap_values[:, featureList[i]], ax=ax, show=False, xmin=0.01)
            # else:
            #     shap.plots.scatter(shap_values[:, featureList[i]], ax=ax, show=False)
            shap.plots.scatter(shap_values[:, featureList[i]], ax=ax, show=False)
            labels = ax.get_xticklabels() + ax.get_yticklabels()        
            for label in labels:
                # label.set_fontname('Times New Roman')
                label.set_fontsize(label_font_size)
            # if i == 1:
            #     plt.xlim(0.9, 1.1) 
            ax.set_xlabel(labelName[i])
            ax.set_ylabel('SHAP Value')
            plt.savefig('../20230913/Figures/' + 'MainEffects_C_' + fileName[i] + '.pdf', bbox_inches = "tight")
            plt.show()
            
    if pattern == 2:
        for i in range(10):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            
            shap.plots.scatter(shap_values[:, featureList[i]], ax=ax, color=shap_values, show=False)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            for label in labels:
                # label.set_fontname('Times New Roman')
                label.set_fontsize(label_font_size)
                
            ax.set_xlabel(labelName[i])
            ax.set_ylabel('SHAP Value')
            # ax2.set_ylabel(font=font)
            plt.savefig('../20230913/Figures/' + 'InteractionEffects_C_' + fileName[i] + '.pdf', bbox_inches = "tight")
            plt.show()
    
    localList = [316, 361]
    fileNameLocal = ['high', 'low']
    if pattern == 3:
        for i in range(2):
            # num = random.randint(0, 643)
            # print(num)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            shap.plots.waterfall(shap_values[localList[i]], max_display = 38, show=False)
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            for label in labels:
                # label.set_fontname('Times New Roman')
                label.set_fontsize(label_font_size)
            plt.savefig('../20230913/Figures/' + 'LocalExplanation_C_' + fileNameLocal[i] + '.pdf', bbox_inches = "tight")
            plt.show()
            
xgboost_shap_C(3)
    
def model_train_test_logisticRegression():
    '''
    Train and test logistic regression classifier

    Returns
    -------
    None.

    '''
    from sklearn.linear_model import LogisticRegression
    
    classifier = LogisticRegression()
    classifier.fit(x_train_onehot, y_train)#predicting the test set results
    y_pred = classifier.predict(x_test_onehot)
    
    print('Accuracy: ',  accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred))
    print('Recall: ', recall_score(y_test, y_pred))
    print('F1 Score: ', f1_score(y_test, y_pred))
    print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
    print('-------------------------------------------------')

# model_train_test_logisticRegression()

def model_train_test_kNN_C():
    '''
    Train and test k-nearest-neighbors classifier

    Returns
    -------
    None.

    '''
    from sklearn.neighbors import KNeighborsClassifier
    
    for n_neighbors in range(3,10):
        for weights in ["uniform", "distance"]:           
            classifier = KNeighborsClassifier(n_neighbors, weights=weights)
            classifier.fit(x_train_onehot, y_train)#predicting the test set results
            y_pred = classifier.predict(x_test_onehot)
                
            print('n_neighbors: ', n_neighbors, ' weights: ', weights)
            
            print('Accuracy: ',  accuracy_score(y_test, y_pred))
            print('Precision: ', precision_score(y_test, y_pred))
            print('Recall: ', recall_score(y_test, y_pred))
            print('F1 Score: ', f1_score(y_test, y_pred))
            print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
            print('-------------------------------------------------')

# model_train_test_kNN_C()

def model_train_test_SVM_C():
    '''
    Train and test support vector machine classifier

    Returns
    -------
    None.

    '''
    from sklearn.svm import SVC
    
    for kernel in ['linear', 'poly', 'rbf']:
        for degree in [2, 3, 4, 5]:
            for C in [1, 10, 100]:
                classifier = SVC(kernel = kernel, degree = degree, C = C)
                classifier.fit(x_train_onehot, y_train)#predicting the test set results
                y_pred = classifier.predict(x_test_onehot)
                
                print('kernel: ', kernel, ' degree: ', degree, ' C: ', C)
    
                print('Accuracy: ',  accuracy_score(y_test, y_pred))
                print('Precision: ', precision_score(y_test, y_pred))
                print('Recall: ', recall_score(y_test, y_pred))
                print('F1 Score: ', f1_score(y_test, y_pred))
                print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
                print('-------------------------------------------------')
                
# model_train_test_SVM_C()

def model_train_test_DT_C():
    '''
    Train and test decision tree classifier

    Returns
    -------
    None.

    '''
    from sklearn.tree  import DecisionTreeClassifier
    
    for criterion in ['gini', 'entropy']:
        for splitter in ["best", "random"]:     
            for max_depth in [3, 4, 5, 6]:
                classifier = DecisionTreeClassifier(criterion = criterion, splitter = splitter, max_depth = max_depth)
                classifier.fit(x_train_onehot, y_train) # predicting the test set results
                y_pred = classifier.predict(x_test_onehot)
                    
                print('criterion: ', criterion, ' splitter: ', splitter, ' max_depth: ', max_depth)
                
                print('Accuracy: ',  accuracy_score(y_test, y_pred))
                print('Precision: ', precision_score(y_test, y_pred))
                print('Recall: ', recall_score(y_test, y_pred))
                print('F1 Score: ', f1_score(y_test, y_pred))
                print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
                print('-------------------------------------------------')

# model_train_test_DT_C()

def model_train_test_RF_C():
    '''
    Train and test random forest classifier

    Returns
    -------
    None.

    '''
    from sklearn.ensemble  import RandomForestClassifier
    
    for criterion in ['gini', 'entropy']:
        for n_estimators in [100, 200, 300, 400]:     
            for max_depth in [3, 4, 5, 6]:
                regressor = RandomForestClassifier(criterion = criterion, n_estimators = n_estimators, max_depth = max_depth)
                regressor.fit(x_train_onehot, y_train)#predicting the test set results
                y_pred = regressor.predict(x_test_onehot)
                    
                print('criterion: ', criterion, ' n_estimators: ', n_estimators, ' max_depth: ', max_depth)
                
                print('Accuracy: ',  accuracy_score(y_test, y_pred))
                print('Precision: ', precision_score(y_test, y_pred))
                print('Recall: ', recall_score(y_test, y_pred))
                print('F1 Score: ', f1_score(y_test, y_pred))
                print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
                print('-------------------------------------------------')

# model_train_test_RF_C()
