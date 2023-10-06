# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:29:25 2023

@author: Nakano_Lab
"""

import pandas as pd
import numpy as np
import os
import re
import seaborn as sn
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from matplotlib.font_manager import FontProperties

participants = ['G2N01', 'G2N02', 'G2N03', 'G2N04', 'G2N05', 'G2N06', 
                'G2N07', 'G2N08', 'G2N09', 'G2N10', 'G2N11', 'G2N12', 
                'G2N13', 'G2N14', 'G2N15', 'G2N16', 'G2N17', 'G2N18',
                'G3N09', 'G3N10', 'G3N11', 'G4N01', 'G4N02', 'G4N03',
                'G4N04', 'G4N05', 'G4N06', 'G4N07', 'G4N08', 'G4N09',
                'G4N10', 'G4N11', 'G4N12', 'G4N13', 'G4N14', 'G4N15',
                'G5N01', 'G5N02', 'G5N03', 'G5N04', 'G5N05', 'G5N06',
                'G5N07', 'G5N08', 'G5N09', 'G5N10', 'G5N11', 'G5N12']

names = ['UNIX_sec','UTC_datetime','is_left','select_left','key', 'correct','wrong']

columns1 = ['num_correct','num_wrong','time','Spd','correct_rate']
columns2 = columns2 = ['avgSpd', 'avgCorrectRate']

avgSpd_avgCorrectRate = []

for p in range(len(participants)):
    df = []
    for dirname, _, filenames in os.walk('I:/ExperimentData/DataProcessing/SuRT-Data/' + participants[p]):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            df_temp = pd.read_csv(os.path.join(dirname, filename), sep=',',
                                  names = names, skiprows = [0])
            df.append(df_temp)
               
    # df[0].head(1)
    temp = []
    for i in range(len(df)):
        num_correct = df[i].tail(1)['correct'].values[0]
        num_wrong   = df[i].tail(1)['wrong'].values[0]
        
        # calculate duration of the SuRT
        time_start_h = float(df[i].head(1)['UTC_datetime'].values[0][9:11])
        time_end_h   = float(df[i].tail(1)['UTC_datetime'].values[0][9:11])
        time_start_m = float(df[i].head(1)['UTC_datetime'].values[0][11:13])
        time_end_m   = float(df[i].tail(1)['UTC_datetime'].values[0][11:13])
        time_start_s = float(df[i].head(1)['UTC_datetime'].values[0][13:])
        time_end_s   = float(df[i].tail(1)['UTC_datetime'].values[0][13:])
        
        hours   = time_end_h - time_start_h
        minutes = time_end_m - time_start_m
        seconds = time_end_s - time_start_s
        
        time = hours * 3600 + minutes * 60 + seconds 
        
        spd = time / (num_correct + num_wrong)
        
        correct_rate = num_correct / (num_correct + num_wrong)
        
        temp.append([num_correct, num_wrong, time, spd, correct_rate])
    
    df_temp = pd.DataFrame(temp, columns = columns1)
    
    total_time = df_temp['time'].sum()
    total_number = df_temp['num_correct'].sum() + df_temp['num_wrong'].sum()
    
    avgSpd = total_time / total_number
    avgCorrectRate = df_temp['num_correct'].sum()/total_number
    
    # print('avgSpd', avgSpd)
    # print('avgCorrectRate', avgCorrectRate)
    
    avgSpd_avgCorrectRate.append([avgSpd, avgCorrectRate])

avgSpd_avgCorrectRate = pd.DataFrame(avgSpd_avgCorrectRate, columns = columns2)

print(avgSpd_avgCorrectRate)

# avgSpd_avgCorrectRate.to_csv('ReactionSpeed.csv')
# df = pd.concat(df,ignore_index=True)         