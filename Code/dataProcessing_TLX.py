# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:42:51 2023

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

participants = ['G4N01', 'G4N02', 'G4N03',
                'G4N04', 'G4N05', 'G4N06', 'G4N07', 'G4N08', 'G4N09',
                'G4N10', 'G4N11', 'G4N12', 'G4N13', 'G4N14', 'G4N15',
                'G5N01', 'G5N02', 'G5N03', 'G5N04', 'G5N05', 'G5N06',
                'G5N07', 'G5N08', 'G5N09', 'G5N10', 'G5N11', 'G5N12']

df = []
for dirname, _, filenames in os.walk('TLX-Data/'):
    for filename in filenames:
            print(os.path.join(dirname, filename))
            df.append(pd.read_excel(os.path.join(dirname, filename),
                      names=['Scenario','Mental Demand','Physical Demand','Temproal Demand',
                             'Predictability', 'Criticality', 'Performance', 'Effort', 'Frustration'
                            ],
            ))
            
TOR_scores = []            
for TOR in range(12):                   # 12 takeover scenarios
    temp = pd.DataFrame()
    for participant in range(len(df)):  # 48 participants
        temp = pd.concat([temp, df[participant].iloc[TOR]], axis = 1)
    TOR_scores.append(temp.transpose())
    
avgScores = pd.DataFrame()
for i in range(len(TOR_scores)):
    avgScores = pd.concat([avgScores, TOR_scores[i].mean(axis=0)], axis = 1)

avgScores.columns = ['TOR1', 'TOR2', 'TOR3', 'TOR4',  'TOR5',  'TOR6',
                     'TOR7', 'TOR8', 'TOR9', 'TOR10', 'TOR11', 'TOR12']

# print(avgScores.transpose())
# avgScores.transpose().to_csv('avgScenarioScores.csv')