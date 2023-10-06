# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:33:37 2023

@author: Chao Huang
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

duration = 1800


TOR_name = ['TOR1','TOR2','TOR3','TOR4','TOR5','TOR6','TOR7','TOR8','TOR9', 'TOR10', 'TOR11', 'TOR12']

usecols=['LocalTime','Acceleration','Speed','SteeringWheel','DistanceToCoL','Yaw','TurnSignal']
 
def TakeoverQualityData():
    '''
    Process excel to extract takeover quality data.

    Returns
    -------
    df : Array of DataFrame

    '''
    for dirname, _, filenames in os.walk('G:/ExperimentData/DataProcessing/DrivingSimulatorData/'):
        df = []
        for filename in filenames:
            participant = filename[0:5]
            tor = filename[6:-5]
            print(participant, tor)
            data = pd.read_excel(os.path.join(dirname, filename))[usecols]

            df.append([participant, tor, data])
    
    return df

def ProcessTakeoverQualityData():
    open_file = open('TakeoverQuality.pkl','rb')
    data      = pickle.load(open_file)
    open_file.close()
    
    data_tq = []
    for i in range(len(data)):
        df = data[i][2]
        
        AccMax       = df['Acceleration'].max()
        
        SpdMean      = df['Speed'].mean()
        SpdStd       = df['Speed'].std()
        
        SteerMean    = df['SteeringWheel'].mean()
        SteerMax     = abs(df['SteeringWheel']).max()
        SteerStd     = df['SteeringWheel'].std()
        
        DisToCoLMean = df['DistanceToCoL'].mean()
        DisToCoLMax  = abs(df['DistanceToCoL']).max()
        DisToCoLStd  = df['DistanceToCoL'].std()
        
        YawMean      = df['Yaw'].mean()/np.pi*180
        YawMax       = abs(df['Yaw']).max()/np.pi*180
        YawStd       = df['Yaw'].std()/np.pi*180
        
        TurnSignal   = df['TurnSignal'].sum() > 0    
        
        data_tq.append([data[i][0], data[i][1], 
                        AccMax, 
                        SpdMean, SpdStd, 
                        SteerMean, SteerMax, SteerStd,
                        DisToCoLMean, DisToCoLMax, DisToCoLStd,
                        YawMean, YawMax, YawStd,
                        TurnSignal])
    
    data_tq = pd.DataFrame(data_tq, 
                           columns = ['Participant', 'Scenario',
                                      'AccMax', 
                                      'SpdMean', 'SpdStd',
                                      'SteerMean', 'SteerMax', 'SteerStd',
                                      'DisToCoLMean', 'DisToCoLMax', 'DisToCoLStd',
                                      'YawMean', 'YawMax', 'YawStd',
                                      'TurnSignal'])
    
    return data_tq

# data = ProcessTakeoverQualityData()
# data.to_csv('TakeoverQualityData.csv')

df = TakeoverQualityData()
open_file = open('TakeoverQuality_.pkl','wb')
pickle.dump(df, open_file)
open_file.close()

            
            
            