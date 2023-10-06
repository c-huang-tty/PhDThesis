# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:28:58 2023

@author: Chao Huang
"""
import pandas as pd
# from mpl_toolkits import mplot3d
import seaborn as sn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import re
# from pathlib import Path
# import os
# import pickle
# import glob
# from PIL import Image
# import re
# from scipy import stats
# import itertools
# import pingouin as pg
# import math
# import turtle

directory = os.getcwd()
parent_directory = os.path.dirname(directory)

# %matplotlib inline

mpl.rcParams.update({'figure.max_open_warning': 0})

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_size(16)

font_legend = FontProperties()
font_legend.set_family('serif')
font_legend.set_name('Times New Roman')
font_legend.set_size(14)

font_legend_2 = FontProperties()
font_legend_2.set_family('serif')
font_legend_2.set_name('Times New Roman')
# font_legend_2.set
font_legend_2.set_size(9)

font_heatmap = FontProperties()
font_heatmap.set_family('serif')
font_heatmap.set_name('Times New Roman')
font_heatmap.set_size(12)

font_eyelidPupil = FontProperties()
font_eyelidPupil.set_family('serif')
font_eyelidPupil.set_name('Times New Roman')
font_eyelidPupil.set_size(12)

w = 9
h = 6
label_font_size = 14

plt.close('all')

w2 = 9
h2 = 4.5

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

# =============================================================================
# 
# =============================================================================
     
        
def time_cal(g_str): 
    df = []
    folder = ''
    if g_str[1] == '2':
        folder = 'G2'
    if g_str[1] == '3':
        folder = 'G3'
    if g_str[1] == '4':
        folder = 'G4'
    if g_str[1] == '5':
        folder = 'G5'
    
    for dirname, _, filenames in os.walk('../' + folder + '/' + g_str + '/DS/csv'):
        for filename in filenames:
            if re.search(r'timer_record', filename):
                print(os.path.join(dirname, filename))
                df.append(pd.read_csv(os.path.join(dirname, filename),sep=',',
                                      names=['Timer_flag','Inter_flag','Timer_steer','Timer_acc',
                                             'Timer_brake','Timer_init','Timer_RtI',
                                             'Timer_inter','Timer_RtA','Timer_stop'],
                                      skiprows = [0]))
               
    
    df = pd.concat(df,ignore_index=True)         
    
    return df['Timer_inter'] - df['Timer_RtI'] 

def time_steer_cal(g_str): 
    df = []
    for dirname, _, filenames in os.walk('../' + g_str + '/DS/csv'):
        for filename in filenames:
            if re.search(r'timer_record', filename):
                # print(os.path.join(dirname, filename))
                df.append(pd.read_csv(os.path.join(dirname, filename),sep=',',
                                      names=['Timer_flag','Inter_flag','Timer_steer','Timer_acc',
                                             'Timer_brake','Timer_init','Timer_RtI',
                                             'Timer_inter','Timer_RtA','Timer_stop'],
                                      skiprows = [0]))
               
    
    df = pd.concat(df,ignore_index=True)         
    
    return df['Timer_steer'] - df['Timer_RtI'] 

    
def get_inter_method(g_str): 
    df = []
    for dirname, _, filenames in os.walk('../' + g_str + '/DS/csv'):
        for filename in filenames:
            if re.search(r'timer_record', filename):
                # print(os.path.join(dirname, filename))
                df.append(pd.read_csv(os.path.join(dirname, filename),sep=',',
                                      names=['Timer_flag','Inter_flag','Timer_steer','Timer_acc',
                                             'Timer_brake','Timer_init','Timer_RtI',
                                             'Timer_inter','Timer_RtA','Timer_stop'],
                                      skiprows = [0]))
               
    
    df = pd.concat(df,ignore_index=True)         
    
    return df['Inter_flag']

def time_record_(g_str): 
    df = []
    folder = ''
    if g_str[1] == '2':
        folder = 'G2'
    if g_str[1] == '3':
        folder = 'G3'
    if g_str[1] == '4':
        folder = 'G4'
    if g_str[1] == '5':
        folder = 'G5'
    
    for dirname, _, filenames in os.walk('../' + folder + '/' + g_str + '/DS/csv'):
        for filename in filenames:
            if re.search(r'timer_record', filename):
                print(os.path.join(dirname, filename))
                df.append(pd.read_csv(os.path.join(dirname, filename),sep=',',
                                      names=['Timer_flag','Inter_flag','Timer_steer','Timer_acc',
                                             'Timer_brake','Timer_init','Timer_RtI',
                                             'Timer_inter','Timer_RtA','Timer_stop'],
                                      skiprows = [0]))
               
    
    df = pd.concat(df,ignore_index=True)         
    
    return df['Timer_RtI']

# =============================================================================
# 
# =============================================================================

participants = ['G2N01', 'G2N02', 'G2N03', 'G2N04', 'G2N05', 'G2N06', 
                'G2N07', 'G2N08', 'G2N09', 'G2N10', 'G2N11', 'G2N12', 
                'G2N13', 'G2N14', 'G2N15', 'G2N16', 'G2N17', 'G2N18',
                'G3N09', 'G3N10', 'G3N11', 'G4N01', 'G4N02', 'G4N03',
                'G4N04', 'G4N05', 'G4N06', 'G4N07', 'G4N08', 'G4N09',
                'G4N10', 'G4N11', 'G4N12', 'G4N13', 'G4N14', 'G4N15',
                'G5N01', 'G5N02', 'G5N03', 'G5N04', 'G5N05', 'G5N06',
                'G5N07', 'G5N08', 'G5N09', 'G5N10', 'G5N11', 'G5N12']

def TakeoverTime():
    time_inter = pd.DataFrame()
    for i in range(len(participants)):
        temp = time_cal(participants[i])
        time_inter = pd.concat([time_inter, temp], axis = 1)
    
    time_inter = time_inter.transpose()
    
    time_inter.index = participants
    
    time_inter.columns = ['TOR1','TOR2','TOR3','TOR4','TOR5','TOR6','TOR7','TOR8','TOR9','P','TOR10','TOR11','TOR12']
    
    # print(time_inter)
    
    time_inter_l = time_inter[['TOR1','TOR2','TOR3','TOR4','TOR5','TOR6','TOR7','TOR8','TOR9','TOR10','TOR11','TOR12']].melt(var_name='Scenario',value_name='Time')

    return time_inter_l

dataset = pd.read_csv('Dataset_Time_G.csv', sep=',')

takeover_time = dataset['Time']

print('25th percentile', np.nanpercentile(takeover_time.values, 25))
print('33th percentile', np.nanpercentile(takeover_time.values, 33))
print('50th percentile', np.nanpercentile(takeover_time.values, 50))
print('67th percentile', np.nanpercentile(takeover_time.values, 67))
print('75th percentile', np.nanpercentile(takeover_time.values, 75))

# temp = (takeover_time <= 3) & (takeover_time >= 1.8)
temp = takeover_time < 1.8
print(temp.sum())

print(takeover_time.describe())

print('Number of nan: ', takeover_time.isna().sum())
print('Number of nan: ', takeover_time.isna().sum())

def plotTakeoverTime():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax1 = sn.histplot(data=dataset, x="Time", kde=True,
                      line_kws={'lw': 2, 'ls': '--'})
    ax1.lines[0].set_color('crimson')
    # sn.kdeplot(data=dataset, x="Time", color='crimson', lw = 2, ls = '--', ax = ax1)
    
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in labels:
        label.set_fontname('Times New Roman')
        label.set_fontsize(label_font_size)
    
    ax.set_xlabel('Takeover Time', font=font)
    ax.set_ylabel('Count', font=font)
    
    fig.set_size_inches(w, h)
    plt.savefig('TakeoverTimeWithDensity.pdf', bbox_inches = "tight")
    
    plt.show()
    
# plotTakeoverTime()    

# def PercentileOfTakeoverTime(x):
#     if x < 1.8:
#         return 'L'
#     elif x <= 3:
#         return 'M'
#     else:
#         return 'H'

# dataset['Time_G'] = dataset['Time'].apply(PercentileOfTakeoverTime)

# dataset.to_csv('Dataset_Time_G.csv')

# df = TakeoverTime()    
# df.to_csv('TakeoverTime.csv')

# time_record = pd.DataFrame()
# for i in range(len(participants)):
#     temp = time_record_(participants[i])
#     time_record = pd.concat([time_record, temp], axis = 1)
    
# time_record = time_record.transpose()

# time_record.index = participants

# time_record.columns = ['TOR1','TOR2','TOR3','TOR4','TOR5','TOR6','TOR7','TOR8','TOR9','P','TOR10','TOR11','TOR12']
   
# time_record.to_csv('time_record.csv')