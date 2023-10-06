# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 17:08:41 2023

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
import pickle

participants = ['G2N01', 'G2N02', 'G2N03', 'G2N04', 'G2N05', 'G2N06', 
                'G2N07', 'G2N08', 'G2N09', 'G2N10', 'G2N11', 'G2N12', 
                'G2N13', 'G2N14', 'G2N15', 'G2N16', 'G2N17', 'G2N18',
                'G3N09', 'G3N10', 'G3N11', 'G4N01', 'G4N02', 'G4N03',
                'G4N04', 'G4N05', 'G4N06', 'G4N07', 'G4N08', 'G4N09',
                'G4N10', 'G4N11', 'G4N12', 'G4N13', 'G4N14', 'G4N15',
                'G5N01', 'G5N02', 'G5N03', 'G5N04', 'G5N05', 'G5N06',
                'G5N07', 'G5N08', 'G5N09', 'G5N10', 'G5N11', 'G5N12']

TOR = ['TOR1', 'TOR2', 'TOR3', 'TOR4',  'TOR5',  'TOR6',
       'TOR7', 'TOR8', 'TOR9', 'TOR10', 'TOR11', 'TOR12']

def EyesOnRoad():
    '''
    Whether drivers' gazes are on road at the moment of takeover request.

    Returns
    -------
    df : DataFrame

    '''
    for dirname, _, filenames in os.walk('SmartEyeData/log_G2_G3_G4_G5/excel/'):
        df = []
        for filename in filenames:
            participant = filename[0:5]
            tor = filename[6:-5]
            # print(participant, tor)
            temp = pd.read_excel(os.path.join(dirname, filename))['ClosestWorldIntersection.objectName'].iloc[900:905]
            # print(temp.to_numpy())
            if 'CenterScreen' in temp.to_numpy():
                EoR = 1
            else:
                EoR = 0
            print(participant, tor, EoR)
            df.append([participant, tor, EoR])
 
    df = pd.DataFrame(df, columns = ['Participants', 'Scenario', 'EoR'])
    
    return df

def excel2pkl():
    '''
    Convert excel format to pickle format.

    Returns
    -------
    None.

    '''
    for dirname, _, filenames in os.walk('SmartEyeData/log_G2_G3_G4_G5/'):
        for filename in filenames:
            filename_pkl = filename[:-5] + '.pkl'
            print(filename_pkl)
            temp = pd.read_excel(os.path.join(dirname, filename))
    
            open_file = open(filename_pkl,'wb')
            pickle.dump(temp, open_file)
            open_file.close()

def ReadFile(participant, tor):
    '''
    Parameters
    ----------
    participant : str
    tor : str

    Returns
    -------
    data : DataFrame

    '''
    path = 'SmartEyeData/log_G2_G3_G4_G5/pkl/' + participant + '_' + tor + '.pkl'
    open_file = open(path,'rb')
    data = pickle.load(open_file)
    open_file.close()
    
    return data

def GazeCountAOIs(participant, tor, window_size):
    '''
    Count gaze points in different AOIs given certain participant, tor and window size.

    Parameters
    ----------
    participant : str
    tor : str
    window_size : int (15, 12, 9, 6, 3)
    
    Returns
    -------
    AOI_gaze_count : DataFrame

    '''
    start = (15 - window_size) * 60
    end   = 15 * 60
    data = ReadFile(participant, tor)[start:end]  
    
    AOI_gaze_count = data.groupby(['ClosestWorldIntersection.objectName']).count()['FrameNumber']  
    
    return AOI_gaze_count

def BlinkCount(window_size):
    '''
    Count the number of blinks given certain window size.

    Parameters
    ----------
    window_size : int

    Returns
    -------
    df : DataFrame

    '''
    start = (15 - window_size) * 60
    end   = 15 * 60
    for dirname, _, filenames in os.walk('SmartEyeData/log_G2_G3_G4_G5/pkl/'):
        df = []
        for filename in filenames:
            participant = filename[0:5]
            tor = filename[6:-4]
            print(participant, tor)
            data = ReadFile(participant, tor)[start:end]['Blink']             
            data = data.loc[~(data==0)] 
            if len(data) == 0:
                nummber_of_blinks = 0
            else:
                nummber_of_blinks = data.iloc[len(data) - 1] - data.iloc[0] + 1
            df.append([participant, tor, nummber_of_blinks])
            
    cols = ['Participant',  'Scenario',
            'NumBlinks'  + '_' + str(window_size) + 's']
    
    df = pd.DataFrame(df, columns=cols)
       
    return df

def FixationCount(window_size):
    '''
    Count the number of fixatios given certain window size.

    Parameters
    ----------
    window_size : int

    Returns
    -------
    df : DataFrame

    '''
    start = (15 - window_size) * 60
    end   = 15 * 60
    for dirname, _, filenames in os.walk('SmartEyeData/log_G2_G3_G4_G5/pkl/'):
        df = []
        for filename in filenames:
            participant = filename[0:5]
            tor = filename[6:-4]
            print(participant, tor)
            data = ReadFile(participant, tor)[start:end]['Fixation']             
            data = data.loc[~(data==0)] 
            if len(data) == 0:
                nummber_of_fixations = 0
            else:
                nummber_of_fixations = data.iloc[len(data) - 1] - data.iloc[0] + 1
            df.append([participant, tor, nummber_of_fixations])
            
    cols = ['Participant',  'Scenario',
            'NumFixations'  + '_' + str(window_size) + 's']
    
    df = pd.DataFrame(df, columns=cols)
       
    return df
    
def PupilDiameterEyelidOpening_Mean_Std_Amplitude(participant, tor, window_size):
    '''
    Calculate mean, std, and amplitude of pupil diameter and eyelid opening
    given certain participant, tor, and window size.


    Parameters
    ----------
    participant : str
    tor : str
    window_size : int (15, 12, 9, 6, 3)
    
    Returns
    -------
    pupil_diameter_eyelid_opening : numpy array

    '''
    start = (15 - window_size) * 60
    end   = 15 * 60
    data = ReadFile(participant, tor)[start:end]  
    pupil_diameter_eyelid_opening = data[['PupilDiameter', 'EyelidOpening']]
    
    pupil_diameter_mean = pupil_diameter_eyelid_opening['PupilDiameter'].mean()
    pupil_diameter_std = pupil_diameter_eyelid_opening['PupilDiameter'].std()
    pupil_diameter_amplitude = pupil_diameter_eyelid_opening['PupilDiameter'].max() - pupil_diameter_eyelid_opening['PupilDiameter'].min()
    
    eyelid_opening_mean = pupil_diameter_eyelid_opening['EyelidOpening'].mean()
    eyelid_opening_std = pupil_diameter_eyelid_opening['EyelidOpening'].std()
    eyelid_opening_amplitude = pupil_diameter_eyelid_opening['EyelidOpening'].max() - pupil_diameter_eyelid_opening['EyelidOpening'].min()
    
    return [pupil_diameter_mean*1000, pupil_diameter_std*1000, pupil_diameter_amplitude*1000, 
            eyelid_opening_mean*1000, eyelid_opening_std*1000, eyelid_opening_amplitude*1000]

def PercentageOfEyesOnRoad(window_size):
    '''
    Calculate percentage of eye gazes in different AOIs
    given certain window size.

    Parameters
    ----------
    window_size : int (15, 12, 9, 6, 3)
    
    Returns
    -------
    df : DataFrame

    '''
    for dirname, _, filenames in os.walk('SmartEyeData/log_G2_G3_G4_G5/pkl/'):
        df = pd.DataFrame()
        for filename in filenames:
            participant = filename[0:5]
            tor = filename[6:-4]
            print(participant, tor)
            AOI_gaze_count = GazeCountAOIs(participant, tor, window_size)
            # print(temp.to_numpy())
            AOI_gaze_count['Participant'] = participant
            AOI_gaze_count['Scenario'] = tor
            df = pd.concat([df, AOI_gaze_count], axis = 1)
    
    cols     = ['Participant',  'Scenario', 
                'LeftScreen' ,   
                'LeftMirror',  
                'Distractor',
                'CenterScreen', 
                'Cluster',
                'RightScreen',  
                'RightMirror']
    
    cols_num = ['LeftScreen'    + '_' + str(window_size) + 's',   
                'LeftMirror'    + '_' + str(window_size) + 's',  
                'Distractor'    + '_' + str(window_size) + 's',
                'CenterScreen'  + '_' + str(window_size) + 's', 
                'Cluster'       + '_' + str(window_size) + 's',
                'RightScreen'   + '_' + str(window_size) + 's',  
                'RightMirror'   + '_' + str(window_size) + 's' ]
    
    cols_Num = ['LeftScreen'    + '_' + str(window_size) + 's',   
                'LeftMirror'    + '_' + str(window_size) + 's',  
                'Distractor'    + '_' + str(window_size) + 's',
                'CenterScreen'  + '_' + str(window_size) + 's', 
                'Cluster'       + '_' + str(window_size) + 's',
                'RightScreen'   + '_' + str(window_size) + 's',  
                'RightMirror'   + '_' + str(window_size) + 's',
                'Others'        + '_' + str(window_size) + 's']
    
    cols_per = ['LeftScreen_Per'   + '_' + str(window_size) + 's',   
                'LeftMirror_Per'   + '_' + str(window_size) + 's',  
                'Distractor_Per'   + '_' + str(window_size) + 's',
                'CenterScreen_Per' + '_' + str(window_size) + 's', 
                'Cluster_Per'      + '_' + str(window_size) + 's', 
                'RightScreen_Per'  + '_' + str(window_size) + 's',  
                'RightMirror_Per'  + '_' + str(window_size) + 's', 
                'Others_Per'       + '_' + str(window_size) + 's']
    
    df = df.fillna(0).transpose().reset_index(drop=True)[cols]
    
    df.rename(columns = {'LeftScreen':'LeftScreen'    + '_' + str(window_size) + 's',   
                         'LeftMirror':'LeftMirror'    + '_' + str(window_size) + 's',  
                         'Distractor':'Distractor'    + '_' + str(window_size) + 's',
                         'CenterScreen':'CenterScreen'+ '_' + str(window_size) + 's', 
                         'Cluster':'Cluster'          + '_' + str(window_size) + 's',
                         'RightScreen':'RightScreen'  + '_' + str(window_size) + 's',  
                         'RightMirror':'RightMirror'  + '_' + str(window_size) + 's' }, inplace=True)
    
    df['Others' + '_' + str(window_size) + 's'] = window_size * 60 - df[cols_num].sum(axis = 1)
    
    for i in range(8):
        df[cols_per[i]] = df[cols_Num[i]]/df[cols_Num].sum(axis = 1)
        
    df['LeftSide'+ '_' + str(window_size) + 's']  = df[['LeftScreen_Per' + '_' + str(window_size) + 's', 
                                                        'LeftMirror_Per' + '_' + str(window_size) + 's', 
                                                        'Distractor_Per' + '_' + str(window_size) + 's']].sum(axis = 1)    
    df['RightSide'+ '_' + str(window_size) + 's'] = df[['RightScreen_Per'+ '_' + str(window_size) + 's', 
                                                        'RightMirror_Per'+ '_' + str(window_size) + 's']].sum(axis = 1)    
    
    return df

    # df = pd.DataFrame(df, columns = ['Participants', 'Scenario', 'EoR'])
    # temp = pd.DataFrame()
    # for participant in participants:
    #     for tor in TOR:
    #         AOI_gaze_count = GazeCountAOIs(participant, tor, window_size)

def PupilDiameterEyelidOpening(window_size):
    '''
    Process pupil diameter and eyelid opening given certain window size.

    Parameters
    ----------
    window_size : int

    Returns
    -------
    df : DataFrame

    '''
    for dirname, _, filenames in os.walk('SmartEyeData/log_G2_G3_G4_G5/pkl/'):
        df = []
        for filename in filenames:
            participant = filename[0:5]
            tor = filename[6:-4]
            print(participant, tor)
            pupil_diameter_eyelid_opening = PupilDiameterEyelidOpening_Mean_Std_Amplitude(participant, tor, window_size)
            pupil_diameter_eyelid_opening.insert(0, tor)
            pupil_diameter_eyelid_opening.insert(0, participant)
            df.append(pupil_diameter_eyelid_opening)
    
    cols = ['Participant',  'Scenario',
            'PupilDiamterMean'  + '_' + str(window_size) + 's', 
            'PupilDiamterStd'   + '_' + str(window_size) + 's', 
            'PupilDiamterAmp'   + '_' + str(window_size) + 's', 
            'EyelidOpeningMean' + '_' + str(window_size) + 's', 
            'EyelidOpeningStd'  + '_' + str(window_size) + 's', 
            'EyelidOpeningAmp'  + '_' + str(window_size) + 's']
    
    df = pd.DataFrame(df, columns=cols)
    
    return df

def dataProcessGrid(window_size):
    '''
    Count number of eye gazes in grids of 8x8.

    Parameters
    ----------
    window_size : int

    Returns
    -------
    df : array of arrays
    - [Participant: str, TOR: str, GazeCount: Dataframe]

    '''
    start = (15 - window_size) * 60
    end   = 15 * 60
    for dirname, _, filenames in os.walk('SmartEyeData/log_G2_G3_G4_G5/pkl/'):
        df = []
        for filename in filenames:
            participant = filename[0:5]
            tor = filename[6:-4]
            print(participant, tor)
            data = ReadFile(participant, tor)[start:end]            

            array2d = np.zeros((8,8))
        
            numData = window_size * 60

            for i in range(numData):
                x = int(data["ClosestWorldIntersection.objectPoint.x"].iloc[i] // 0.25)
                y = 7 - int(data["ClosestWorldIntersection.objectPoint.y"].iloc[i] // 0.20)
                if x < 8 and y < 8:
                    array2d[y][x] += 1


            df2d = pd.DataFrame(array2d, columns=['0', '1', '2', '3', '4', '5', '6', '7'])
            
            df.append([participant, tor, df2d])

    return df

def GazeEntropy(window_size):
    '''
    Calculate gaze entropy given centain window size.

    Parameters
    ----------
    window_size : int

    Returns
    -------
    df : DataFrame

    '''
    fileName  = 'GazeCountGrid_' + str(window_size) + 's.pkl' 
    open_file = open(fileName,'rb')
    data      = pickle.load(open_file)
    open_file.close()
    
    df = []
    
    for i in range(len(data)):
        temp     = data[i][2].values.reshape(1,64)
        temp_per = temp / temp.sum()
        entropy  = -np.nansum(np.multiply(np.log2(temp_per), temp_per)) 
        df.append([data[i][0], data[i][1], entropy])
        
  
    df = pd.DataFrame(df, columns=['Participant', 'Scenario', 'GazeEntropy_' + str(window_size) + 's'])
    
    return df

    
# temp = PercentageOfEyesOnRoad(participants[0], TOR[11], 15)  
# print(temp)  
              
# excel2pkl()

# df = EyesOnRoad()
# df.to_csv('EoR.csv')

# data = ReadFile(participants[0], TOR[0])

# for window_size in [3, 6, 9, 12, 15]:
#     temp = PercentageOfEyesOnRoad(window_size)
#     temp.to_csv('PercentageOfEyesOnRoad_' + str(window_size) + 's.csv')

# temp = PupilDiameterEyelidOpening_Mean_Std_Amplitude(participants[0], TOR[5], 3)
# print(temp)

# for window_size in [3, 6, 9, 12, 15]:
#     temp = PupilDiameterEyelidOpening(window_size)
#     temp.to_csv('PupilDiameterEyelidOpening_' + str(window_size) + 's.csv')

# for window_size in [3, 6, 9, 12, 15]:
#     df = BlinkCount(window_size)
#     df.to_csv('NumberOfBlinks_' + str(window_size) + 's.csv')

# for window_size in [3, 6, 9, 12, 15]:
#     df = FixationCount(window_size)
#     df.to_csv('NumberOfFixations_' + str(window_size) + 's.csv')

# for window_size in [3, 6, 9, 12, 15]:
#     df = dataProcessGrid(window_size)
#     fileName  = 'GazeCountGrid_' + str(window_size) + 's.pkl' 
#     open_file = open(fileName,'wb')
#     pickle.dump(df, open_file)
#     open_file.close()

# fileName  = 'GazeCountGrid_' + str(3) + 's.pkl' 
# open_file = open(fileName,'rb')
# df        = pickle.load(open_file)
# open_file.close()

# for window_size in [3, 6, 9, 12, 15]:
#     df = GazeEntropy(window_size)
#     df.to_csv('GazeEntropy_' + str(window_size) + '.csv')
    
# data = pd.read_csv('Dataset.csv', sep=',')

# temp = data['Time_Raw'] < 0
# print(temp.sum())

# print(np.nonzero(temp.values))

# [ 22,  29,  37,            85,      119,                172, 221,      268, 280, 294,      311,                364,      413, 421, 443, 550] 16
# [      29,  37,            85, 102, 119,                172, 221,      268,      294, 297, 311,      325,      364,      413, 421, 443, 550] 17
# [      29,  37,  70,       85, 102, 119,                172, 221,      268,      294, 297, 311,      325,      364, 412, 413, 421, 443, 550] 19
# [      29,  37,  70,  77,  85, 102, 119, 125, 133, 167, 172, 221, 229, 268,      294, 297, 311, 317, 325,      364, 412, 413, 421,      550] 24
# [      29,  37,  70,  77,  85, 102, 119, 125, 133, 167, 172, 221, 229, 268,      294, 297, 311, 317, 325, 359, 364, 412, 413, 421]           24

