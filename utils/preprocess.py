# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:10:19 2021

@author: jankos
"""
import numpy as np
import pandas as pd
from scipy import signal, stats, interpolate

def load_detections(path, frame_shape):
    data = np.loadtxt(path)
    df = pd.DataFrame(data, columns=['frame', 'class',
                                       'x', 'y','width', 'height', 'conf'])   
    df[['x', 'width']] = df[['x','width']]*frame_shape[1]
    df[['y', 'height']] = df[['y','height']]*frame_shape[0]
    
    return [group.drop_duplicates(subset='frame', keep='last').\
            reset_index(drop=True) for i, group in df.groupby('class')]

def load_segments(path, time_shift=None):
    segments = pd.read_csv(path)
    if time_shift != None:
        segments['Time'] = segments['Time'] - time_shift
    segments.loc[:,'Frame'] = segments['Time']*segments['FPS']
    stops = segments[segments['startstop'] == 'STOP']
    segments = segments[segments['startstop'] == 'START'].reset_index(drop=True)
    segments = segments.assign(stop=stops['Frame'].values)
    return segments

def load_gazedata(path, frame_shape):
    df = pd.read_csv(path, sep=';')
    if df.shape[1] == 1:
        df = pd.read_csv(path)
    df['Frame Number'] = df['Frame Number '].astype(int)
    df[[' Point Of Regard X ', ' Point Of Regard Y ']] \
        = df[[' Point Of Regard X ', ' Point Of Regard Y ']].astype(np.float)
    df.loc[(df[' Point Of Regard X ']<0)|(df[' Point Of Regard X ']>1), ' Point Of Regard X '] = np.nan
    df.loc[(df[' Point Of Regard Y ']<0)|(df[' Point Of Regard Y ']>1), ' Point Of Regard Y '] = np.nan
    
    df[' Point Of Regard X '] = df[' Point Of Regard X ']*frame_shape[1]
    df[' Point Of Regard Y '] = df[' Point Of Regard Y ']*frame_shape[0]
    df = df.rename(columns={' Point Of Regard X ':'x',
                            ' Point Of Regard Y ':'y'})
    df = df.rename(columns={'Frame Number ':'frame'})
    df['frame'] = df['frame'] - df['frame'].iloc[0]+1
    return df

def load_interruptions(path):
    return pd.read_csv(path)

def apply_filter(data, f_s, f_c, order):
    #Low pass filter, cutoff in Hz
    sos = signal.butter(order, f_c, fs=f_s, output='sos')
    return signal.sosfiltfilt(sos,data), sos

def map_interruptions(data, interruptions):

    d = pd.cut(data['frame'], bins=interruptions['frame'], labels=np.arange(1,len(interruptions)))
    d = d.astype(float)
    d = d.fillna(0)
    d = d.astype(int)
    return data.assign(interruption=d)

def between(interruptions, even=True):
    if even:
        return interruptions%2==0
    else:
        return interruptions%2==1

def label_tools(data):
    tool_map = {0:'pierce',1:'microforceps',2:'needleholder',
                3:'needle', 4:'microscissors'}
    data.loc[:,'class'] = data['class'].map(tool_map)
    data.loc[:,'class'] = data['class'].fillna(method='pad')
    return data

def reindex_frames(data):
    newindex = np.arange(data['frame'].min(), data['frame'].max(), 1)
    data = data.set_index('frame', drop=False)
    data_newindex = data.reindex(index=newindex)   
    return data_newindex

def interpolate_frames(data):
    data_newindex = reindex_frames(data)
    data_newindex = data_newindex.interpolate(method='linear')
    data_newindex = data_newindex.reset_index(drop=True)
    return data_newindex
    
def match_frames(tool, gaze):
    merged = pd.merge(tool, gaze, left_on=['frame'],
                right_on=['frame'], suffixes=['_tool', '_gaze'])
    return merged

def label_changes(labels, frames, frame_lim=1):
    codes = [0]
    i = 0
    prev_lbl = labels[0]
    prev_frame = frames[0]
    for j, lf in enumerate(zip(labels[1:], frames[1:])):
        lbl = lf[0]
        frame = lf[1]
        if np.isnan(frame):
            if prev_lbl != lbl:
                i+=1
                prev_lbl = lbl
            codes.append(i)
            continue
        if prev_lbl != lbl or (frame-prev_frame)>frame_lim:
            i += 1
        prev_lbl = lbl
        prev_frame = frame
        codes.append(i)
            
    return codes

def calc_metrics(x, prefix='value', operations=['mean', 'sd']):
    ops = {'mean': np.mean, 'sd': lambda x: np.std(x, ddof=1), 'sum': np.sum}
    X = {f'{prefix}_{op}':[ops[op](x)] for op in operations}
    return pd.DataFrame(X)