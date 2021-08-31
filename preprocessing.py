# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 08:56:39 2021

@author: jankos
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

from utils.preprocess import load_detections, load_segments, load_gazedata, apply_filter
from utils.preprocess import load_interruptions, map_interruptions, between
from utils.preprocess import label_tools, match_frames, label_changes
#%%
FRAME_SIZE = (480,640)
#%%Load data
case_n = 1
detection_file = f'case_study_data/input_data/pilot2-case{case_n}-microscope-view.txt'
segment_file = f'case_study_data/input_data/pilot2-case{case_n}-et-actions.csv'
gaze_file = f'case_study_data/input_data/gaze_pilot2_case{case_n}.csv'
interruption_file = f'case_study_data/input_data/pilot2-case{case_n}-interruptions.csv'

tools=load_detections(detection_file, frame_shape=FRAME_SIZE)
segments = load_segments(segment_file)
gaze = load_gazedata(gaze_file, frame_shape=FRAME_SIZE)
interruptions = load_interruptions(interruption_file)
#%%Fill frames with missing detection with nan
for i, tool in enumerate(tools):
    newindex = np.arange(tool['frame'].min(), tool['frame'].max(), 1)
    tool = tool.set_index('frame', drop=False)
    tool_newindex = tool.reindex(index=newindex)
    missing_originally = tool_newindex['x'].isna().reset_index()
    tool_newindex['frame'] = tool_newindex['frame'].interpolate(method='linear')
    tool_newindex['class'] = tool_newindex['class'].interpolate(method='linear')
    tool_newindex = tool_newindex.reset_index(drop=True)
    tool_newindex = tool_newindex.assign(missing_originally=missing_originally['x'])
    tools[i] = tool_newindex
#%%Interpolate if < 5 consecutive frames missing
for i, tool in enumerate(tools):
    tool = match_frames(tool, gaze)
    tool = tool.interpolate('linear', limit=5)
    tool['missing_after_interpolation'] = tool['x_tool'].isna().astype(int)
    tools[i] = tool
#%%Map actions
for i, tool in enumerate(tools):
    actions = pd.Series(['NA' for n in range(len(tool))])
    action_changes = pd.Series(np.zeros(len(tool)))
    for j, segment in segments.iterrows():
        actions.loc[tool['frame'].between(segment['Frame'], segment['stop'])] = segment['Annotation code']
        action_changes.loc[tool['frame'].between(segment['Frame'], segment['stop'])] = j
    tool['action'] = actions
    tool['action_change'] = action_changes #When action changes e.g. from D to EVS
#%%Map interruptions (participant moves away from microscope)
for i, tool in enumerate(tools):
    tool2 = map_interruptions(tool, interruptions)
    tools[i] = tool2[between(tool2['interruption'])]
#%%Label tools with correct names, e.g. 1=microforceps, etc.
for i, tool in enumerate(tools):
        tools[i] = label_tools(tool)
#%%Label actions that had too many missing frames after interpolation
for i, tool in enumerate(tools):
    tool = tool.drop_duplicates('frame') #if same tool had 2 different detections (only 1 of each in this dataset)
    tool['frame2'] = tool['frame']
    tool.loc[tool['missing_after_interpolation'] == 1, 'frame2'] = np.nan
    #code continuous action segments
    tool['action_code'] = label_changes(tool['action'].values, tool['frame2'].values, frame_lim=1) #interpolated gaps smaller than 5
    tools[i] = tool
#%%Label magnification changes
mag_changes = [0, 10713, 32300, 39034] if case_n == 1 else [0, 11000, 20800, 23744, 25050, 26000, 34960] 
for i, tool in enumerate(tools):
    tool['mag_chg'] = pd.cut(tool['frame'], mag_changes)
    tools[i] = tool
    
#%%Combine data for both tools and save
combined = pd.concat(tools)
#pick important column
combined = combined[['class', 'frame', 'x_tool', 'y_tool', 'x_gaze', 'y_gaze',
                     'missing_after_interpolation', 'missing_originally',
                     'action','action_change', 'mag_chg', 'action_code']]
combined.to_csv(f'case_study_data/processed_data/pilot-2-case-{case_n}-tool-gaze-locations.csv')
