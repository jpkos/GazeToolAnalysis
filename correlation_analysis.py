# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 09:15:48 2021

@author: jankos
"""

import pandas as pd 
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from colorspacious import cspace_converter
#%%
sessions = [1,2]
tools = ['Microforceps', 'Microscissors']
dfs = []
new_cols = []
new_order = ['PL','Mean V','SD V',
 'Mean A', 'SD A',
 'Mean J', 'SD J',
 'Mean C', 'SD C',
 'Mean TD', 'SD TD']
sessions_tools = list(itertools.product(sessions, tools))
for st in sessions_tools:
    df_correlation = pd.read_csv(r"case_study_data/input_data/correlations session {} tool {} .csv".format(st[0], st[1]),
                 index_col=[0])
    df_critical = pd.read_csv(r"case_study_data/input_data/crit vals session {} tool {} .csv".format(st[0], st[1]),
                 index_col=[0])
    df_critical = df_critical<0.05
    df_correlation = df_correlation[df_critical]
    new_cols = []
    #Rename for plotting
    for name in df_correlation.columns:
        split = name.split('_')
        if split[0] == 'tip':
            var = split[2].capitalize() if split[2] == 'mean' else split[2].upper()
            new_cols.append(var + ' ' + (split[0][0] + split[1][0]).upper())
        elif split[0] == 'path':
            new_cols.append((split[0][0] + split[1][0]).upper())            
        else:
            var = split[1].capitalize() if split[1] == 'mean' else split[1].upper()
            new_cols.append(var + ' ' +split[0][0].upper())
    
    df_correlation.columns = new_cols
    df_correlation.index = new_cols
    df_correlation = df_correlation[new_order]
    df_correlation = df_correlation.reindex(new_order)
    dfs.append(df_correlation)

#%%
fig, ax = plt.subplots(figsize=(5,5),
                       nrows=2, ncols=2,
                       sharey=True, sharex=True)
axes = ax.flatten()
cbar_ax = fig.add_axes([.88, .3, .03, .4])
for i, df in enumerate(dfs):
    g = sns.heatmap(df, ax=axes[i], vmin=-1, vmax=1, cbar=i==0, xticklabels=True, yticklabels=True,
                cbar_ax=None if i else cbar_ax, cmap='PiYG', mask=df.isnull(), linecolor='black',
                linewidths=0.5, square=True)

    # axes[i].set_facecolor('lightgray')
    axes[i].set_title(f'Session {sessions_tools[i][0]} \n {sessions_tools[i][1]}')
