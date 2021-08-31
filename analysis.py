# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 08:57:14 2021

@author: jankos
"""
import pandas as pd
import numpy as np
import time
from utils.preprocess import calc_metrics
#%%
V_LIM = 50 #velocity/curvature threshold
M_LIMIT = 0.2  #percentage of missing frames allowed in action
#%%Load data
case_n = 1
df = pd.read_csv(f'case_study_data/processed_data/pilot-2-case-{case_n}-tool-gaze-locations.csv')
#%%Check which actions had >20% missing
m = df.groupby(['mag_chg', 'class', 'action','action_change'])['missing_after_interpolation'].apply(lambda x: x.sum()/len(x)).dropna()
mg = m.reset_index().groupby('action_change').apply(lambda x:x['missing_after_interpolation'].max()<M_LIMIT).reset_index()
nmg = mg[~mg[0]]
mg = mg[mg[0]] #actions that had less than m_limit % of missing values

#%%Calculate metrics frame-by-frame, for each continuous segment and tool
grouped = df.groupby(['action', 'class', 'action_code'])
groups = []
removed = 0
for name, group in grouped: #This is super slow, try something better
    group['dx'] = group['x_tool'].diff() #component x of velocity vector
    group['dx2'] = group['dx'].diff() #component x of acceleration vector
    group['dy'] = group['y_tool'].diff()#component y of velcity vector
    group['dy2'] = group['dy'].diff() #component y of acceleration vector
    group['v'] = np.sqrt(group['dx']**2 + group['dy']**2) #displacement per frame, i.e velocity
    group['a'] = group['v'].diff().abs()
    group['j'] = group['a'].diff().abs()
    group['c'] = np.abs(np.cross(group[['dx2', 'dy2']].values,
                                 group[['dx', 'dy']].values))/group['v']**3
    remove = (group['v']>V_LIM) | (group['c']>V_LIM)
    removed += len(group[(remove)]) #keep count of how many removed
    group.loc[(remove), ['v', 'a', 'j', 'c', 'dx', 'dy', 'dx2', 'dy2']] = np.nan  #set high values to nan
    groups.append(group)
#%%concatenate and drop actions with too many missing frames
df = pd.concat(groups)
df = df.sort_index()
df = df[df['action_change'].isin(mg['action_change'])] #drop actions with too much missing
df = df.reset_index(drop=True)
#%% Path length
s = df.groupby(['mag_chg', 'class', 'action', 'action_change'])['v'].apply(calc_metrics, prefix='v', operations=['sum'])
s = s.reset_index(level=-1, drop=True)
s['v_sum_norm'] = s.groupby(['class', 'mag_chg'])['v_sum'].apply(lambda x: x/x.max())
#%%Time
t = df.groupby(['mag_chg', 'class', 'action', 'action_change'])['frame'].apply(lambda x: (x.max() - x.min())+1).dropna()
s = s.rename(columns={'v_sum':'path_length', 'v_sum_norm':'path_length_norm'})
#%%Calculate other metrics
for metric in ['v', 'a', 'c', 'j']:
    c = df.groupby(['mag_chg', 'class', 'action', 'action_change'])[metric].\
        apply(calc_metrics, prefix=metric) #mean and sd by default
    c = c.reset_index(level=-1, drop=True)
    cols = c.columns
    c_norm = c.groupby(['class', 'mag_chg'])[cols].apply(lambda x: x/x[cols[0]].max())
    print(cols[0])
    c_norm.columns  = [f'{col}_norm' for col in c_norm.columns]
    c[f'{metric}_var'] = c[cols[1]]/c[cols[0]]
    s = pd.concat([s, c, c_norm], axis=1) 

#%% Tip distance (needs to be calculated separately)
df_pivoted = pd.pivot_table(df, values=['x_tool', 'y_tool'], index='frame', columns='class').reset_index()
df_pivoted.columns = [''.join(col).strip() for col in df_pivoted.columns]
df_pivoted = pd.merge(df_pivoted, df[['frame', 'mag_chg', 'action', 'action_change']], on='frame', how='right')
df_pivoted = df_pivoted.drop_duplicates('frame')
df_pivoted['tip_distance'] = np.sqrt((df_pivoted['x_toolmicroforceps'] - df_pivoted['x_toolmicroscissors'])**2\
                                     + (df_pivoted['y_toolmicroforceps'] - df_pivoted['y_toolmicroscissors'])**2)
    
td = df_pivoted.groupby(['mag_chg', 'action', 'action_change'])['tip_distance'].apply(calc_metrics, prefix='tip_distance')
td = td.reset_index(level=-1, drop=True)
td_norm = td.groupby(['mag_chg'])[['tip_distance_mean', 'tip_distance_sd']].apply(lambda x: x/x['tip_distance_mean'].max())  
td_norm.columns  = [f'{col}_norm' for col in td_norm.columns]
#Merge tool tip distance and other metrics
s = pd.merge(s, td, right_index=True, left_index=True)
s = pd.merge(s, td_norm, right_index=True, left_index=True)
#%% Add % of missing frames
s = pd.merge(s, m, right_index=True, left_index=True)
#%% merge time
s = pd.merge(s, t, right_index=True, left_index=True)
#%%
s = s.reset_index()
s.loc[s['action'] == 'C', 'action'] = 'D'
s = s.rename(columns={'frame':'duration_f'})
# s = s.drop(labels=['level_0', 'index'], axis=1)
s.to_csv(f'case_study_data/output_data/pilot-2-case-{case_n}-tool-kinematics.csv', index=None)
#%%Gaze-tool distance
df['gaze_tool_d'] = np.sqrt((df['x_gaze'] - df['x_tool'])**2 + (df['y_gaze'] - df['y_tool'])**2)
df.loc[df['x_gaze'].diff()==0, 'gaze_tool_d'] = np.nan
df['gaze_missing'] = df['gaze_tool_d'].isna().astype(int)
g_m = df.groupby(['class', 'action_change'])['gaze_missing'].apply(lambda x: x.sum()/len(x))
g_m = g_m.reset_index()
g_mOK = g_m[g_m['gaze_missing']<M_LIMIT].drop_duplicates('action_change')
#%%
g_m_map = dict(zip(g_mOK['action_change'], g_mOK['gaze_missing']))
g = df.groupby(['mag_chg', 'class', 'action', 'action_change'])['gaze_tool_d'].\
apply(calc_metrics, prefix='gaze_tool').reset_index(level=-1,drop=True).reset_index()
g = g[g['action_change'].isin(g_mOK['action_change'])]
gnorm = g.groupby(['class', 'mag_chg'])[['gaze_tool_mean', 'gaze_tool_sd']].apply(lambda x: x/x['gaze_tool_mean'].max())
g[['gaze_tool_mean_norm', 'gaze_tool_sd_norm']] = gnorm
g['gaze_missing'] = g['action_change'].map(g_m_map)
#%%
g.to_csv(f'case_study_data/output_data/pilot-2-case-{case_n}-gaze-tool.csv')

