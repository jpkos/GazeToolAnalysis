# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:07:15 2021

@author: jankos
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%%
def load_data(exp):
    r = pd.read_csv(f'model_train_test_data/test_results/r_{exp}.txt', header=None)
    px = pd.read_csv(f'model_train_test_data/test_results/px_{exp}.txt', header=None)
    p = pd.read_csv(f'model_train_test_data/test_results/p_{exp}.txt', header=None)
    f1 = pd.read_csv(f'model_train_test_data/test_results/f1_{exp}.txt', header=None)
    return r, px, p, f1

def calc_maxes(px, p, r, f1):
    bep = np.round(px.iloc[np.argmax(f1)].iloc[0],2)
    maxp = np.round(p.iloc[np.argmax(f1)],2)
    maxr = np.round(r.iloc[np.argmax(f1)],2)
    maxf1 = np.round(np.max(f1),2)    
    return bep, maxp, maxr, maxf1
#%%F1-confidence-precision-recall plot for test data
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(ncols=3, figsize=(14,3), sharey=True)
for i in range(3):
    if i<2:
        r, px, p, f1 = load_data(f'exp{i+1}')
    elif i==2:
        r, px, p, f1 = load_data(f'exp{i+1}_OR')
    r = r.mean()
    p = p.mean()
    f1 = f1.mean()
    ax2 = ax[i].twinx()
    ax[i].plot(p, r, '--')
    ax[i].plot(p, f1, '-.')
    bep, maxp, maxr, maxf1 = calc_maxes(px, p, r, f1)
    print('break even point: ', bep)
    ax2.plot(p, px, 'v', markevery=51, color='black')
    if i==0:
        ax[i].set(ylabel='Recall/F1')
        ax[i].legend(['Recall', 'F1'], bbox_to_anchor=(0,1), loc='lower left')
    else:
        ax[i].legend().remove()
    if i==1:
        ax2.legend(['Confidence'], bbox_to_anchor=(-0.1,1), loc='lower left')
    if i==2:
        ax2.set(ylabel='Confidence threshold')
    ax[i].set(xlabel='Precision', xlim=(0,1), ylim=(0,1))
    ax2.set(ylim=(0,1))
    if i<2:
        ax2.set_yticklabels([])
    ax[i].axhline(bep, color='red')
    ax[i].axvline(maxp, color='red')
    #If exp 3, plot x-y-lines for OR+Sim case
    if i==2:
        r2, px2, p2, f12 = load_data(f'exp{i+1}_ORandSim')
        r2 = r2.mean()
        p2 = p2.mean()
        f12 = f12.mean()
        bep2, maxp2, maxr2, maxf12 = calc_maxes(px2, p2, r2, f12)
        ax[i].axhline(bep2, color='green')
        ax[i].axvline(maxp2, color='green')
    plt.annotate(f'Break even point: {bep}', (0,-0.3), color='black', fontsize=10, annotation_clip=False)
    plt.annotate(f'Precision: {maxp}', (0,-0.4), color='black', fontsize=10, annotation_clip=False)
    plt.annotate(f'Recall: {maxr}', (0,-0.5), color='black', fontsize=10, annotation_clip=False)
    plt.annotate(f'F1: {maxf1}', (0,-0.6), color='black', fontsize=10, annotation_clip=False)

#%%Validation results during training
hdrs = ['epoch', 'mem', 'box', 'objectness', 'classification', 'total', 'targets', 'img_size',
        'precision', 'recall', 'mAP05', 'mAP0595', 'valBox', 'valObj', 'valClass']
exp1results = pd.read_csv('model_train_test_data/training_results/exp1_results.txt', names=hdrs,delim_whitespace=True)
exp21results = pd.read_csv('model_train_test_data/training_results/exp2-1_results.txt', names=hdrs, delim_whitespace=True)
exp3_OR = pd.read_csv('model_train_test_data/training_results/exp3-ORonly-results.csv')[['     metrics/mAP_0.5']]
exp3_ORSim = pd.read_csv('model_train_test_data/training_results/exp3-ORandSim-results.csv')[['     metrics/mAP_0.5']]

comb = exp1results[['mAP05']]
comb.columns = ['Exp 1']
comb['Exp 2'] = exp21results['mAP05']
comb['Exp 3 (OR only)'] = exp3_OR
comb['Exp 3 (OR+Sim)'] = exp3_ORSim
comb['Epoch'] = np.arange(1,len(comb)+1)

styles = ['-.', '--', '-', ':']
colors = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
 (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
 (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
 (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)]

a = comb.plot(x='Epoch', figsize=(5,3), ylabel='mAP@0.5', xlabel='Epoch',
              style=styles, xlim=(0,160), color=colors)
plt.subplots_adjust(
top=0.946,
bottom=0.196,
left=0.112,
right=0.973,
hspace=0.2,
wspace=0.2)
plt.yticks(np.linspace(0,1,6), np.round(np.linspace(0,1,6),1))
plt.tight_layout()
#%%
