#%%
import numpy as np
import sys
import math
import matplotlib.pyplot as plt 

import os

os.chdir("/home/mcisse/keckAOSim/keckSim/")
sys.path.insert(0, os.getcwd())
from simulations_codes.KAO_parameter_file import initializeParameterFile
from simulations_codes.initialize_AO import initialize_AO_hardware

plt.ion()

param = initializeParameterFile()
path = param['pathOutput']
ZWFS_param = {'activate': False, 'max_CLiter':5,'gain':1,'n_average':5,'avg_time':4,'freq':4,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

'''
#%% Alpha Studies
mag = 8
gainCL = 0.3
alpha_list = [1e13,1e14,1e15,1e16]
results = {}
name_study = f'{path}{param["name"]}_NGS_magnitude_{mag}_r0_{int(param["r0"]*100)}cm_AO_data_alpha_study.npy'
name_plot = name_study.split('.')

results = np.load(name_study, allow_pickle=True).item()
keys = sorted(results.keys())

SR_mat  = np.stack([results[k]['SR']       for k in keys], axis=0)
res_mat = np.stack([results[k]['residual'] for k in keys], axis=0)
sr_mean = np.mean(SR_mat[:,-100:],axis=1)
res_mean = np.mean(res_mat[:,-100:],axis=1)

# --- Pre-create figures/axes (single plot per figure)
fig_res = plt.figure(); ax_res = plt.gca()
ax_res.set_title(f'Star mag {mag} r0 = {int(param["r0"]*100)}cm: Residual vs Iterations'); ax_res.set_xlabel('Iterations'); ax_res.set_ylabel('Residual OPD nm RMS'); ax_res.grid(True)

fig_sr  = plt.figure(); ax_sr  = plt.gca()
ax_sr.set_title(f'Star mag {mag} r0 = {int(param["r0"]*100)}cm: SR vs Iterations'); ax_sr.set_xlabel('Iterations'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)
    
for k in keys:
    label = f'alpha {alpha_list[k]:.0e}, SR = {np.round(sr_mean[k]*100,1)}@K'
    ax_res.plot(res_mat[k], label=label)
    ax_sr.plot(SR_mat[k],   label=label)
                
ax_res.legend()
ax_sr.legend()       
fig_res.savefig(f'{name_plot[0]}_residual.png', dpi=300, bbox_inches='tight')
fig_sr.savefig(f'{name_plot[0]}_SR.png', dpi=300, bbox_inches='tight')
'''
#%% Freq/Gain study plots for differents magnitude
#'''    
mag_list = np.arange(6,16,2)
frequency = [2000,1000,700,500,200]
gainCL = 0.3
idx_start = 500
best_results = {mag: [] for mag in mag_list}  

for mag in mag_list:

    name_study = f'{path}{param["name"]}_NGS_magnitude_{mag}_r0_{int(param["r0"]*100)}cm_AO_data_freq_study.npy' 
    study = name_study.split('data')  
    
    if 'freq' in study[-1]:
        info = 'Frequency'
    elif 'gain' in study[-1]:
        info = 'Gain'

    name_plot = name_study.split('.')
    
    results = np.load(name_study, allow_pickle=True).item()
    keys = results.keys()
    
    SR_mat  = np.stack([results[k]['SR']       for k in keys], axis=0)
    res_mat = np.stack([results[k]['residual'] for k in keys], axis=0)
    
    sr_mean = np.mean(SR_mat[:,-idx_start:],axis=1)
    res_mean = np.mean(res_mat[:,-idx_start:],axis=1)  
    
    best_idx = sr_mean.argmax()
    thr = sr_mean[best_idx]-0.02
    mask = (sr_mean<sr_mean[best_idx]) & (sr_mean>=thr)
    if mask.any():
        best_idx = np.argmax(sr_mean*mask)

    best_freq = frequency[best_idx]
    best_SR = sr_mean[best_idx]*100
    best_results[mag]={'SR': best_SR,'Frequency': best_freq}
    
    #plots
    # --- Pre-create figures/axes (single plot per figure)
    fig_res = plt.figure(); ax_res = plt.gca()
    ax_res.set_title(f'Star mag {mag} r0 = {int(param["r0"]*100)}cm: Residual vs Iterations');      ax_res.set_xlabel('Iterations'); ax_res.set_ylabel('Residual OPD nm RMS'); ax_res.grid(True)

    fig_sr  = plt.figure(); ax_sr  = plt.gca()
    ax_sr.set_title(f'Star mag {mag} r0 = {int(param["r0"]*100)}cm: SR vs Iterations'); ax_sr.set_xlabel('Iterations'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)
    
    for i,k in enumerate(keys):
        label = f'F = {frequency[i]}Hz, SR = {np.round(sr_mean[i]*100,1)}@K'
        ax_res.plot(res_mat[i], label=label)
        ax_sr.plot(SR_mat[i],   label=label)
                
    ax_res.legend()
    ax_sr.legend()       
    fig_res.savefig(f'{name_plot[0]}_residual.png', dpi=300, bbox_inches='tight')
    fig_sr.savefig(f'{name_plot[0]}_SR.png', dpi=300, bbox_inches='tight')
        

fig_all = plt.figure(); ax_all  = plt.gca()
ax_all.set_title(f'{info} study, r0 = {int(param["r0"]*100)}cm: SR vs Magnitude'); ax_all.set_xlabel('Magnitude'); ax_all.set_ylabel('SR@K'); ax_all.grid(True)

for mag in best_results.keys():
    label = f'F = {best_results[mag]["Frequency"]}Hz'
    ax_all.plot(mag,best_results[mag]["SR"], marker='o', markersize=8, label=label)

ax_all.legend()
fig_all.savefig(f'{name_plot[0]}_SR_All.png', dpi=300, bbox_inches='tight')     
name_best_results = name_study.split('magnitude') 
filename = f'{name_best_results[0]}magnitude_{mag_list[0]}_{mag_list[-1]}_r0_{int(param["r0"]*100)}cm_AO_data_{info}_study_SR_All.npy'
np.save(filename, best_results)          
        
#'''     
        
        
        
        
        
         
