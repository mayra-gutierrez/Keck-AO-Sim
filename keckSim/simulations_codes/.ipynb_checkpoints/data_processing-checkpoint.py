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

#%% Data from the gain study Xinetics
r0_list = [0.16]#[0.35,0.16,0.12]
mag = np.arange(6, 16, 2)
sys_ = 'KAO_R_band_SH_28x28_NGS_magnitude'
study = 'freq_study'
path = '/home/mcisse/keckAOSim/keckSim/data/'
gain_list = [0.5,0.5,0.4,0.3,0.3]#[0.3,0.4,0.5,0.6]
freq = [2000,1000,700,500,200]
M1_vect = [False,True]
NCPA_vect = [False,True]
start_idx = 50
'''
for i in range(len(mag)):
    r0 = r0_list[0]
    gain_str = f"{int(gain_list[i] * 10):02d}"
    name_gen = f'{path}{sys_}_{mag[i]}_M1_{M1_vect[0]}_NCPA_{NCPA_vect[0]}_KL_control_r0_{int(r0*100)}cm_gainCL_{gain_str}_AO_data_{study}.npy'
    
    AO_data = np.load(name_gen, allow_pickle=True).item()
    keys = sorted(AO_data.keys())
    
    ncols = math.ceil(np.sqrt(len(keys)))
    nrows = math.ceil(len(keys) / ncols)
    
    SR_mat  = np.stack([AO_data[k]['SR']       for k in keys], axis=0)
    res_mat = np.stack([AO_data[k]['residual'] for k in keys], axis=0)
    PSF_mat = np.stack([AO_data[k]['PSF_LE']   for k in keys], axis=0)
            
    aa = name_gen.split('.')
    bb = study
        
    SR_av = SR_mat[:,  start_idx:].mean(axis=1)
    res_av = res_mat[:, start_idx:].mean(axis=1)
    
    # --- Pre-create 4 figures/axes (single plot per figure)
    fig_res = plt.figure(); ax_res = plt.gca()
    ax_res.set_title(f'Star mag {mag[i]} r0 = {r0*100:.0f}cm: Residual vs Iterations'); ax_res.set_xlabel('Iterations'); ax_res.set_ylabel('Residual OPD nm RMS'); ax_res.grid(True)

    fig_sr  = plt.figure(); ax_sr  = plt.gca()
    ax_sr.set_title(f'Star mag {mag[i]} r0 = {r0*100:.0f}cm: SR vs Iterations'); ax_sr.set_xlabel('Iterations'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.ravel();fig.suptitle(f'Star mag {mag[i]} r0 = {r0*100:.0f}cm', fontsize=14)

    fig_avg = plt.figure(); ax_avg = plt.gca()
    ax_avg.set_title(f'Star mag {mag[i]} r0 = {r0*100:.0f}cm: Average residual OPD nm RMS'); ax_avg.set_xlabel('Loop frequency'); ax_avg.set_ylabel('Average residual nm RMS'); ax_avg.grid(True)
    
    fig_avg_sr = plt.figure(); ax_avg_sr = plt.gca()
    ax_avg_sr.set_title(f'Star mag {mag[i]} r0 = {r0*100:.0f}cm: Average SR'); ax_avg_sr.set_xlabel('Loop frequency'); ax_avg_sr.set_ylabel('Average SR'); ax_avg_sr.grid(True)

        
     # --- Single pass over datasets; plot into all three time-series figures
    for k in range(len(keys)):
        label = f'{freq[k]}Hz'
        ax_res.plot(res_mat[k], label=label)
        ax_sr.plot(SR_mat[k],   label=label)
        
        ax = axes[k]
        im = ax.imshow(PSF_mat[k]**0.2, cmap='viridis')
        ax.set_xlim([206, 306])
        ax.set_ylim([206, 306])
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    for j in range(k+1, len(axes)):
        axes[j].axis('off')
        
    ax_res.legend()
    ax_sr.legend()

    # --- 4th figure: averages vs frequency
    ax_avg_sr.plot(freq, SR_av,  'o-', label='Average SR')
    ax_avg.plot(freq, res_av, 'o-', label='Average residual OPD nm RMS')
    ax_avg.legend()
    ax_avg_sr.legend()

    plt.show(block=False)
    
    fig_res.savefig(f'{aa[0]}_residual.png', dpi=300, bbox_inches='tight')
    fig_sr.savefig(f'{aa[0]}_SR.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{aa[0]}PSF_LE.png', dpi=300, bbox_inches='tight')
    fig_avg.savefig(f'{aa[0]}_average_residual.png', dpi=300, bbox_inches='tight')
    fig_avg_sr.savefig(f'{aa[0]}_average_SR.png', dpi=300, bbox_inches='tight')
    plt.close('all')
    
'''      
#%%  
'''  
 
results = {r0: [] for r0 in r0_list}   
m1_i = 0
n1_i = 0        
for k in range(len(r0_list)):
    r0 = r0_list[k]
    
    for m in range(len(mag)):
        gain_str = f"{int(gain_list[m] * 10):02d}"
        #name_gen = f'{path}{sys_}_{m}_M1_{M1_vect[m1_i]}_NCPA_{NCPA_vect[n1_i]}_r0_{int(r0*100)}cm_AO_data_gain_study.npy'
        name_gen = f'{path}{sys_}_{mag[m]}_M1_{M1_vect[0]}_NCPA_{NCPA_vect[0]}_KL_control_r0_{int(r0*100)}cm_gainCL_{gain_str}_AO_data_{study}.npy'
        AO_data = np.load(name_gen, allow_pickle=True).item()
        keys = sorted(AO_data.keys())
        SR_mat  = np.stack([AO_data[k]['SR']       for k in keys], axis=0)

        SR_av = SR_mat[:,  start_idx:].mean(axis=1)
        
        best_idx = SR_av.argmax()
        best_freq = freq[best_idx]
        #best_gain = gain_list[best_idx]
        best_SR = SR_av[best_idx]
        
        results[r0].append((mag[m], best_SR, best_freq))
        
aa = name_gen.split('magnitude')        
filename = f'{aa[0]}magnitude_{mag[0]}_{mag[-1]}_M1_{M1_vect[0]}_NCPA_{NCPA_vect[0]}_KL_control_r0_{int(r0*100)}cm_AO_data_{study}_SR_All.npy'
np.save(filename, results)  

fig_name = filename.split('.npy')

plt.figure(figsize=(8, 5))
for r0, data in results.items():
    mags = [d[0] for d in data]
    SRs = [d[1] for d in data]
    plt.plot(mags, SRs, marker='o', label=f"r0 = {int(r0*100)} cm")

plt.xlabel("Magnitude")
plt.ylabel("Max SR")
plt.title("SR vs Magnitude per r0")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show(block=False)  
#plt.savefig(f'{fig_name[0]}.png')   

r0, data = next(iter(results.items()))
fig_sr  = plt.figure(); ax_sr  = plt.gca()
ax_sr.set_title(f'r0 = {r0*100:.0f}cm: SR vs Magnitude'); ax_sr.set_xlabel('Magnitude'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)
    
for i in range(len(data)):
    dd = data[i]
    m,sr,f = dd[0], dd[1], dd[2]
    label = f'{f}Hz'
    ax_sr.plot(m,sr,marker='o',label=label)
ax_sr.legend()
plt.tight_layout()
plt.show(block=False)
fig_sr.savefig(f'{fig_name[0]}.png', dpi=300, bbox_inches='tight')
'''
       
#%% write in a text file the results
'''
# Map file name flags to dataset labels
def dataset_label(filename):
    m1_flag = "M1_True" in filename
    ncpa_flag = "NCPA_True" in filename
    if not m1_flag and not ncpa_flag:
        return "Only atmosphere"
    elif m1_flag and not ncpa_flag:
        return "With M1 cophasing error"
    elif m1_flag and ncpa_flag:
        return "With M1 + NCPA"
    else:
        return "Unknown dataset"
        
summary_file = os.path.join(path, "summary_results.txt")
context_sentence = "\n\n HAKA (28x28) closed-loop frequency study for different star magnitudes for r0=16cm\n"#"Xinetics closed-loop gain study for different star magnitudes and different r0\n"
# Column widths
w_r0 = 8
w_mag = 16
w_gain = 16
w_sr = 16
# Group files by dataset type
datasets = {}
for file in os.listdir(path):
    if file.endswith("_SR_All.npy"):#("_SR_All_r0.npy"):
        label = dataset_label(file)
        datasets.setdefault(label, []).append(file)
        
# Open the output text file
with open(summary_file, "a") as f:
    # Write header
    f.write(context_sentence)
    for label, files in datasets.items():
        f.write(f"{label}\n")
        f.write(f"{'r0(cm)':<{w_r0}}{'Magnitude':<{w_mag}}{'Best_Frequency':<{w_gain}}{'Max_SR':<{w_sr}}\n")
        #f.write(f"{'r0(cm)':<{w_r0}}{'Magnitude':<{w_mag}}{'Best_Gain':<{w_gain}}{'Max_SR':<{w_sr}}\n")
        for file in files:  
            filepath = os.path.join(path, file)
            results = np.load(filepath, allow_pickle=True).item()
            for r0, data_list in results.items():
                r0_cm = int(r0 * 100)
                for mag, SR, gain in data_list:
                    #f.write(f"{r0_cm:<{w_r0}}{mag:<{w_mag}}{gain:<{w_gain}.2f}{SR:<{w_sr}.2f}\n")
                    f.write(f"{r0_cm:<{w_r0}}{mag:<{w_mag}}{gain:<{w_gain}}{SR:<{w_sr}.2f}\n")
        f.write("\n")  # blank line between datasets

print(f"Summary written to: {summary_file}")
'''

#%% plot 56x56 VS 28x28
'''
def dataset_label_SH(filename):
    HO = "56" in filename
    if not HO:
        return "SH 28x28"
    else:
        return "SH 56x56"

datasets = {} 
for file in os.listdir(path):
    if file.endswith("_SR_All.npy"):
        label = dataset_label_SH(file)
        datasets.setdefault(label, []).append(file)

fig_sr  = plt.figure(); ax_sr  = plt.gca()
ax_sr.set_title(f'r0 = {r0*100:.0f}cm: SR vs Magnitude'); ax_sr.set_xlabel('Magnitude'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)
comp_name = 'KAO_R_band_SH_28x28_VS_SH_56x56_M1_False_NCPA_False_r0_16cm_AO_data_freq_study_SR_All'
for label, files in datasets.items():    
    for file in files: 
        filepath = os.path.join(path, file)
        results = np.load(filepath, allow_pickle=True).item()
        r0, data = next(iter(results.items()))
        data = np.array(data)
        ax_sr.plot(data[:,0],data[:,1],'o-',label=label)
        
            
ax_sr.legend()
plt.tight_layout()
plt.show(block=False)
fig_sr.savefig(f'{comp_name}.png', dpi=300, bbox_inches='tight')  
'''
#%% Data from frequency study ZWFS Xinetics
'''
r0_list = [0.35,0.16,0.12]
mag = 8
gain_list = [0.4,0.5,0.6] # to be defined with the study above
freq_zwfs = np.arange(10,110,10)
NCPA_vect = [False,True]
start_idx = 50

path = '/home/mcisse/keckAOSim/keckSim/data/'
sys_ = f'{path}KAO_R_band_20x20_NGS_magnitude_{mag}_'
study = 'AO_data_frequency_study'

for i in range(2):#(len(r0_list)):
    r0 = r0_list[i]
    gain_str = f"{int(gain_list[i] * 10):02d}"
    name_gen = f'{sys_}gainCL{gain_str}_r0_{int(r0*100)}cm_NCPA_{NCPA_vect[0]}{study}.npy'
    AO_data_gains = np.load(name_gen, allow_pickle=True).item()
    keys = sorted(AO_data_gains.keys())
    
    ncols = math.ceil(np.sqrt(len(keys)))
    nrows = math.ceil(len(keys) / ncols)
    
    SR_mat  = np.stack([AO_data_gains[k]['SR']       for k in keys], axis=0)
    res_mat = np.stack([AO_data_gains[k]['residual'] for k in keys], axis=0)
    PSF_mat = np.stack([AO_data_gains[k]['PSF_LE']   for k in keys], axis=0)
            
    aa = name_gen.split('.')
    bb = study.split('data')
        
    SR_av = SR_mat[:,  start_idx:].mean(axis=1)
    res_av = res_mat[:, start_idx:].mean(axis=1)
    
    # --- Pre-create 4 figures/axes (single plot per figure)
    fig_res = plt.figure(); ax_res = plt.gca()
    ax_res.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: Residual vs Iterations'); ax_res.set_xlabel('Iterations'); ax_res.set_ylabel('Residual nm RMS'); ax_res.grid(True)

    fig_sr  = plt.figure(); ax_sr  = plt.gca()
    ax_sr.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: SR vs Iterations'); ax_sr.set_xlabel('Iterations'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.ravel();fig.suptitle(f'Star mag {mag} r0 = {r0*100:.0f}cm', fontsize=14)

    fig_avg = plt.figure(); ax_avg = plt.gca()
    ax_avg.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: Average res nm RMS'); ax_avg.set_xlabel('Frequency'); ax_avg.set_ylabel('Average residual nm RMS'); ax_avg.grid(True)
    
    fig_avg_sr = plt.figure(); ax_avg_sr = plt.gca()
    ax_avg_sr.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: Average SR'); ax_avg_sr.set_xlabel('Frequency'); ax_avg_sr.set_ylabel('Average SR'); ax_avg_sr.grid(True)

        
     # --- Single pass over datasets; plot into all three time-series figures
    for i in range(len(keys)):
        label = f'{freq_zwfs[i]}Hz'
        ax_res.plot(res_mat[i], label=label)
        ax_sr.plot(SR_mat[i],   label=label)
        
        ax = axes[i]
        im = ax.imshow(PSF_mat[i]**0.2, cmap='viridis')
        ax.set_xlim([206, 306])
        ax.set_ylim([206, 306])
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    ax_res.legend()
    ax_sr.legend()

    # --- 4th figure: averages vs frequency
    ax_avg_sr.plot(freq_zwfs, SR_av,  'o-', label='SR avg')
    ax_avg.plot(freq_zwfs, res_av, 'o-', label='Residual avg')
    ax_avg.legend()
    ax_avg_sr.legend()

    plt.show(block=False)
    
    fig_res.savefig(f'{aa[0]}_residual.png', dpi=300, bbox_inches='tight')
    fig_sr.savefig(f'{aa[0]}_SR.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{aa[0]}PSF_LE.png', dpi=300, bbox_inches='tight')
    fig_avg.savefig(f'{aa[0]}_average_residual.png', dpi=300, bbox_inches='tight')
    fig_avg_sr.savefig(f'{aa[0]}_average_SR.png', dpi=300, bbox_inches='tight')

'''





















