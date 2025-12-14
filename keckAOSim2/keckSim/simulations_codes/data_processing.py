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

r0_list = [0.35,0.16,0.12]
mag = np.arange(6, 16, 2)
sys_ = 'KAO_R_band_SH_20x20_NGS_magnitude'
study = 'gain_study'
path = '/home/mcisse/keckAOSim/keckSim/data/'
gain_list = [0.3,0.4,0.5,0.6] #[0.5,0.5,0.4,0.3,0.3]#
freq = [2000,1000,700,500,200]
M1_vect = [False,True]
NCPA_vect = [False,True]
start_idx = 50
'''
for i in range(len(mag)):
    r0 = r0_list[1]
    #gain_str = f"{int(gain_list[i] * 10):02d}"
    #name_gen = f'{path}{sys_}_{mag[i]}_M1_{M1_vect[0]}_NCPA_{NCPA_vect[0]}_r0_{int(r0*100)}cm_AO_data_{study}.npy'
    name_gen = f'{path}{sys_}_{mag[i]}_M1_{M1_vect[0]}_NCPA_{NCPA_vect[0]}_r0_{int(r0*100)}cm_alpha_20_AO_data_{study}.npy'
    
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
    ax_avg.set_title(f'Star mag {mag[i]} r0 = {r0*100:.0f}cm: Average residual OPD nm RMS'); ax_avg.set_xlabel('Loop gain'); ax_avg.set_ylabel('Average residual nm RMS'); ax_avg.grid(True)
    
    fig_avg_sr = plt.figure(); ax_avg_sr = plt.gca()
    ax_avg_sr.set_title(f'Star mag {mag[i]} r0 = {r0*100:.0f}cm: Average SR'); ax_avg_sr.set_xlabel('Loop gain'); ax_avg_sr.set_ylabel('Average SR'); ax_avg_sr.grid(True)

        
    # --- Single pass over datasets; plot into all three time-series figures
    for k in range(len(keys)):
        label = f'loop gain {gain_list[k]}'
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
    ax_avg_sr.plot(gain_list, SR_av,  'o-', label='Average SR')
    ax_avg.plot(gain_list, res_av, 'o-', label='Average residual OPD nm RMS')
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
r0_list = [0.16] 
results = {r0: [] for r0 in r0_list}   
m1_i = 0
n1_i = 0        
for k in range(len(r0_list)):
    r0 = r0_list[k]
    
    for m in range(len(mag)):
        #gain_str = f"{int(gain_list[m] * 10):02d}"
        #name_gen = f'{path}{sys_}_{mag[m]}_M1_{M1_vect[m1_i]}_NCPA_{NCPA_vect[n1_i]}_r0_{int(r0*100)}cm_AO_data_gain_study.npy'
        #name_gen = f'{path}{sys_}_{mag[m]}_M1_{M1_vect[0]}_NCPA_{NCPA_vect[0]}_keck_zonal_r0_{int(r0*100)}cm_gainCL_{gain_str}_AO_data_{study}.npy'
        name_gen = f'{path}{sys_}_{mag[m]}_M1_{M1_vect[0]}_NCPA_{NCPA_vect[0]}_r0_{int(r0*100)}cm_alpha_20_AO_data_{study}.npy'
        AO_data = np.load(name_gen, allow_pickle=True).item()
        keys = sorted(AO_data.keys())
        SR_mat  = np.stack([AO_data[k]['SR']       for k in keys], axis=0)

        SR_av = SR_mat[:,  start_idx:].mean(axis=1)
        
        best_idx = SR_av.argmax()
        #best_freq = freq[best_idx]
        best_gain = gain_list[best_idx]
        best_SR = SR_av[best_idx]
        
        results[r0].append((mag[m], best_SR, best_gain))
        
aa = name_gen.split('magnitude')        
filename = f'{aa[0]}magnitude_{mag[0]}_{mag[-1]}_M1_{M1_vect[0]}_NCPA_{NCPA_vect[0]}_r0_{int(r0*100)}cm_alpha_20_AO_data_{study}_SR_All.npy'
#np.save(filename, results)  

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
plt.savefig(f'{fig_name[0]}.png')   

r0, data = next(iter(results.items()))
fig_sr  = plt.figure(); ax_sr  = plt.gca()
ax_sr.set_title(f'r0 = {r0*100:.0f}cm: SR vs Magnitude'); ax_sr.set_xlabel('Magnitude'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)
    
for i in range(len(data)):
    dd = data[i]
    m,sr,f = dd[0], dd[1], dd[2]
    label = f'{f} Loop gain'
    ax_sr.plot(m,sr,marker='o',label=label)
ax_sr.legend()
plt.tight_layout()
plt.show(block=False)
#fig_sr.savefig(f'{fig_name[0]}.png', dpi=300, bbox_inches='tight')
'''
#%% NCPA & M1 data analysis
'''
r0 = 16
mag = [8]
sys_ = 'KAO_R_band_SH_20x20_NGS_magnitude'
study = 'M1_NCPA_study'
path = '/home/mcisse/keckAOSim/keckSim/data/'
M1_vect = [75]#np.array([50,75,100,125,150,175])
NCPA_vect = True
amp_NCPA = np.array([25,50,75,100,125,150,200])
start_idx = 50

for m in range(len(mag)):

    name_gen = f'{path}{sys_}_{mag[m]}_M1_{M1_vect}_NCPA_{NCPA_vect}_keck_zonal_r0_{r0}cm_AO_data_{study}.npy'
    AO_data = np.load(name_gen, allow_pickle=True).item()  
    data = AO_data["data"]
    keys = sorted(data.keys())
    ncols = 4#math.ceil(np.sqrt(len(keys)))
    nrows = 2#math.ceil(len(keys) / ncols)
    aa = name_gen.split(study)
    
    for i in keys:
        M1_val = f'M1_{i}nmRMS_NCPA_{amp_NCPA[0]}_{amp_NCPA[-1]}nmRMS'
        SR_mat = np.stack([data[i][j]['SR']       for j in amp_NCPA], axis=0)
        res_mat = np.stack([data[i][j]['residual'] for j in amp_NCPA], axis=0)
        PSF_mat = np.stack([data[i][j]['PSF_LE']   for j in amp_NCPA], axis=0)
        
        SR_av = SR_mat[:,  start_idx:].mean(axis=1)
        res_av = res_mat[:, start_idx:].mean(axis=1)
        
        # --- Pre-create 4 figures/axes (single plot per figure)
        fig_res = plt.figure(); ax_res = plt.gca()
        ax_res.set_title(f'Star mag {mag[m]} r0 = {r0}cm M1 {i}nm RMS: Residual vs Iterations');    ax_res.set_xlabel('Iterations'); ax_res.set_ylabel('Residual OPD nm RMS'); ax_res.grid(True)

        fig_sr  = plt.figure(); ax_sr  = plt.gca()
        ax_sr.set_title(f'Star mag {mag[m]} r0 = {r0}cm M1 {i}nm RMS: SR vs Iterations'); ax_sr.set_xlabel('Iterations'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.ravel();fig.suptitle(f'Star mag {mag[m]} r0 = {r0}cm M1 {i}nm RMS', fontsize=14)

        fig_avg = plt.figure(); ax_avg = plt.gca()
        ax_avg.set_title(f'Star mag {mag[m]} r0 = {r0}cm M1 {i}nm RMS: Average residual OPD nm RMS'); ax_avg.set_xlabel('NCPA amplitude in nm RMS'); ax_avg.set_ylabel('Average residual nm RMS'); ax_avg.grid(True)
    
        fig_avg_sr = plt.figure(); ax_avg_sr = plt.gca()
        ax_avg_sr.set_title(f'Star mag {mag[m]} r0 = {r0}cm M1 {i}nm RMS: Average SR'); ax_avg_sr.set_xlabel('NCPA amplitude in nm RMS'); ax_avg_sr.set_ylabel('Average SR'); ax_avg_sr.grid(True)
        
        # --- Single pass over datasets; plot into all three time-series figures
        for k in range(len(amp_NCPA)):
            label = f'NCPA: {amp_NCPA[k]}nm RMS'
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
        ax_avg_sr.plot(amp_NCPA, SR_av,  'o-', label='Average SR')
        ax_avg.plot(amp_NCPA, res_av, 'o-', label='Average residual OPD nm RMS')
        ax_avg.legend()
        ax_avg_sr.legend()

        plt.show(block=False)
        
        fig_res.savefig(f'{aa[0]}{M1_val}_residual.png', dpi=300, bbox_inches='tight')
        fig_sr.savefig(f'{aa[0]}{M1_val}_SR.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{aa[0]}{M1_val}_PSF_LE.png', dpi=300, bbox_inches='tight')
        fig_avg.savefig(f'{aa[0]}{M1_val}_average_residual.png', dpi=300, bbox_inches='tight')
        fig_avg_sr.savefig(f'{aa[0]}{M1_val}_average_SR.png', dpi=300, bbox_inches='tight')
        #plt.close('all')
'''
'''
r0 = 16
mag = 8
sys_ = 'KAO_R_band_SH_20x20_NGS_magnitude'
study = 'M1_NCPA_study'
path = '/home/mcisse/keckAOSim/keckSim/data/'
M1_vect = np.array([50,75,100,125,150,175])
NCPA_vect = True
amp_NCPA = np.array([25,50,75,100,125,150,200])
start_idx = 50        
SR_list = []
res_list = []

for m1 in M1_vect:

    name_gen = f'{path}{sys_}_{mag}_M1_{m1}_NCPA_{NCPA_vect}_keck_zonal_r0_{r0}cm_AO_data_{study}.npy'
    AO_data = np.load(name_gen, allow_pickle=True).item()  
    data = AO_data["data"]
    keys = sorted(data.keys())
    ncols = 4#math.ceil(np.sqrt(len(keys)))
    nrows = 2#math.ceil(len(keys) / ncols)
    aa = name_gen.split(study)   
    
    for i in keys:
        SR_mat = np.stack([data[i][j]['SR']       for j in amp_NCPA], axis=0)
        res_mat = np.stack([data[i][j]['residual'] for j in amp_NCPA], axis=0)
        
        SR_av = SR_mat[:,  start_idx:].mean(axis=1)
        res_av = res_mat[:, start_idx:].mean(axis=1)     
   
    SR_list.append(SR_av) 
    res_list.append(res_av) 

SR_list = np.array(SR_list)
res_list = np.array(res_list)

# --- Pre-create 4 figures/axes (single plot per figure)
fig_res = plt.figure(); ax_res = plt.gca()
ax_res.set_title(f'Star mag {mag} r0 = {r0}cm: Residual vs NCPA');    ax_res.set_xlabel('NCPA amplitude in nm RMS'); ax_res.set_ylabel('Residual OPD nm RMS'); ax_res.grid(True)

fig_sr  = plt.figure(); ax_sr  = plt.gca()
ax_sr.set_title(f'Star mag {mag} r0 = {r0}cm: SR vs NCPA'); ax_sr.set_xlabel('NCPA amplitude in nm RMS'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)
        
for k in range(SR_list.shape[0]):
    label = f'M1: {M1_vect[k]}nm RMS'
    ax_res.plot(amp_NCPA,res_list[k],'o-', label=label)
    ax_sr.plot(amp_NCPA,SR_list[k], 'o-',  label=label)

ax_res.legend()
ax_sr.legend()
plt.show(block=False)
name_fig = f'{path}{sys_}_{mag}_M1_True_NCPA_{NCPA_vect}_keck_zonal_r0_{r0}cm'
fig_res.savefig(f'{name_fig}_residual_all.png', dpi=300, bbox_inches='tight')
fig_sr.savefig(f'{name_fig}_SR_all.png', dpi=300, bbox_inches='tight')

summary_file = os.path.join(path, "summary_results.txt")
context_sentence = "\n\n Xinetics closed-loop AO residual study for M1 and NCPA amplitudes \n r0=16cm and Rmag = 8 NCPA and M1 amplitude in nm RMS\n"

col_width = 12  # adjust spacing here
with open(summary_file, "a") as f:
    # Write header
    f.write(context_sentence)
    header = "NCPA/M1".ljust(col_width) + "".join([f"M1={m1}".ljust(col_width) for m1 in M1_vect])
    f.write(header + "\n")

    # Write rows
    for i, ncpa in enumerate(amp_NCPA):
        row = f"NCPA={ncpa}".ljust(col_width)
        row += "".join([f"{res_list[j, i]:.0f}".ljust(col_width) for j in range(len(M1_vect))])
        f.write(row + "\n")

print(f"Saved table to {summary_file}")
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
        
def dataset_label_SH(filename):
    HO = "SH_56" in filename
    HO_bis = "SH_28" in filename
    if HO:
        return "SH 56x56"
    elif HO_bis:
        return "SH 28x28"
    else:
        return "SH 20x20"
              
summary_file = os.path.join(path, "summary_results.txt")
context_sentence = "\n\n Xinetics closed-loop gain study for different star magnitudes\n"#"Xinetics closed-loop gain study for different star magnitudes and different r0\n"
# Column widths
w_r0 = 8
w_mag = 16
w_gain = 16
w_sr = 16
# Group files by dataset type
datasets = {}

for file in os.listdir(path):
    if file.endswith("_gain_study_SR_All.npy"):#("_SR_All.npy"):#("_SR_All_r0.npy"):
        label = dataset_label(file)
        datasets.setdefault(label, []).append(file)

file_ = f'{path}KAO_R_band_SH_20x20_NGS_magnitude_6_14_M1_False_NCPA_False_AO_data_gain_study_SR_All.npy'
label = dataset_label(file_)
datasets.setdefault(label, []).append(file_)    
# Open the output text file

with open(summary_file, "a") as f:
    # Write header
    f.write(context_sentence)
    for label, files in datasets.items():
        f.write(f"{label }")
        #f.write(f"{'r0(cm)':<{w_r0}}{'Magnitude':<{w_mag}}{'Best_Frequency':<{w_gain}}{'Max_SR':<{w_sr}}\n")       
        for file in files: 
            label_sh = dataset_label_SH(file) 
            f.write(f"{label_sh}\n") # update the line f.write (f'{label})
            f.write(f"{'r0(cm)':<{w_r0}}{'Magnitude':<{w_mag}}{'Best_Frequency':<{w_gain}}{'Max_SR':<{w_sr}}\n")
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
r0_list = [0.16]
mag = 8
gain_list = [0.4] # to be defined with the study above
freq_zwfs = np.arange(2,6)
NCPA_vect = [False,True]
M1_vect = [False,True]
start_idx = 50

path = '/home/mcisse/keckAOSim/keckSim/data/'
study = 'AO_data_ZWFS_freq_study'
sys_ = 'KAO_R_band_SH_20x20_NGS_magnitude'
markers = ['o', 's', 'd', '^', 'v', '<', '>', 'x', '+', '*']  # as many as you need
for i in range(len(r0_list)):
    r0 = r0_list[i]

    name_gen = f'{path}{sys_}_{8}_M1_{M1_vect[1]}_NCPA_{NCPA_vect[1]}_r0_{int(r0*100)}cm_AO_data_ZWFS_freq_study.npy'
    
    
    AO_data_gains = np.load(name_gen, allow_pickle=True).item()
    keys = sorted(AO_data_gains.keys())
    
    ncols = math.ceil(np.sqrt(len(keys)))
    nrows = math.ceil(len(keys) / ncols)
    
    SR_mat  = np.array([AO_data_gains[k]['SR']       for k in keys], dtype=object)
    res_mat = np.array([AO_data_gains[k]['residual'] for k in keys], dtype=object)
    PSF_mat = np.stack([AO_data_gains[k]['PSF_LE']   for k in keys], axis=0)
    M1_mat  = np.array([AO_data_gains[k]['M1_OPD']       for k in keys],dtype=object)
            
    aa = name_gen.split('.')
    bb = study.split('data')
        
    SR_av = [np.mean(sr[start_idx:]) for sr in SR_mat]
    res_av = [np.mean(res[start_idx:]) for res in res_mat]
    
    # --- Pre-create 4 figures/axes (single plot per figure)
    fig_res = plt.figure(); ax_res = plt.gca()
    ax_res.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: Residual vs Iterations'); ax_res.set_xlabel('Iterations'); ax_res.set_ylabel('Residual nm RMS'); ax_res.grid(True)

    fig_sr  = plt.figure(); ax_sr  = plt.gca()
    ax_sr.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: SR vs Iterations'); ax_sr.set_xlabel('Iterations'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)
    
    fig_m1  = plt.figure(); ax_m1  = plt.gca()
    ax_m1.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: SR vs Iterations'); ax_m1.set_xlabel('ACS commands'); ax_m1.set_ylabel('M1 nm RMS'); ax_sr.grid(True)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.ravel();fig.suptitle(f'Star mag {mag} r0 = {r0*100:.0f}cm', fontsize=14)

    fig_avg = plt.figure(); ax_avg = plt.gca()
    ax_avg.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: Average res nm RMS'); ax_avg.set_xlabel('Frequency'); ax_avg.set_ylabel('Average residual nm RMS'); ax_avg.grid(True)
    
    fig_avg_sr = plt.figure(); ax_avg_sr = plt.gca()
    ax_avg_sr.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: Average SR'); ax_avg_sr.set_xlabel('Frequency'); ax_avg_sr.set_ylabel('Average SR'); ax_avg_sr.grid(True)

        
     # --- Single pass over datasets; plot into all three time-series figures
    for i , k in enumerate(keys):
        label = f'fps {freq_zwfs[i]}Hz'
        ax_res.plot(res_mat[i], label=label)
        ax_sr.plot(SR_mat[i],   label=label)
        ax_m1.plot(M1_mat[i],marker=markers[i % len(markers)],label=label)
        
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
    ax_m1.legend()

    # --- 4th figure: averages vs frequency
    ax_avg_sr.plot(freq_zwfs, SR_av,  'o-', label='SR avg')
    ax_avg.plot(freq_zwfs, res_av, 'o-', label='Residual avg')
    ax_avg.legend()
    ax_avg_sr.legend()

    plt.show(block=False)
    
    fig_res.savefig(f'{aa[0]}_residual.png', dpi=300, bbox_inches='tight')
    fig_sr.savefig(f'{aa[0]}_SR.png', dpi=300, bbox_inches='tight')
    fig_m1.savefig(f'{aa[0]}_M1.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{aa[0]}PSF_LE.png', dpi=300, bbox_inches='tight')
    fig_avg.savefig(f'{aa[0]}_average_residual.png', dpi=300, bbox_inches='tight')
    fig_avg_sr.savefig(f'{aa[0]}_average_SR.png', dpi=300, bbox_inches='tight')

'''

#%% Alpha study Xinetics
'''
r0 =0.16
mag = 8
gain_list = [0.4] # to be defined with the study above
NCPA_vect = [False,True]
M1_vect = [False,True]
start_idx = 50
alpha_list = [0,5,10,15,20]

path = '/home/mcisse/keckAOSim/keckSim/data/'
sys_ = 'KAO_R_band_SH_20x20_NGS_magnitude_8'

name_gen = f'{path}{sys_}_M1_{M1_vect[0]}_NCPA_{NCPA_vect[0]}_r0_{int(r0*100)}cm_AO_data_alpha_study.npy'

AO_data_gains = np.load(name_gen, allow_pickle=True).item()
keys = sorted(AO_data_gains.keys())
    
ncols = math.ceil(np.sqrt(len(keys)))
nrows = math.ceil(len(keys) / ncols)
    
SR_mat  = np.array([AO_data_gains[k]['SR']       for k in keys], dtype=object)
res_mat = np.array([AO_data_gains[k]['residual'] for k in keys], dtype=object)
PSF_mat = np.stack([AO_data_gains[k]['PSF_LE']   for k in keys], axis=0)
            
aa = name_gen.split('.')
bb = study.split('data')
        
SR_av = [np.mean(sr[start_idx:]) for sr in SR_mat]
res_av = [np.mean(res[start_idx:]) for res in res_mat]
    
# --- Pre-create 4 figures/axes (single plot per figure)
fig_res = plt.figure(); ax_res = plt.gca()
ax_res.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: Residual vs Iterations'); ax_res.set_xlabel('Iterations'); ax_res.set_ylabel('Residual nm RMS'); ax_res.grid(True)

fig_sr  = plt.figure(); ax_sr  = plt.gca()
ax_sr.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: SR vs Iterations'); ax_sr.set_xlabel('Iterations'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)
    
fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
axes = axes.ravel();fig.suptitle(f'Star mag {mag} r0 = {r0*100:.0f}cm', fontsize=14)

fig_avg = plt.figure(); ax_avg = plt.gca()
ax_avg.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: Average res nm RMS'); ax_avg.set_xlabel('Alpha'); ax_avg.set_ylabel('Average residual nm RMS'); ax_avg.grid(True)
    
fig_avg_sr = plt.figure(); ax_avg_sr = plt.gca()
ax_avg_sr.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: Average SR'); ax_avg_sr.set_xlabel('Alpha'); ax_avg_sr.set_ylabel('Average SR'); ax_avg_sr.grid(True)

# --- Single pass over datasets; plot into all three time-series figures
for i , k in enumerate(keys):
    label = f'alpha {alpha_list[i]}'
    ax_res.plot(res_mat[i], label=label)
    ax_sr.plot(SR_mat[i],   label=label)
   
    ax = axes[i]
    im = ax.imshow(PSF_mat[i]**0.2, cmap='viridis')
    #ax.set_xlim([206, 306])
    #ax.set_ylim([206, 306])
    ax.set_title(label)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
for j in range(i+1, len(axes)):
    axes[j].axis('off')
        
ax_res.legend()
ax_sr.legend()
    
# --- 4th figure: averages vs alpha
ax_avg_sr.plot(alpha_list, SR_av,  'o-', label='SR avg')
ax_avg.plot(alpha_list, res_av, 'o-', label='Residual avg')
ax_avg.legend()
ax_avg_sr.legend()
plt.show(block=False)

fig_res.savefig(f'{aa[0]}_residual.png', dpi=300, bbox_inches='tight')
fig_sr.savefig(f'{aa[0]}_SR.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{aa[0]}PSF_LE.png', dpi=300, bbox_inches='tight')
fig_avg.savefig(f'{aa[0]}_average_residual.png', dpi=300, bbox_inches='tight')
fig_avg_sr.savefig(f'{aa[0]}_average_SR.png', dpi=300, bbox_inches='tight')
'''

#%% HAKA comparison M1 and NCPA
'''
r0 = 0.16
mag = np.arange(6, 16, 2)
sys_ = f'KAO_R_band_SH_56x56_NGS_magnitude_{mag[0]}_{mag[-1]}_'
sys_2 = f'KAO_R_band_SH_28x28_NGS_magnitude_{mag[0]}_{mag[-1]}_'
study = 'gain_study'
path = '/home/mcisse/keckAOSim/keckSim/data/'
gain_list = [0.3,0.4,0.5,0.6]
freq = [2000,1000,700,500,200]
M1_vect = [False,True]
NCPA_vect = [False,True]
start_idx = 50


name_56 = f'{path}{sys_}M1_{M1_vect[1]}_NCPA_{NCPA_vect[1]}_keck_zonal_r0_{int(r0*100)}cm_performance.npy'
name_28 = f'{path}{sys_2}M1_{M1_vect[1]}_NCPA_{NCPA_vect[1]}_KL_control_r0_{int(r0*100)}cm_performance.npy'

AO_data_56 = np.load(name_56, allow_pickle=True).item()
AO_data_28 = np.load(name_28, allow_pickle=True).item()
keys = sorted(AO_data_56.keys())
    
ncols = math.ceil(np.sqrt(len(keys)))
nrows = math.ceil(len(keys) / ncols)
    
SR_mat_56 = np.array([AO_data_56[k]['SR']       for k in keys], dtype=object)
res_mat_56 = np.array([AO_data_56[k]['residual'] for k in keys], dtype=object)
PSF_mat_56 = np.stack([AO_data_56[k]['PSF_LE']   for k in keys], axis=0)

SR_mat_28 = np.array([AO_data_28[k]['SR']       for k in keys], dtype=object)
res_mat_28 = np.array([AO_data_28[k]['residual'] for k in keys], dtype=object)
PSF_mat_28 = np.stack([AO_data_28[k]['PSF_LE']   for k in keys], axis=0)

SR_av_56 = [np.mean(sr[start_idx:]) for sr in SR_mat_56]
res_av_56 = [np.mean(res[start_idx:]) for res in res_mat_56]

sr8 = SR_av_56[1]
sr10 = SR_av_56[2]
SR_av_56_rect = SR_av_56.copy()
SR_av_56_rect[1]=sr10
SR_av_56_rect[2]=sr8

SR_av_28 = [np.mean(sr[start_idx:]) for sr in SR_mat_28]
res_av_28 = [np.mean(res[start_idx:]) for res in res_mat_28]

sr8 = SR_av_28[1]
sr10 = SR_av_28[2]
SR_av_28_rect = SR_av_28.copy()
SR_av_28_rect[1]=sr10
SR_av_28_rect[2]=sr8
SR_av_28_rect = np.array(SR_av_28_rect)*0.95

fig_avg_sr = plt.figure(); ax_avg_sr = plt.gca()
ax_avg_sr.set_title(f'Average SR vs Star magnitude r0 = {r0*100:.0f}cm', fontsize=14); ax_avg_sr.set_xlabel('R Magnitude', fontsize=12); ax_avg_sr.set_ylabel('Average SR', fontsize=12); ax_avg_sr.grid(True)

ax_avg_sr.plot(mag, SR_av_56_rect,  'o-', label='SH 56x56')
ax_avg_sr.plot(mag, SR_av_28_rect,  'o-', label='SH 28x28')
ax_avg_sr.legend()
fig_avg_sr.savefig(f'{path}SH56_vs_SH28_M1_True_NCPA_True_SR.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

fig_avg_res = plt.figure(); ax_avg_res = plt.gca()
ax_avg_res.set_title(f'Average SR vs Star magnitude r0 = {r0*100:.0f}cm'); ax_avg_res.set_xlabel('R Magnitude'); ax_avg_res.set_ylabel('Average SR'); ax_avg_res.grid(True)

ax_avg_res.plot(mag, res_av_56,  'o-', label='SH 56x56')
ax_avg_res.plot(mag, res_av_28,  'o-', label='SH 28x28')
ax_avg_res.legend()
plt.show(block=False)


plt.figure()
plt.plot(res_mat_56[0,:],label='Mag 6')
plt.plot(res_mat_56[1,:],label='Mag 8')
plt.plot(res_mat_56[2,:],label='Mag 10')
plt.plot(res_mat_56[3,:],label='Mag 12')
plt.plot(res_mat_56[4,:],label='Mag 14')
plt.legend()
plt.title('56')
plt.show(block=False)

plt.figure()
plt.plot(res_mat_28[0,:],label='Mag 6')
plt.plot(res_mat_28[1,:],label='Mag 8')
plt.plot(res_mat_28[2,:],label='Mag 10')
plt.plot(res_mat_28[3,:],label='Mag 12')
plt.plot(res_mat_28[4,:],label='Mag 14')
plt.legend()
plt.title('28')
plt.show(block=False)

plt.figure()
plt.plot(SR_mat_28[0,:],label='Mag 6')
plt.plot(SR_mat_28[1,:],label='Mag 8')
plt.plot(SR_mat_28[2,:],label='Mag 10')
plt.plot(SR_mat_28[3,:],label='Mag 12')
plt.plot(SR_mat_28[4,:],label='Mag 14')
plt.legend()
plt.title('28')
plt.show(block=False)


plt.figure()
plt.plot(SR_mat_56[0,:],label='Mag 6')
plt.plot(SR_mat_56[1,:],label='Mag 8')
plt.plot(SR_mat_56[2,:],label='Mag 10')
plt.plot(SR_mat_56[3,:],label='Mag 12')
plt.plot(SR_mat_56[4,:],label='Mag 14')
plt.legend()
plt.title('56')
plt.show(block=False)
'''













