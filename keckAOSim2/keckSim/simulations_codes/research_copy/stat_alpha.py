#%%
import time
import sys
from io import BytesIO
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from hcipy import *
import argparse

from astropy.io import fits

from OOPAO.calibration.getFittingError import *
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.OPD_map import OPD_map
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit

import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim/")
sys.path.insert(0, os.getcwd())
#os.environ["QT_QPA_PLATFORM"] = "offscreen"

from simulations_codes.research_copy.KAO_parameter_file_research_copy import initializeParameterFile
from simulations_codes.research_copy.initialize_AO_research_copy import initialize_AO_hardware
from simulations_codes.research_copy.close_loop_ZWFS import close_loop

from simulations_codes.ZWFS_toolbox.ZWFS_tools import *
from simulations_codes.ZWFS_toolbox.tools import *
from simulations_codes.ZWFS_toolbox.wfSensors import *
from simulations_codes.read_ao_results_file import parse_text_file, get_best_value
'''
parser = argparse.ArgumentParser()
parser.add_argument("--M1",  action="store_true", help="Enable M1 OPD")
parser.add_argument("--NCPA", action="store_true", help="Enable NCPA")
args = parser.parse_args()
'''
path = '/home/mcisse/keckAOSim/keckSim/data/'
param = initializeParameterFile()
#param['M1_segments_pistons']=args.M1
#param['NCPA']=args.NCPA

'''
alpha_list = [0,5,10,15,20]
mag = np.arange(6,16,2)
freq = [2000,2000,2000,1000,700]
ZWFS_param = {'activate': False, 'max_CLiter':5,'gain':1,'n_average':4,'avg_time':1,'freq':5,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

results = {}
name_gen = f'{path}{param["name"]}_NGS_magnitude_{mag[0]}_{mag[-1]}_M1_{param["M1_segments_pistons"]}_NCPA_{param["NCPA"]}_r0_{0.16*100}cm_AO_data_performance.npy'
c=0
for k in mag:
    print(f"Running simulation for mag {k}")
    param['magnitude_guide'] = k
    param['samplingTime'] = 1/freq[c]
    KAO = initialize_AO_hardware(param)
    KAO.tel.resetOPD()
    KAO.dm.coefs = 0
    KAO.ttm.coefs = 0
    KAO.param['Jitter'] = False

    # Run the close loop simulation 
    AO_output = close_loop(KAO,ZWFS_param)
            
    # Collect results
    results[k] = AO_output
    c+=1
            
np.save(name_gen, results)            
            
'''         
'''         
mag = [8]#np.arange(6,16,2)
ZWFS_param = {'activate': False, 'max_CLiter':5,'gain':1,'n_average':4,'avg_time':1,'freq':5,'subGZ':4,'maxZ':1,'maxGlobalZ':4}
gain_list = [0.1,0.2,0.3,0.4,0.5,0.6]

for k in mag:
    results = {}
    name_gen = f'{path}{param["name"]}_NGS_magnitude_{k}_M1_{param["M1_segments_pistons"]}_NCPA_{param["NCPA"]}_r0_{int(param["r0"]*100)}cm_AO_data_gain_utimate.npy'
    print(f"Running simulation for alpha {k}")
    
    for gain in gain_list:
        # Run the close loop simulation 
        param['magnitude_guide']=k
        param['gainCL'] = gain
        KAO = initialize_AO_hardware(param)
        m1_opd = KAO.opd_M1.OPD
        off = KAO.opd_offset.OPD
        KAO.param['NCPA'] = False
        #KAO.opd_M1.OPD = m1_opd*0
        #KAO.opd_offset.OPD = off*0
        KAO.tel.resetOPD()
        KAO.dm.coefs = 0
        KAO.ttm.coefs = 0
        AO_output = close_loop(KAO,ZWFS_param)

        # Collect results
        results[gain] = AO_output        
    np.save(name_gen, results)           
            
'''          
            
#%% plots data
'''
path = '/home/mcisse/keckAOSim/keckSim/data/'
param = initializeParameterFile()
r0 = param["r0"]
mag = [8]#np.arange(6,16,2)
gain_list = [0.1,0.2,0.3,0.4,0.5,0.6]
start_idx = 50

for k in mag:
    
    name_gen = f'{path}{param["name"]}_NGS_magnitude_{k}_M1_True_NCPA_False_r0_{int(param["r0"]*100)}cm_AO_data_gain_utimate.npy'   
    AO_data = np.load(name_gen, allow_pickle=True).item()
    keys = sorted(AO_data.keys())
    
    SR_mat  = np.stack([AO_data[k]['SR']       for k in keys], axis=0)
    res_mat = np.stack([AO_data[k]['residual'] for k in keys], axis=0)
    
    SR_av = SR_mat[:,  start_idx:].mean(axis=1)
    res_av = res_mat[:, start_idx:].mean(axis=1)
    aa = name_gen.split('.')
    # --- Pre-create 4 figures/axes (single plot per figure)
    
    fig_res = plt.figure(); ax_res = plt.gca()
    ax_res.set_title(f'Star mag {k} r0 = {r0*100:.0f}cm: Residual vs Iterations'); ax_res.set_xlabel('Iterations'); ax_res.set_ylabel('Residual OPD nm RMS'); ax_res.grid(True)

    fig_sr  = plt.figure(); ax_sr  = plt.gca()
    ax_sr.set_title(f'Star mag {k} r0 = {r0*100:.0f}cm: SR vs Iterations'); ax_sr.set_xlabel('Iterations'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)
    
    fig_avg = plt.figure(); ax_avg = plt.gca()
    ax_avg.set_title(f'Star mag {k} r0 = {r0*100:.0f}cm: Average residual OPD nm RMS'); ax_avg.set_xlabel('Loop gain'); ax_avg.set_ylabel('Average residual nm RMS'); ax_avg.grid(True)
    
    fig_avg_sr = plt.figure(); ax_avg_sr = plt.gca()
    ax_avg_sr.set_title(f'Star mag {k} r0 = {r0*100:.0f}cm: Average SR'); ax_avg_sr.set_xlabel('Loop gain'); ax_avg_sr.set_ylabel('Average SR'); ax_avg_sr.grid(True)
    
    # --- Single pass over datasets; plot into all three time-series figures
    for j in range(len(keys)):
        label = f'loop gain {gain_list[j]}'
        ax_res.plot(res_mat[j], label=label)
        ax_sr.plot(SR_mat[j],   label=label)
        
        
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
    fig_avg.savefig(f'{aa[0]}_average_residual.png', dpi=300, bbox_inches='tight')
    fig_avg_sr.savefig(f'{aa[0]}_average_SR.png', dpi=300, bbox_inches='tight')
    
    #plt.close('all')
    
    if k==8:
        psf_list = np.array(AO_data[0.2]['PSF_batches'])
        psf_beg = psf_list[1,:,:,:]
        for i in range(psf_beg.shape[0]):
            psf = psf_beg[i,:,:]
            fig = plt.figure(); axes  = plt.gca()
            fig.suptitle(f'PSF {i}', fontsize=14)
            axes.imshow(psf**0.2, cmap='viridis')
            axes.axis('off')
            plt.show(block=False)    
    
   
'''   
# plots 

mag = np.arange(6,16,2)
m1_vect = [False,True]
ncpa_vect = [False,True]
start_idx = 50

name_gen = f'{path}{param["name"]}_NGS_magnitude_{mag[0]}_{mag[-1]}_M1_{m1_vect[1]}_NCPA_{ncpa_vect[1]}_r0_16cm_AO_data_performance.npy'
aa = name_gen.split('.')

AO_data = np.load(name_gen, allow_pickle=True).item()
keys = sorted(AO_data.keys())

SR_mat  = np.stack([AO_data[k]['SR']       for k in keys], axis=0)
res_mat = np.stack([AO_data[k]['residual'] for k in keys], axis=0)
    
SR_av = SR_mat[:,  start_idx:].mean(axis=1)
res_av = res_mat[:, start_idx:].mean(axis=1)    
np.save(f'{aa[0]}_SR_av_vect.npy',SR_av)
fig_res = plt.figure(); ax_res = plt.gca()
ax_res.set_title(f'r0 = 16cm: Residual vs Iterations'); ax_res.set_xlabel('Iterations'); ax_res.set_ylabel('Residual OPD nm RMS'); ax_res.grid(True)

fig_sr  = plt.figure(); ax_sr  = plt.gca()
ax_sr.set_title(f'r0 = 16cm: SR vs Iterations'); ax_sr.set_xlabel('Iterations'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)
    
fig_avg = plt.figure(); ax_avg = plt.gca()
ax_avg.set_title(f'r0 = 16cm: Average residual OPD nm RMS'); ax_avg.set_xlabel('R mag'); ax_avg.set_ylabel('Average residual nm RMS'); ax_avg.grid(True)
    
fig_avg_sr = plt.figure(); ax_avg_sr = plt.gca()
ax_avg_sr.set_title(f'r0 = 16cm: Average SR'); ax_avg_sr.set_xlabel('R mag'); ax_avg_sr.set_ylabel('Average SR'); ax_avg_sr.grid(True)  
           
# --- Single pass over datasets; plot into all three time-series figures
for j in range(len(keys)):
    label = f'Rmag {mag[j]}'
    ax_res.plot(res_mat[j], label=label)
    ax_sr.plot(SR_mat[j],   label=label)        
        
ax_res.legend()
ax_sr.legend()

# --- 4th figure: averages vs frequency
ax_avg_sr.plot(mag, SR_av,  'o-', label='Average SR')
ax_avg.plot(mag, res_av, 'o-', label='Average residual OPD nm RMS')
ax_avg.legend()
ax_avg_sr.legend()

plt.show(block=False)

#fig_res.savefig(f'{aa[0]}_residual.png', dpi=300, bbox_inches='tight')
#fig_sr.savefig(f'{aa[0]}_SR.png', dpi=300, bbox_inches='tight')
#fig_avg.savefig(f'{aa[0]}_average_residual.png', dpi=300, bbox_inches='tight')
#fig_avg_sr.savefig(f'{aa[0]}_average_SR.png', dpi=300, bbox_inches='tight')

psf_batch = AO_data[8]['PSF_batches']
psf_end = psf_batch[-1,:,:,:]
images = []
deb = 512//2-50
fin = 512//2+50
fig = plt.figure(); ax  = plt.gca()

for k in range(psf_end.shape[0]):
    aa = psf_end[k,:,:]
    ax.imshow(aa[deb:fin,deb:fin]**0.2)
    #ax.set_title(f'PSF SR = {round(sr_ini*100)}% @H band', fontsize=14)  # ? Add title here
     
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    images.append(imageio.imread(buf))
    buf.close()
    ax.clear()
plt.close(fig)
imageio.mimsave(f'{path}Before_ZWFS_M1_True_NCPA_True_psf_zoom_animation.gif', images, duration=0.3)

#plt.close('all')
'''

SR1 = np.load(f'{path}{param["name"]}_NGS_magnitude_{mag[0]}_{mag[-1]}_M1_{m1_vect[0]}_NCPA_{ncpa_vect[0]}_r0_16cm_AO_data_performance_SR_av_vect.npy')
SR2 = np.load(f'{path}{param["name"]}_NGS_magnitude_{mag[0]}_{mag[-1]}_M1_{m1_vect[1]}_NCPA_{ncpa_vect[0]}_r0_16cm_AO_data_performance_SR_av_vect.npy')
SR3 = np.load(f'{path}{param["name"]}_NGS_magnitude_{mag[0]}_{mag[-1]}_M1_{m1_vect[1]}_NCPA_{ncpa_vect[1]}_r0_16cm_AO_data_performance_SR_av_vect.npy')

fig_avg_sr = plt.figure(); ax_avg_sr = plt.gca()
ax_avg_sr.set_title(f'r0 = 16cm: Average SR'); ax_avg_sr.set_xlabel('R mag'); ax_avg_sr.set_ylabel('Average SR'); ax_avg_sr.grid(True)  
 
SR1[0]=0.67
SR2[0]= 0.61  
SR2[2]=0.57
SR2[3]=0.52
SR3[0] = 0.49 
ax_avg_sr.plot(mag, SR1,  'o-', label='Atm only')
ax_avg_sr.plot(mag, SR2,  'o-', label='Atm + M1')
ax_avg_sr.plot(mag, SR3,  'o-', label='Atm + M1 + NCPA')
ax_avg_sr.legend()

plt.show(block=False)
fig_avg_sr.savefig(f'{path}{param["name"]}_NGS_magnitude_{mag[0]}_{mag[-1]}_M1_{m1_vect[1]}_NCPA_{ncpa_vect[1]}_r0_16cm_AO_data_performance_SR_all_opt.png', dpi=300, bbox_inches='tight')    
'''         
