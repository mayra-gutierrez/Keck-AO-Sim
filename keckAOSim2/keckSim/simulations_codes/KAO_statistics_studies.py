#%%
import time
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from hcipy import *
import argparse
from astropy.io import fits

import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim/")
sys.path.insert(0, os.getcwd()) 
from simulations_codes.KAO_parameter_file import initializeParameterFile
from simulations_codes.initialize_AO import initialize_AO_hardware
from simulations_codes.close_loop import close_loop

parser = argparse.ArgumentParser()
parser.add_argument("--M1",  action="store_true", help="Enable M1 OPD")
parser.add_argument("--NCPA", action="store_true", help="Enable NCPA")
parser.add_argument("--offset", type=float, help="Set Pure NCPA amplitude in nm (e.g., 50 for 50 nm)")
args = parser.parse_args()


param = initializeParameterFile()
param['M1_segments_pistons']=args.M1
param['NCPA']=args.NCPA

if args.offset is not None:
    param['Pure_NCPA_amplitude'] = args.offset * 1e-9  # convert from nm to meters
    offset = True
    print('NCPA offset')
else:
    param['Pure_NCPA_amplitude'] = 0  
    offset = False
    
#%% Parameter import

param = initializeParameterFile()
path = param['pathOutput']
ZWFS_param = {'activate': False, 'max_CLiter':5,'gain':1,'n_average':5,'avg_time':4,'freq':4,'subGZ':4,'maxZ':1,'maxGlobalZ':4}
#%% Alpha Studies
'''
mag = 8
gainCL = 0.3
alpha_list = [1e13,1e14,1e15,1e16]
results = {}
name_study = f'{path}{param["name"]}_NGS_magnitude_{mag}_r0_{int(param["r0"]*100)}cm_AO_data_alpha_study.npy'

for k in range(len(alpha_list)):
    param['magnitude_guide'] = mag
    param['gainCL'] = gainCL
    param['nLoop'] = 3000
    param['alpha'] = alpha_list[k]
    
    KAO = initialize_AO_hardware(param)
    AO_output = close_loop(KAO, ZWFS_param)
    
    results_k = {'SR': AO_output['SR'],'residual': AO_output['residual']}
    results[k] = results_k
    
np.save(name_study, results)    
'''
#%% Frequency study
'''
print('FREQUENCY')
mag_list = np.arange(6,16,2)
frequency = [2000,1000,700,500,200]
gainCL = 0.3 #[0.2,0.3,0.3,0.4,0.5,0.6,0.7]

if np.isscalar(frequency) and not np.isscalar(gainCL):
    study_type = 'gain'
elif not np.isscalar(frequency) and np.isscalar(gainCL):
    study_type = 'freq'
else:
    raise ValueError("Invalid study: exactly one of 'frequency' of 'gainCL' must be scalar")
        
for mag in mag_list:
    param['magnitude_guide'] = mag
    param['gainCL'] = gainCL
    param['nLoop'] = 3000
    
    name_study = f'{path}{param["name"]}_NGS_magnitude_{mag}_r0_{int(param["r0"]*100)}cm_AO_data_{study_type}_study.npy'
    results = {}
    
    print(f"Starting Frequency Study Data for magnitude {mag}")
    
    for freq in frequency:
        param['samplingTime'] = 1/freq
        
        KAO = initialize_AO_hardware(param)
        AO_output = close_loop(KAO, ZWFS_param)
    
        results_k = {'SR': AO_output['SR'],'residual': AO_output['residual']}
        results[freq] = results_k
    
    np.save(name_study, results)
    print(f"Frequency Study Data saved for magnitude {mag}")
'''
#%% Gain study
print('GAIN')
mag_list = np.arange(6,16,2)
frequency = [2000,2000,700,700,200]
gainCL = [0.2,0.3,0.3,0.4,0.5,0.6,0.7]
param['nLoop'] = 3000

for i,mag in enumerate(mag_list):
    print(f'freq={frequency[i]}, mag = {mag}')
    param['magnitude_guide'] = mag
    param['samplingTime'] = 1/frequency[i]     
    
    name_study = f'{path}{param["name"]}_NGS_magnitude_{mag}_r0_{int(param["r0"]*100)}cm_AO_data_gain_study.npy'
    results = {}
    
    print(f"Starting Gain Study Data for magnitude {mag}")
    
    for gain in gainCL:
    
        param['gainCL'] = gain
        
        KAO = initialize_AO_hardware(param)
        AO_output = close_loop(KAO, ZWFS_param)
    
        results_k = {'SR': AO_output['SR'],'residual': AO_output['residual']}
        results[gain] = results_k
    
    np.save(name_study, results)
    print(f"Gain Study Data saved for magnitude {mag}")






















    

