#%%
import time
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from hcipy import *

from astropy.io import fits

from OOPAO.calibration.getFittingError import *
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.OPD_map import OPD_map
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit

import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim/")
sys.path.insert(0, os.getcwd())
os.environ["QT_QPA_PLATFORM"] = "offscreen"
 
from simulations_codes.KAO_parameter_file import initializeParameterFile
from simulations_codes.initialize_AO import initialize_AO_hardware
from simulations_codes.close_loop import close_loop
from simulations_codes.read_ao_results_file import parse_text_file, get_best_value

# %% -----------------------     Import system  ----------------------------------

param = initializeParameterFile()
#KAO = initialize_AO_hardware(param)
#M1_opd = KAO.opd_M1.OPD
path = '/home/mcisse/keckAOSim/keckSim/data/'

# %% Magnitude different magnitudes/gain/r0
'''
mag = np.arange(6, 16, 2)
freq = [2000,2000,1000,700,200]
gain_list = [0.3,0.4,0.5,0.6]
r0_list = [0.35,0.16,0.12]
seeds = len(gain_list)
res_seed = np.zeros((KAO.param['nLoop'],seeds))
SR_seed = np.zeros((KAO.param['nLoop'],seeds))
PSF_seed = np.zeros((KAO.param['scienceDetectorRes'], KAO.param['scienceDetectorRes'], seeds))
r0 = f"{int(KAO.atm.r0 * 100)}"

for r0 in range(len(r0_list)):
    KAO.atm.r0 = r0_list[r0]
    
    for m in range(len(mag)):
        KAO.param['magnitude_guide'] = mag[m]
        KAO.tel.samplingTime = 1/freq[m]
        results = {}
        name_gen = f'{path}{KAO.param["name"]}_NGS_magnitude_{mag[m]}_M1_{param["M1_segments_pistons"]}_NCPA_{param["NCPA"]}_r0_{int(r0_list[r0]*100)}cm_AO_data_gain_study.npy'

        for s in range(len(gain_list)):
            gainCL = gain_list[s]
            print(f"Running simulation for magnitude {m} with gain {gainCL}")
            KAO.param['gainCL'] = gainCL
            # Reset the system
            KAO.tel.resetOPD()
            KAO.dm.coefs = 0
            KAO.ttm.coefs = 0

            # Run the close loop simulation 
            AO_output = close_loop(KAO)
            
            # Collect results
            results[s] = AO_output
            
        # Save results or perform analysis as needed
        np.save(name_gen, results)
        print(f"Data saved for magnitude {mag[m]} and r0 {int(r0_list[r0]*100)}cm")

'''
#%% statistic performance as a function of the star mag and the r0
'''
mag = np.arange(6, 16, 2)
gain_list = [0.5,0.5,0.4,0.3]
r0_list = [0.35,0.16,0.12]
freq_list = # to be determine
nn = len(r0_list)
res_seed = np.zeros((KAO.param['nLoop'],nn))
SR_seed = np.zeros((KAO.param['nLoop'],nn))
PSF_seed = np.zeros((KAO.param['scienceDetectorRes'], KAO.param['scienceDetectorRes'], nn))
#gain_str = f"{int(KAO.param['gainCL'] * 10):02d}"
r0 = f"{int(KAO.atm.r0 * 100)}"

for m in range(len(mag)):
    KAO.param['magnitude_guide'] = mag[m]
    KAO.param['gainCL'] = gain_list[m]
    KAO.tel.samplingTime = freq[f]
    
    gain_str = f"{int(KAO.param['gainCL'] * 10):02d}"

    for r0 in range(len(r0_list)):
        KAO.atm.r0 = r0_list[r0]
        print(f"Running simulation for magnitude {m} with r0 {r0_list[r0]}")
        # Reset the system
        KAO.tel.resetOPD()
        KAO.dm.coefs = 0
        KAO.ttm.coefs = 0

        # Run the close loop simulation
        AO_output = close_loop(KAO)
        # Collect results
        res_seed[:, s] = AO_output['residual']
        SR_seed[:, s] = AO_output['SR']
        PSF_seed[:, :, s] = AO_output['PSF_LE']
        print(f"Completed simulation for magnitude {m} with gain {gainCL}")

    # Save results or perform analysis as needed
    name_res = f'{KAO.param["name"]}_NGS_magnitude_{m}_freq_{freq[f]}Hz_gainCL_{gain_str}_residualsOPD_seeing_study.npy'
    name_SR = f'{KAO.param["name"]}_NGS_magnitude_{m}_freq_{freq[f]}Hz_gainCL_{gain_str}_SR_gainCL_{gain_str}_seeing_study.npy'
    name_PSF = f'{KAO.param["name"]}_NGS_magnitude_{m}_freq_{freq[f]}Hz_gainCL_{gain_str}_PSF_gainCL_{gain_str}_seeing_study.npy'
    np.save(name_res, res_seed)
    np.save(name_SR, SR_seed)
    np.save(name_PSF, PSF_seed)
    
'''
#%% Statistic perf as a function of the loop frequency
# check param r0 = 16cm for that study
'''
mag = np.arange(6, 16, 2)
gain_list = [0.5,0.5,0.4,0.3,0.3]
r0 = f"{int(KAO.atm.r0 * 100)}"
freq = [2000,1000,700,500,200]
rec = param['type_rec']

if rec == 'keck':
    type_rec = 'keck_zonal'
else:
    type_rec = 'KL_control'

for m in range(len(mag)):
    KAO.param['magnitude_guide'] = mag[m]
    KAO.param['gainCL'] = gain_list[m]
    
    gain_str = f"{int(KAO.param['gainCL'] * 10):02d}"
    name_gen = f'{path}{KAO.param["name"]}_NGS_magnitude_{mag[m]}_M1_{param["M1_segments_pistons"]}_NCPA_{param["NCPA"]}_{type_rec}_r0_{r0}cm_gainCL_{gain_str}_AO_data_freq_study.npy'
    mag_results = {}
    
    for f in range(len(freq)):
        KAO.tel.samplingTime = 1/freq[f]
        print(f"Running simulation for magnitude {mag[m]} with freq {freq[f]}")
        # Reset the system (done in close_loop(AO_sys))
        KAO.tel.resetOPD()
        KAO.dm.coefs = 0
        KAO.ttm.coefs = 0

        # Run the close loop simulation
        AO_output = close_loop(KAO)
        mag_results[f] = AO_output
        print(f"Completed simulation for magnitude {mag[m]} with freq {freq[f]}")

    # Save results or perform analysis as needed
    np.save(name_gen, mag_results)
    print(f"Frequency Study Data saved for magnitude {mag[m]}")
'''

#%% Performances with NCPA and M1
mag = np.arange(6, 16, 2)
filename = f"{path}summary_results.txt"   # your text file
data = parse_text_file(filename)
rec = param['type_rec']
r0 = int(param['r0']*100) 
if rec == 'keck':
    type_rec = 'keck_zonal'
else:
    type_rec = 'KL_control'
    
name_gen = f'{path}{param["name"]}_NGS_magnitude_{mag[0]}_{mag[-1]}_M1_{param["M1_segments_pistons"]}_NCPA_{param["NCPA"]}_{type_rec}_r0_{r0}cm_performance.npy'

mag_results = {}

system = f"sh{param['nSubaperture']}"   # could also be "xin", "sh28", etc.
if param['nSubaperture'] == 56:
    gain = [0.5,0.5,0.3,0.4,0.4]
else:
    gain = [0.5,0.5,0.5,0.4,0.6]
    
param['print_display'] =False
param['display_loop']=False
param['Jitter']=True
for k in range(len(mag)):
    magnitude = mag[k]
    param_k = initializeParameterFile()
    param_k['magnitude_guide'] = magnitude
    param_k['gainCL'] = 0.5
    KAO_k = initialize_AO_hardware(param_k)
    KAO_k.param['Jitter']=False
    
    try:
        freq_result = get_best_value(data, system, r0, magnitude, "frequency")
        
        KAO_k.tel.samplingTime = 1/freq_result["Best_Frequency"]
        print(f"Updated frequency to {freq_result['Best_Frequency']} Hz from results file ")
    except ValueError as e:
        print("Gain lookup failed:", e)
    
    print(f"Running simulation for magnitude {magnitude}")
    # Reset the system (done in close_loop(AO_sys))
    KAO_k.tel.resetOPD()
    KAO_k.dm.coefs = 0
    KAO_k.ttm.coefs = 0
    
    # Run the close loop simulation
    AO_output = close_loop(KAO_k)
    mag_results[magnitude]=AO_output
# Save results or perform analysis as needed
np.save(name_gen, mag_results)














