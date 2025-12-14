#%%
import time
import sys
import matplotlib
matplotlib.use("Agg")
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
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from simulations_codes.research_copy.KAO_parameter_file_research_copy import initializeParameterFile
from simulations_codes.research_copy.initialize_AO_research_copy import initialize_AO_hardware
from simulations_codes.research_copy.close_loop_ZWFS import close_loop

from simulations_codes.ZWFS_toolbox.ZWFS_tools import *
from simulations_codes.ZWFS_toolbox.tools import *
from simulations_codes.ZWFS_toolbox.wfSensors import *
from simulations_codes.read_ao_results_file import parse_text_file, get_best_value

parser = argparse.ArgumentParser()
parser.add_argument("--M1",  action="store_true", help="Enable M1 OPD")
parser.add_argument("--NCPA", action="store_true", help="Enable NCPA")
parser.add_argument("--offset", type=float, help="Set Pure NCPA amplitude in nm (e.g., 50 for 50 nm)")
args = parser.parse_args()


param = initializeParameterFile()
param['M1_segments_pistons']=args.M1
param['NCPA']=args.NCPA
# Update Pure NCPA amplitude only if user provided --offset
if args.offset is not None:
    param['Pure_NCPA_amplitude'] = args.offset * 1e-9  # convert from nm to meters
    offset = True
    print('NCPA offset')
else:
    param['Pure_NCPA_amplitude'] = 0  
    offset = False


KAO = initialize_AO_hardware(param)
path = '/home/mcisse/keckAOSim/keckSim/data/'

# %% Magnitude loop
'''
mag = np.arange(6, 16, 2)
gain_list = [0.3,0.4,0.5,0.6]
freq = [2000,2000,2000,1000,700]
r0_list = [0.16]#[0.35,0.16,0.12]
seeds = len(freq)
res_seed = np.zeros((KAO.param['nLoop'],seeds))
SR_seed = np.zeros((KAO.param['nLoop'],seeds))
PSF_seed = np.zeros((KAO.param['scienceDetectorRes'], KAO.param['scienceDetectorRes'], seeds))
r0 = f"{int(KAO.atm.r0 * 100)}"

# Loop over different magnitudes/gain
ZWFS_param = {'activate': False, 'max_CLiter':5,'gain':1,'n_average':4,'avg_time':1,'freq':5,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

for r0 in range(len(r0_list)):
    KAO.atm.r0 = r0_list[r0]
    
    for m in range(len(mag)):
        KAO.param['magnitude_guide'] = mag[m]
        KAO.tel.samplingTime = 1/freq[m]
        KAO.param['Jitter'] = False
        results = {}
        name_gen = f'{path}{KAO.param["name"]}_NGS_magnitude_{mag[m]}_M1_{param["M1_segments_pistons"]}_NCPA_{param["NCPA"]}_r0_{int(r0_list[r0]*100)}cm_alpha_{param["alpha"]}_AO_data_gain_study.npy'

        for s in range(len(gain_list)):
            gainCL = gain_list[s]
            print(f"Running simulation for magnitude {m} with gain {gainCL}")
            KAO.param['gainCL'] = gainCL
            # Reset the system
            KAO.tel.resetOPD()
            KAO.dm.coefs = 0
            KAO.ttm.coefs = 0

            # Run the close loop simulation 
            AO_output = close_loop(KAO,ZWFS_param)
            
            # Collect results
            results[s] = AO_output
            
        # Save results or perform analysis as needed
        np.save(name_gen, results)
        print(f"Data saved for magnitude {m} and r0 {int(r0_list[r0]*100)}cm")

'''
#%% frequency study
'''
mag = np.arange(6, 16, 2)
gain_list = [0.5,0.5,0.4,0.3,0.3]
r0 = f"{int(KAO.atm.r0 * 100)}"
freq = [2000,1000,700,500,200]
rec = param['type_rec']
ZWFS_param = {'activate': False, 'max_iter':10,'gain':1,'mean_img':5,'freq':50,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

if rec == 'keck':
    type_rec = 'keck_zonal'
else:
    type_rec = 'KL_control'

for m in range(len(mag)):
    KAO.param['magnitude_guide'] = mag[m]
    KAO.param['gainCL'] = gain_list[m]
    
    gain_str = f"{int(KAO.param['gainCL'] * 10):02d}"
    name_gen = f'{path}{KAO.param["name"]}_NGS_magnitude_{mag[m]}_M1_{param["M1_segments_pistons"]}_NCPA_{param["NCPA"]}_{type_rec}_r0_{r0}cm_gainCL_{gain_str}_alpha_{param["alpha"]}_AO_data_freq_study.npy'
    mag_results = {}
    
    for f in range(len(freq)):
        KAO.tel.samplingTime = 1/freq[f]
        print(f"Running simulation for magnitude {mag[m]} with freq {freq[f]}")
        # Reset the system (done in close_loop(AO_sys))
        KAO.tel.resetOPD()
        KAO.dm.coefs = 0
        KAO.ttm.coefs = 0

        # Run the close loop simulation
        AO_output = close_loop(KAO,ZWFS_param)
        mag_results[f] = AO_output
        print(f"Completed simulation for magnitude {mag[m]} with freq {freq[f]}")

    # Save results or perform analysis as needed
    np.save(name_gen, mag_results)
    print(f"Frequency Study Data saved for magnitude {mag[m]}")
'''
#%% Study of M1 Amplitude
'''
mag = [8]
amp_NCPA = np.array([25,50,75,100,125,150,200])
amp_M1 = [175]#np.array([50,75,100,125,150])
M1_OPD = KAO.opd_M1.OPD
N1_OPD = KAO.opd_ncpa.OPD

study = 'M1_NCPA_study'
filename = f"{path}summary_results.txt"   # your text file
data = parse_text_file(filename)

system = f"sh{param['nSubaperture']}"   # could also be "xin", "sh28", etc.
r0 = f"{int(KAO.atm.r0 * 100)}"
magnitude = KAO.param['magnitude_guide']      # guide star magnitude

ZWFS_param = {'activate': False, 'max_iter':10,'gain':1,'mean_img':5,'freq':50,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

# Reconstruction type
type_rec = "keck_zonal" if param['type_rec'] == "keck" else "KL_control"
r0_ = param['r0']*100
for m in range(len(mag)):
    gain_result = get_best_value(data, system, r0_, mag[m], "gain")
    freq_result = get_best_value(data, system, r0_, mag[m], "frequency")
    KAO.param['gainCL'] = gain_result["Best_Gain"]
    KAO.tel.samplingTime = 1/freq_result["Best_Frequency"]
    KAO.param['magnitude_guide'] = mag[m]
    print(f"Updated closed-loop gain = {KAO.param['gainCL']} and frequency to {freq_result['Best_Frequency']} Hz from results file")
    
    name_gen = f'{path}{KAO.param["name"]}_NGS_magnitude_{mag[m]}_M1_{amp_M1[0]}_NCPA_{param["NCPA"]}_{type_rec}_r0_{r0}cm_AO_data_{study}.npy'
    
    # Storage: nested dict
    results = {
        "magnitude": mag[m],
        "r0_cm": r0,
        "type_rec": type_rec,
        "gainCL": KAO.param["gainCL"],
        "frequency": freq_result["Best_Frequency"],
        "data": {},  # Nested results
    }

    for amp_i in amp_M1:
    
        results["data"][amp_i] = {}
        M1_i = M1_OPD/KAO.param['M1_OPD_amplitude']*(amp_i*1e-9)
        KAO.opd_M1.OPD = M1_i
        
        for amp_j in amp_NCPA:
        
            Nc_i = N1_OPD/KAO.param['NCPA_amplitude']*(amp_j*1e-9)
            KAO.opd_ncpa.OPD = Nc_i
            
            # New Cog 
            KAO.tel.isPaired = True # DO NOT CHANGE THIS
            KAO.tel.resetOPD()
            KAO.ttm.coefs = 0
            KAO.dm.coefs = 0
            KAO.science * KAO.tel * KAO.opd_ncpa * KAO.ttm * KAO.dm * KAO.wfs 
            ncpa_slopes = KAO.wfs.signal_2D
            ncpa_cog = ncpa_slopes* KAO.wfs.slopes_units + KAO.cog_zeros
            KAO.cog_ncpa = ncpa_cog

            print(f"Running: mag={mag[m]}, M1={amp_i}nm, NCPA={amp_j}nm")
            # Reset the system (done in close_loop(AO_sys))
            KAO.tel.resetOPD()
            KAO.dm.coefs = 0
            KAO.ttm.coefs = 0

            # Run the close loop simulation
            AO_output = close_loop(KAO,ZWFS_param)
            results["data"][amp_i][amp_j] = AO_output
            print(f"Completed: mag={mag[m]}, M1={amp_i}nm, NCPA={amp_j}nm")
    
    # Save results or perform analysis as needed
    np.save(name_gen, results)
    print(f"Frequency Study Data saved for magnitude {mag[m]}")        
'''
#%% Study of the image averaging for the ZWFS
'''
M1_OPD = KAO.opd_M1.OPD
ncpa = KAO.opd_ncpa.OPD
r0_list = [0.16]
gain_list = [0.3] # to be defined with the study above

freq_zwfs = np.arange(2,6)
expo_zwfs = np.arange(1,32,10)
ZWFS_param = {'activate': True, 'max_CLiter':5,'gain':1,'n_average':5,'avg_time':1,'freq':4,'subGZ':4,'maxZ':1,'maxGlobalZ':4}
KAO.param['Jitter'] = False

for k in range(len(r0_list)):
    KAO.atm.r0=r0_list[k]
    KAO.param['gainCL'] = gain_list[k]

    name_gen = f'{path}{KAO.param["name"]}_NGS_magnitude_{8}_M1_{param["M1_segments_pistons"]}_NCPA_{param["NCPA"]}_r0_{int(r0_list[k]*100)}cm_alpha_{param["alpha"]}_ZWFS_fps{ZWFS_param["freq"]}_AO_data_ZWFS_exposure_study.npy'

    
    print(f"Running simulation for magnitude {KAO.param['magnitude_guide']} with r0 {r0_list[k]}")
    r0_results = {}
    
    for f in range(len(expo_zwfs)):
        ZWFS_param['avg_time']=expo_zwfs[f]
        print(f"Running simulation for magnitude {KAO.param['magnitude_guide']} r0 {r0_list[k]} with exposure at {expo_zwfs[f]}s")
        # Reset the system
        KAO.tel.resetOPD()
        KAO.dm.coefs = 0
        KAO.ttm.coefs = 0
        KAO.opd_M1.OPD = M1_OPD # need to reset the M1 OPD
        KAO.opd_ncpa.OPD = ncpa 

        # Run the close loop simulation 
        AO_output = close_loop(KAO,ZWFS_param)
        r0_results[f] = AO_output
        
    # Save results or perform analysis as needed        
    np.save(name_gen, r0_results)
    
'''
#'''
M1_OPD = KAO.opd_M1.OPD
ncpa = KAO.opd_ncpa.OPD
off = KAO.opd_offset.OPD
ZWFS_param = {'activate': True, 'max_CLiter':5,'gain':1,'n_average':5,'avg_time':4,'freq':4,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

KAO.opd_M1.OPD = M1_OPD
KAO.param['Jitter']=False
name_gen = f'{path}New_{KAO.param["name"]}_NGS_magnitude_{param["magnitude_guide"]}_M1_{param["M1_segments_pistons"]}_NCPA_{param["NCPA"]}_r0_{int(KAO.atm.r0*100)}cm_alpha_1e13_ZWFS_fps{ZWFS_param["freq"]}_exposure_{ZWFS_param["avg_time"]}s_offset_{offset}.npy'

AO_output = close_loop(KAO,ZWFS_param)

AO_output['Input_M1'] = M1_OPD
AO_output['Offset_seg'] = KAO.offset_seg_proj
AO_output['Offset'] = KAO.opd_offset.OPD
np.save(name_gen,AO_output)

#'''













