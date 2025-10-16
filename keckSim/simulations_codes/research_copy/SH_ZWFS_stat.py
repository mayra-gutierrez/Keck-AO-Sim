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

parser = argparse.ArgumentParser()
parser.add_argument("--M1",  action="store_true", help="Enable M1 OPD")
parser.add_argument("--NCPA", action="store_true", help="Enable NCPA")
args = parser.parse_args()


param = initializeParameterFile()
param['M1_segments_pistons']=args.M1
param['NCPA']=args.NCPA

KAO = initialize_AO_hardware(param)
path = '/home/mcisse/keckAOSim/keckSim/data/'

# %% Magnitude loop
'''
mag = np.arange(6, 14, 2)

gain_list = [0.3,0.4,0.5,0.6]
r0_list = [0.35,0.16,0.12]
seeds = len(gain_list)
res_seed = np.zeros((KAO.param['nLoop'],seeds))
SR_seed = np.zeros((KAO.param['nLoop'],seeds))
PSF_seed = np.zeros((KAO.param['scienceDetectorRes'], KAO.param['scienceDetectorRes'], seeds))
r0 = f"{int(KAO.atm.r0 * 100)}"


# Loop over different magnitudes/gain
ZWFS_param = {'activate': False, 'max_iter':10,'gain':1,'mean_img':5,'freq':50,'subGZ':4,'maxZ':1,'maxGlobalZ':4}
for r0 in range(len(r0_list)):
    KAO.atm.r0 = r0_list[r0]
    
    for m in mag:
        KAO.param['magnitude_guide'] = m
        results = {}
        name_gen = f'{path}{KAO.param["name"]}_NGS_magnitude_{m}_M1_{param["M1_segments_pistons"]}_NCPA_{param["NCPA"]}_r0_{int(r0_list[r0]*100)}cm_AO_data_gain_study.npy'

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
#%% Study of the image averaging for the ZWFS
M1_OPD = KAO.opd_M1.OPD
ncpa = KAO.opd_ncpa.OPD
r0_list = [0.35,0.16,0.12]
gain_list = [0.4,0.5,0.6] # to be defined with the study above

freq_zwfs = np.arange(10,110,10)
ZWFS_param = {'activate': True, 'max_iter':10,'gain':1,'mean_img':5,'freq':50,'subGZ':4,'maxZ':1,'maxGlobalZ':4}
sys_ = f'{path}{KAO.param["name"]}_NGS_magnitude_{KAO.param["magnitude_guide"]}_'

# Dictionary to hold all results

for k in range(len(r0_list)):
    KAO.atm.r0=r0_list[k]
    KAO.param['gainCL'] = gain_list[k]
    
    gain_str = f"{int(gain_list[k] * 10):02d}"
    name_gen = f'{sys_}gainCL{gain_str}_r0_{int(r0_list[k]*100)}cm_NCPA_{param["NCPA"]}'
    
    print(f"Running simulation for magnitude {KAO.param['magnitude_guide']} with r0 {r0_list[k]}")
    r0_results = {}
    
    for f in range(len(freq_zwfs)):
        ZWFS_param['freq']=freq_zwfs[f]
        print(f"Running simulation for magnitude {KAO.param['magnitude_guide']} r0 {r0_list[k]} with averaging at {freq_zwfs[f]}Hz")
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
    name_AO_data = os.path.join(path, f"{name_gen}AO_data_frequency_study.npy")
    np.save(name_AO_data, r0_results)
    


