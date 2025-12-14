#%%
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from hcipy import *

from OOPAO.calibration.getFittingError import *
from OOPAO.OPD_map import OPD_map
from OOPAO.Zernike import Zernike

#sys.path.append('/home/mcisse/PycharmProjects/data_pyao/')
import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim/")
sys.path.insert(0, os.getcwd())

from simulations_codes.research_copy.KAO_parameter_file_research_copy import initializeParameterFile
from simulations_codes.research_copy.initialize_AO_research_copy import initialize_AO_hardware
from simulations_codes.research_copy.close_loop_ZWFS import close_loop

from simulations_codes.ZWFS_toolbox.ZWFS_tools import *
from simulations_codes.ZWFS_toolbox.tools import *
from simulations_codes.ZWFS_toolbox.wfSensors import *
from simulations_codes.read_ao_results_file import parse_text_file, get_best_value

path = '/home/mcisse/keckAOSim/keckSim/data/'

#%% Import KAO system
param = initializeParameterFile()
KAO = initialize_AO_hardware(param)

M1_opd = KAO.opd_M1.OPD
ncpa = KAO.opd_ncpa.OPD
jitter_x = KAO.jitter[0]

#plots aberrations
plt.figure(), plt.imshow(KAO.PSF_ncpa**0.2), plt.colorbar(), plt.title('PSF NCPA'), plt.show(block=False)
plt.figure(), plt.imshow(ncpa*1e9), plt.colorbar(), plt.title('NCPA OPD [nm]'), plt.show(block=False)
plt.figure(), plt.imshow(M1_opd*1e9), plt.colorbar(), plt.title('M1 OPD [nm]'), plt.show(block=False)
plt.figure(), plt.imshow((M1_opd+KAO.offset_seg_proj)*KAO.tel.pupil*1e9), plt.colorbar(), plt.title('M1 OPD + offset [nm]'), plt.show(block=False)

#%% compact AO loop

'''
# Load AO study results once
path = '/home/mcisse/keckAOSim/keckSim/data/'
filename = f"{path}summary_results.txt"   # your text file
data = parse_text_file(filename)

# Define the system you are using
system = f"sh{param['nSubaperture']}"   # could also be "xin", "sh28", etc.
r0 = param['r0']*100            # cm
magnitude = KAO.param['magnitude_guide']      # guide star magnitude

try:
    gain_result = get_best_value(data, system, r0, magnitude, "gain")
    3#freq_result = get_best_value(data, system, r0, magnitude, "frequency")
    KAO.param['gainCL'] = gain_result["Best_Gain"]
    #KAO.tel.samplingTime = 1/freq_result["Best_Frequency"]
    print(f"Updated closed-loop gain = {KAO.param['gainCL']} ")#and frequency to {freq_result['Best_Frequency']} Hz from results file")
except ValueError as e:
    print("Gain lookup failed:", e)
'''

# how to change parameters on the fly
#KAO.wfs.cam.gain = 50
#KAO.param['NCPA'] = False

ZWFS_param = {'activate': False, 'max_CLiter':5,'gain':1,'n_average':5,'avg_time':4,'freq':4,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

KAO.param['print_display']=True
KAO.param['nLoop'] = 400
KAO.param['display_loop'] = True    
KAO.param['Jitter'] = False
KAO.param['NCPA'] = False
KAO.param['gainCL'] = 0.2

# AO loop
AO_output = close_loop(KAO,ZWFS_param)
'''
#plots
plt.figure()
plt.plot(AO_output['SR'],'r')
plt.legend()
plt.ylabel('SR')
plt.xlabel('Iterations')
plt.show(block=False)
psf_list = np.array(AO_output['PSF_batches'])
psf_beg = psf_list[1,:,:,:]
for i in range(psf_beg.shape[0]):
            psf = psf_beg[i,:,:]
            fig = plt.figure(); axes  = plt.gca()
            fig.suptitle(f'PSF {i}', fontsize=14)
            axes.imshow(psf**0.2, cmap='viridis')
            axes.axis('off')
            plt.show(block=False)

plt.figure()
plt.plot(AO_output['residual'],'r')
plt.legend()
plt.ylabel('SR')
plt.xlabel('Iterations')
plt.show(block=False)

if ZWFS_param['activate']:
    plt.figure()
    plt.plot(AO_output['M1_OPD'],'ro')
    plt.legend()
    plt.ylabel('M1 OPD nm RMS')
    plt.xlabel('Iterations')
    plt.show(block=False)

PSF1 = AO_output['PSF_LE']
plt.figure(), plt.imshow(PSF1**0.2), plt.colorbar(), plt.show(block=False)
'''


