#%%
import time
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import welch
import os, sys
os.chdir("/Users/mayragutierrez/home/lab/maygut/keckAOSim2/keckSim/")
sys.path.insert(0, os.getcwd())
from simulations_codes.KAO_parameter_file import initializeParameterFile
from simulations_codes.initialize_AO import initialize_AO_hardware
from simulations_codes.close_loop import close_loop

plt.ion()
# %% -----------------------     Import system  ----------------------------------
path = '/home/mcisse/keckAOSim/keckSim/data/'
param = initializeParameterFile()
KAO = initialize_AO_hardware(param)
ZWFS_param = {'activate': False, 'max_CLiter':5,'gain':1,'n_average':5,'avg_time':4,'freq':4,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

# %% -----------------------     AO loop  ----------------------------------
        
# change parameter before launching CL example:
# KAO.param['magnitude_guide'] = 10
# KAO.param['type_rec'] = 'modal' #'zonal'
KAO.param['gainCL'] = 0.3
KAO.param['print_display'] =True
KAO.param['display_loop']=True

AO_output = close_loop(KAO,ZWFS_param, atm_seed=np.random)

# %% -----------------------     plots  ----------------------------------
plt.figure()
plt.plot(AO_output['SR'],'r')
plt.legend()
plt.ylabel('SR')
plt.xlabel('Iterations')

#%%PSD Jitter
'''
f, P = welch(np.array(AO_output['jitter']),fs=KAO.param['Jitter_nFrames'])
plt.figure(), plt.semilogy(f,P), plt.xlabel('Frequency [Hz]'), plt.ylabel('PSD [nm**2/Hz]'), plt.xlim(0,500)
'''

