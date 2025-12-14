#%%
import time
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import welch
import os, sys
os.chdir("/Users/mayragutierrez/home/lab/maygut/keckAOSim/keckSim")
sys.path.insert(0, os.path.join(os.getcwd(), "simulations_codes"))
os.environ["OOPAO_PATH"]="/Users/mayragutierrez/home/lab/maygut/OOPAO"
sys.path.insert(0, "/Users/mayragutierrez/home/lab/maygut/OOPAO")
#from simulations_codes.KAO_parameter_file import initializeParameterFile
#from simulations_codes.initialize_AO import initialize_AO_hardware
from KAO_parameter_file import initializeParameterFile
from initialize_AO import initialize_AO_hardware

import importlib
import close_loop_coro
from close_loop_coro import close_loop
importlib.reload(close_loop_coro)
import multiprocessing as mp
from multiprocessing import Pool
#mp.set_start_method("spawn", force=True)
from threadpoolctl import threadpool_limits

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
KAO.param['print_display'] = True
KAO.param['display_loop'] = False
#KAO.param['nLoop']= 3000 #20000


def run_loop(Run):

    start_time = datetime.now()
    with open("run_log.txt", "a") as log:
        log.write(f"[{start_time:%Y-%m-%d %H:%M:%S}] START process {Run}\n")

    print(f"[{start_time:%Y-%m-%d %H:%M:%S}] Starting process {Run}")
    AO_output_modal = close_loop(KAO, ZWFS_param, Run)
    SR_mode = AO_output_modal['SR']
    res_mode = AO_output_modal['residual']
    PSF_LE_mode = AO_output_modal['PSF_LE']
    
    end_time = datetime.now()
    elapsed = end_time - start_time

    with open("run_log.txt", "a") as log:
        log.write(f"[{end_time:%Y-%m-%d %H:%M:%S}] END process {Name} | Elapsed: {elapsed}\n")

    print(f"[{end_time:%Y-%m-%d %H:%M:%S}] Finished process {Name} (elapsed: {elapsed})")

def run_loop_single_threaded(x):
    with threadpool_limits(limits=1):
        run_loop(x)

N = 5

with Pool(N) as p:
    p.map(run_loop_single_threaded, range(N))
    

'''
# %% -----------------------     plots  ----------------------------------
plt.figure()
plt.plot(AO_output['SR'],'r')
plt.legend()
plt.ylabel('SR')
plt.xlabel('Iterations')

#%%PSD Jitter

f, P = welch(np.array(AO_output['jitter']),fs=KAO.param['Jitter_nFrames'])
plt.figure(), plt.semilogy(f,P), plt.xlabel('Frequency [Hz]'), plt.ylabel('PSD [nm**2/Hz]'), plt.xlim(0,500)
'''


