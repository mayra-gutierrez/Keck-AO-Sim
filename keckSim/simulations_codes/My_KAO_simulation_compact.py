#%%
import time
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import hcipy
from hcipy import *

from astropy.io import fits
print(fits)

import os, sys
os.chdir("/home/lab/maygut/keckAOSim/keckSim")
sys.path.insert(0, os.path.join(os.getcwd(), "simulations_codes"))
os.environ["OOPAO_PATH"]="/home/lab/maygut/OOPAO"
sys.path.insert(0, "/home/lab/maygut/OOPAO")
#sys.path.insert(0, os.getcwd())

from OOPAO.calibration.getFittingError import *
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.OPD_map import OPD_map
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit

import importlib
from KAO_parameter_file import initializeParameterFile
from initialize_AO import initialize_AO_hardware
import close_loop_coro
from close_loop_coro import close_loop
importlib.reload(close_loop_coro)
import multiprocessing as mp
from multiprocessing import Pool
#mp.set_start_method("spawn", force=True)
from threadpoolctl import threadpool_limits

# %% -----------------------     Import system  ----------------------------------

param = initializeParameterFile()
KAO = initialize_AO_hardware(param)

M1_opd = KAO.opd_M1.OPD
ncpa = KAO.opd_ncpa.OPD
jitter_x = KAO.jitter.x

# plots the OPD map
'''
plt.figure(), plt.imshow(KAO.opd_M1.OPD), plt.colorbar(), plt.title('M1 OPD'), plt.show()
plt.figure(), plt.imshow(KAO.opd_ncpa.OPD), plt.colorbar(), plt.title('NCPA OPD'), plt.show()
plt.figure(), plt.plot(jitter_x), plt.title('Jitter amplitude'), plt.show()

plt.figure()
plt.imshow(np.reshape(np.sum(KAO.dm.modes**5,axis=1),[KAO.tel.resolution,KAO.tel.resolution]).T +
            KAO.tel.pupil,extent=[-KAO.tel.D/2,KAO.tel.D/2,-KAO.tel.D/2,KAO.tel.D/2])
plt.plot(KAO.dm.coordinates[:,0],KAO.dm.coordinates[:,1],'rx')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates keck')
plt.colorbar()
plt.show()

plt.figure(), plt.plot(KAO.calib_zonal.eigenValues,'rx'), plt.ylabel('eigenValues zonal'), plt.show(block=False)
plt.figure(), plt.imshow(KAO.calib_zonal.D), plt.colorbar(), plt.title('Zonal Imat'), plt.show(block=False)

KAO.tel+KAO.atm
KAO.ngs * KAO.tel * KAO.wfs
plt.figure(), plt.imshow(KAO.wfs.signal_2D), plt.colorbar(), plt.title('Slopes'), plt.show(block=False)

print(f'Incoming photon on WFS {np.sum(KAO.wfs.signal)}')
# %% -----------------------     plot PSF  ----------------------------------

# diff at the science wavelength
LD = KAO.science.wavelength/KAO.tel.D * 206265 * 1000 # in mas

PSF_diff = KAO.PSF_diff
PSF_ncpa = KAO.PSF_ncpa
size_psf = PSF_diff.shape[0]//2
wind = 100
plt.figure(), plt.imshow(PSF_diff[size_psf-wind:size_psf+wind,size_psf-wind:size_psf+wind]**0.2), plt.colorbar(), plt.title('Diffraction limited PSF'), plt.show()
plt.figure(), plt.imshow(PSF_ncpa[size_psf-wind:size_psf+wind,size_psf-wind:size_psf+wind]**0.2), plt.colorbar(), plt.title('NCPA PSF'), plt.show()
'''

# %% -----------------------     AO loop  ----------------------------------
# change parameter before launching CL example:
# KAO.param['magnitude_guide] = 10
#KAO.param['gainCL'] = 0.5
#KAO.param['type_rec'] = 'modal'
KAO.param['print_display'] =True
KAO.param['nLoop']=20000

def run_loop(Name):

    start_time = datetime.now()
    with open("run_log.txt", "a") as log:
        log.write(f"[{start_time:%Y-%m-%d %H:%M:%S}] START process {Name}\n")

    print(f"[{start_time:%Y-%m-%d %H:%M:%S}] Starting process {Name}")
    AO_output_modal = close_loop(KAO,Name)
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

N = 15

with Pool(N) as p:
    p.map(run_loop_single_threaded, range(N))

'''
if __name__ == "__main__":
    
    N = 10

    def run_loop_single_threaded(x):
        with threadpool_limits(limits=1):
            run_loop(x)

    with mp.Pool(N) as p:
        p.map(run_loop_single_threaded, range(N))

'''

'''
KAO.param['type_rec'] = 'keck'
AO_output_keck = close_loop(KAO)
SR = AO_output_keck['SR']
res = AO_output_keck['residual']
PSF_LE = AO_output_keck['PSF_LE']
'''
