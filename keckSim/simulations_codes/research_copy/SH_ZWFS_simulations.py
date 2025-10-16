#%%
import time
import matplotlib.pyplot as plt
import numpy as np
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

#%% Import KAO system
param = initializeParameterFile()
KAO = initialize_AO_hardware(param)

M1_opd = KAO.opd_M1.OPD
ncpa = KAO.opd_ncpa.OPD
jitter_x = KAO.jitter.x

# plot the OPD map
plt.figure(), plt.imshow(M1_opd), plt.colorbar(), plt.title('M1 OPD'), plt.show(block=False)
plt.figure(), plt.imshow(ncpa), plt.colorbar(), plt.title('NCPA OPD'), plt.show(block=False)
plt.figure(), plt.plot(jitter_x), plt.title('Jitter amplitude'), plt.show(block=False)


#%% test mag
'''
from OOPAO.tools.displayTools import cl_plot
from OOPAO.Source import Source
from OOPAO.Detector import Detector as Det_OOPAO

src4 = Source('K',4)
src10 = Source('H',10)
KAO.science.magnitude = 15
niter =200
flux4 = np.zeros(niter)
flux10 = np.zeros(niter)

cam = Det_OOPAO(nRes=param['scienceDetectorRes'],
                     integrationTime=param['science_integrationTime'],
                     bits=14,
                     sensor='CCD',
                     QE=param['science_QE'],
                     binning=param['science_binning'],
                     psf_sampling=param['science_psf_sampling'],
                     darkCurrent=param['science_darkCurrent'],
                     readoutNoise=param['science_ron'],
                     photonNoise=param['science_photonNoise'])
                     
cam2 = Det_OOPAO(nRes=param['scienceDetectorRes'],
                     integrationTime=param['science_integrationTime'],
                     bits=14,
                     sensor='CCD',
                     QE=param['science_QE'],
                     binning=param['science_binning'],
                     psf_sampling=param['science_psf_sampling'],
                     darkCurrent=param['science_darkCurrent'],
                     readoutNoise=param['science_ron'],
                     photonNoise=param['science_photonNoise'])    
               

for i in range(niter):
    KAO.atm.generateNewPhaseScreen(i)
    KAO.tel+KAO.atm
    KAO.ngs*KAO.tel
    KAO.science*KAO.tel*KAO.science_detector
    #src10*KAO.tel*cam2
    flux4[i]=np.sum(KAO.science_detector.frame)
    #flux10[i]=np.sum(cam2.frame[:])

plt.figure()
plt.plot(np.arange(niter),flux4,'r',label='Mag15')
plt.plot(np.arange(niter),flux1,'g',label='Mag0')
#plt.plot(np.arange(niter), flux4_10,'k',label='mag4')
plt.legend()
plt.show(block=False)
'''
#%% compact AO loop

ZWFS_param = {'activate': False, 'max_iter':10,'gain':1,'mean_img':5,'freq':50,'subGZ':4,'maxZ':1,'maxGlobalZ':4}
KAO.param['print_display']=True
#KAO.wfs.cam.gain = 50
KAO.param['nLoop']=100

AO_output4 = close_loop(KAO,ZWFS_param, atm_seed=10)

KAO.param['science_magnitude']=15
AO_output15 = close_loop(KAO,ZWFS_param, atm_seed=10)

# %%
