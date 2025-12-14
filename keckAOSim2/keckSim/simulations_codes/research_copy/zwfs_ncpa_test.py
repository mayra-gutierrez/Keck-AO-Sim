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
from OOPAO.Detector import Detector as Det_OOPAO
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
ZWFS_param = {'activate': True, 'max_CLiter':5,'gain':1,'n_average':5,'avg_time':5,'freq':4,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

M1_opd = KAO.opd_M1.OPD
offset = KAO.opd_ncpa.OPD

NIRC2 = Det_OOPAO(nRes=param['scienceDetectorRes'],
                     integrationTime=param['science_integrationTime'],
                     bits=14,
                     sensor='CCD',
                     QE=param['science_QE'],
                     binning=param['science_binning'],
                     psf_sampling=param['science_psf_sampling'],
                     darkCurrent=param['science_darkCurrent'],
                     readoutNoise=param['science_ron'],
                     photonNoise=param['science_photonNoise'])
    
Rec = KAO.keck_reconstructor
maxZ = ZWFS_param['maxZ']
subGZ = ZWFS_param['subGZ']
maxGlobalZ = ZWFS_param['maxGlobalZ']
seg_vect2D = KAO.seg2D
segments = KAO.seg1D
proj_seg = KAO.proj_seg
Nseg = param['numberSegments']
pupil_spider = KAO.tel.pupil
z2p, p2z = zernikeBasis_nonCirc(maxGlobalZ, pupil_spider)

Z = Zernike(KAO.tel, param['nb_Zpolynomials'])
Z.computeZernike(KAO.tel)
mode = Z.modesFullRes[:,:,5]
opd_screen = mode * KAO.ngs.wavelength/(2*np.pi)
amp_screen = np.std(opd_screen[np.where(KAO.tel.pupil > 0)])
com = opd_screen/amp_screen * 50*1e-9  
com_off = OPD_map(telescope=KAO.tel)
com_off.OPD = com *KAO.tel.pupil
com_amp = float(np.std(com_off.OPD[np.where(KAO.tel.pupil > 0)])) * 1e9

com_seg_coefs = np.dot(com_off.OPD.flatten(),proj_seg)
com_seg = np.dot(com_seg_coefs,segments).reshape(param['resolution'],param['resolution'])
com_amp_seg = float(np.std(com_seg[np.where(KAO.tel.pupil > 0)])) * 1e9

#proj offset on seg
offset_seg_coefs = np.dot(offset.flatten(),proj_seg)
offset_seg = np.dot(offset_seg_coefs,segments).reshape(param['resolution'],param['resolution'])

gainCL = param['gainCL']
gainTTM = gainCL
leak = param['leak']
            
KAO.wfs.is_geometric = False
KAO.tel.isPaired = True 
KAO.tel.resetOPD()
KAO.dm.coefs = 0
KAO.ttm.coefs = 0
KAO.opd_M1.OPD = M1_opd
KAO.wfs.reference_slopes_maps = KAO.cog_ncpa #applied ncpa
KAO.opd_ncpa.OPD = -offset
KAO.ngs * KAO.tel
KAO.tel * KAO.ttm * KAO.dm * KAO.wfs
m1_amp = []
m1_z = float(np.std(KAO.opd_M1.OPD[np.where(KAO.tel.pupil > 0)])) * 1e9
m1_amp.append(m1_z)
total_M1_applied = 0
niter = 100
psf = []
psf0 = 0
# initial PSF

for k in range(10):
    KAO.tel.resetOPD()
    KAO.ngs * KAO.tel * KAO.opd_M1 * KAO.ttm * KAO.dm * KAO.wfs  
    KAO.science * KAO.tel * com_off* NIRC2 #PSF for the added aberration only
    psf0 += NIRC2.frame[:]
psf0 /=10 

res_phi = float(np.var(KAO.tel.src.phase[np.where(KAO.tel.pupil > 0)]))
SR_in = np.exp(-res_phi**2)

NIRC2_loop = Det_OOPAO(nRes=param['scienceDetectorRes'],
                     integrationTime=param['science_integrationTime'],
                     bits=14,
                     sensor='CCD',
                     QE=param['science_QE'],
                     binning=param['science_binning'],
                     psf_sampling=param['science_psf_sampling'],
                     darkCurrent=param['science_darkCurrent'],
                     readoutNoise=param['science_ron'],
                     photonNoise=param['science_photonNoise'])
                     
for k in range(niter):
    KAO.tel.resetOPD()
    KAO.ngs * KAO.tel * KAO.opd_M1 * KAO.ttm * KAO.dm * KAO.wfs  
    wfsSignal = KAO.wfs.signal  
    
    KAO.science * KAO.tel *KAO.opd_ncpa * com_off *NIRC2_loop 
        
    psf.append(NIRC2_loop.frame[:])

    tel_OPD = KAO.tel.OPD
    phase = tel_OPD*2*np.pi/KAO.science.wavelength
    
    img_ao = KAO.zwfs.getImageSimu(KAO.tel.src.phase)
    _, opd_wttf, seg_Zmode_coeffs = ReconPhase_Segments(img_ao,z2p, p2z,pupil_spider,
                                            KAO.matching_inds,seg_vect2D,
                                            KAO.zwfs,maxZ=maxZ,maxSeg=Nseg, subGZ=subGZ)
                
    coef_ucsc = np.squeeze(seg_Zmode_coeffs)*1e-9
    seg_rec_ = np.dot(coef_ucsc,segments).reshape(param['resolution'],param['resolution'])
    seg_rec_ = seg_rec_ * KAO.tel.pupil
    seg_rec = seg_rec_ - np.mean(seg_rec_[np.where(KAO.tel.pupil>0)])  
    total_M1_applied += seg_rec
              
    KAO.opd_M1.OPD = KAO.opd_M1.OPD - seg_rec
    m1_z = float(np.std(KAO.opd_M1.OPD[np.where(KAO.tel.pupil > 0)])) * 1e9
    m1_amp.append(m1_z)
    command = Rec @ wfsSignal
    KAO.dm.coefs = leak*KAO.dm.coefs - gainCL * command[:KAO.dm.nValidAct]
    KAO.ttm.coefs = KAO.ttm.coefs - gainTTM * command[KAO.dm.nValidAct:]

psf = np.array(psf)
LE_PSF = np.mean(psf[niter-10:,:,:],axis=0)

residual = float(np.std(KAO.tel.OPD[np.where(KAO.tel.pupil > 0)])) * 1e9
res_phi = float(np.var(KAO.tel.src.phase[np.where(KAO.tel.pupil > 0)]))
SR = np.exp(-res_phi**2)

delta_mirror = KAO.opd_M1.OPD+KAO.dm.OPD
# plots
n1 = f'M1_{round(m1_amp[0])}nmRMS'
n2 = f'NCPA_{round(com_amp)}nmRMS'
 
plt.figure()
plt.imshow(KAO.PSF_ncpa**0.2)
plt.colorbar(), plt.title('PSF Offset')
plt.show(block=False)

plt.figure()
plt.imshow(psf0**0.2)
plt.colorbar(), plt.title('PSF NCPA')
plt.show(block=False)

plt.figure()
plt.imshow(offset*1e9)
plt.colorbar()
plt.title('Offset OPD [nm]')
plt.show(block=False)
#plt.savefig(f'{path}{n1}_{n2}_phase_map_cog.png', dpi=300, bbox_inches='tight')  

plt.figure()
plt.imshow(offset_seg*KAO.tel.pupil*1e9)
plt.colorbar(),plt.title('Offset projected on segments [nm]')
plt.show(block=False)
#plt.savefig(f'{path}{n1}_{n2}_phase_map_cog_proj_seg.png', dpi=300, bbox_inches='tight')  

if m1_amp[0]>0:
    plt.figure()
    plt.imshow(M1_opd*1e9)
    plt.colorbar()
    plt.title('M1 OPD [nm]')
    plt.show(block=False)
 #   plt.savefig(f'{path}{n1}_{n2}_M1_initial_shape.png', dpi=300, bbox_inches='tight') 

if com_amp>0:
    plt.figure()
    plt.imshow(com*1e9)
    plt.colorbar()
    plt.title('NCPA OPD [nm]')
    plt.show(block=False)
  #  plt.savefig(f'{path}{n1}_{n2}_ncpa.png', dpi=300, bbox_inches='tight')
    
    plt.figure()
    plt.imshow(com_seg*KAO.tel.pupil*1e9)
    plt.colorbar()
    plt.title('NCPA projected on segments [nm]')
    plt.show(block=False)
   # plt.savefig(f'{path}{n1}_{n2}_ncpa_proj_seg.png', dpi=300, bbox_inches='tight')
   
    plt.figure()
    plt.imshow((M1_opd+com_seg)*KAO.tel.pupil*1e9)
    plt.colorbar()
    plt.title('NCPA + M1 projected on segments [nm]')
    plt.show(block=False)

plt.figure()
plt.imshow(LE_PSF**0.2)
plt.colorbar()
plt.title('PSF')
plt.show(block=False)
#plt.savefig(f'{path}{n1}_{n2}_PSF.png', dpi=300, bbox_inches='tight')

plt.figure()
plt.imshow(total_M1_applied*1e9)
plt.colorbar()
plt.title('Total ZWFS correction [nm]')
plt.show(block=False)
#plt.savefig(f'{path}{n1}_{n2}_ZWFS_total_correction.png', dpi=300, bbox_inches='tight')

plt.figure()
plt.imshow(KAO.opd_M1.OPD*1e9)
plt.colorbar()
plt.title('M1 shape after ZWFS CL [nm]')
plt.show(block=False)
#plt.savefig(f'{path}{n1}_{n2}_M1_final_shape.png', dpi=300, bbox_inches='tight')

plt.figure()
plt.imshow(KAO.dm.OPD*KAO.tel.pupil*1e9)
plt.colorbar()
plt.title('DM OPD [nm]')
plt.show(block=False)
#plt.savefig(f'{path}{n1}_{n2}_DM_final_shape.png', dpi=300, bbox_inches='tight')

plt.figure()
plt.imshow((delta_mirror)*KAO.tel.pupil*1e9)
plt.colorbar()
plt.title('Delta OPD [nm]')
plt.show(block=False)

plt.figure()
plt.imshow(KAO.tel.OPD*1e9)
plt.colorbar()
plt.title('Science Output phase [nm]')
plt.show(block=False)









