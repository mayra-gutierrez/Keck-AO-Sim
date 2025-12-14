#%%
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from hcipy import *
import matplotlib.animation as animation
from io import BytesIO
import imageio.v2 as imageio
import os

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
from simulations_codes.vandamstrehl import *

#%% AO system
#%% Import KAO system
param = initializeParameterFile()
KAO = initialize_AO_hardware(param)
ZWFS_param = {'activate': False, 'max_CLiter':5,'gain':1,'n_average':5,'avg_time':4,'freq':4,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

M1_opd = KAO.opd_M1.OPD
ncpa = KAO.opd_ncpa.OPD
offset_ini = KAO.opd_offset.OPD

#%% One mode reconstruction
'''
basis = KAO.basis
proj = KAO.projector_modal
Rec = KAO.keck_reconstructor
opd_input = OPD_map(telescope=KAO.tel)
screen = (basis[:,1]*2+basis[:,0]*0.5+basis[:,7]*1)
screen = screen.reshape(param['resolution'],param['resolution'])
opd_input.OPD = screen*KAO.ngs.wavelength/(2*np.pi)*0

#ncpa
KAO.opd_ncpa.OPD = -ncpa
KAO.wfs.reference_slopes_maps = KAO.cog_ncpa

KAO.tel.isPaired = True # DO NOT CHANGE THIS
KAO.tel.resetOPD()
KAO.dm.coefs = 0
KAO.ttm.coefs = 0
KAO.ngs * KAO.tel * opd_input * KAO.ttm * KAO.dm * KAO.wfs  
opd_ini= KAO.tel.OPD 
amp_ini = np.std(opd_ini) * 1e9
coef_ini = proj @ opd_ini.flatten()

plt.figure(), plt.imshow(opd_ini), plt.colorbar(), plt.title('Initial OPD'), plt.show(block=False)

signal = KAO.wfs.signal
slope_z1 = KAO.wfs.signal
plt.figure(), plt.imshow(KAO.wfs.signal_2D), plt.colorbar(), plt.title('slopes map screen'), plt.show(block=False)

command = Rec @ signal
cc_dm = command[:KAO.dm.nValidAct] 
KAO.dm.coefs = KAO.dm.coefs-cc_dm
KAO.ttm.coefs = KAO.ttm.coefs-command[KAO.dm.nValidAct:]
#tel.resetOPD()
KAO.ngs * KAO.tel * opd_input * KAO.opd_ncpa *KAO.ttm * KAO.dm
opd_res = KAO.tel.OPD
amp_res = np.std(opd_res) * 1e9
amp_dm = np.std(KAO.dm.OPD*KAO.tel.pupil)*1e9
amp_ttm = np.std(KAO.ttm.OPD*KAO.tel.pupil)*1e9
plt.figure(), plt.imshow(opd_res*1e9), plt.colorbar(), plt.title('Residual OPD'), plt.show(block=False)
plt.figure(), plt.imshow(KAO.dm.OPD*KAO.tel.pupil*1e9), plt.colorbar(), plt.title('DM OPD'), plt.show(block=False)
plt.figure(), plt.imshow(KAO.ttm.OPD*1e9), plt.colorbar(), plt.title('TTM OPD'), plt.show(block=False)

print(f'Amp initial {amp_ini} and amp residual {amp_res}')
'''
#%% close loop test
'''
KAO.param['print_display']=True
KAO.param['nLoop'] = 1000
KAO.param['display_loop'] = False   
KAO.param['gainCL'] = 0.1 
KAO.param['NCPA'] = True
KAO.opd_M1.opd = M1_opd*1
AO_output = close_loop(KAO,ZWFS_param)
'''

#%% NCPA as phase map not taken into account in the calibration

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

#apply offset
#KAO.opd_offset.OPD = offset_ini*1
#offset_seg_coefs = np.dot(KAO.opd_offset.OPD.flatten(),proj_seg)
basis = KAO.basis

screen = basis[:,7]
screen = screen.reshape(param['resolution'],param['resolution'])
KAO.opd_offset.OPD = screen*KAO.ngs.wavelength/(2*np.pi)*0.2
offset_seg_coefs = np.dot(KAO.opd_offset.OPD.flatten(),proj_seg)
offset_seg = np.dot(offset_seg_coefs,segments).reshape(param['resolution'],param['resolution'])*KAO.tel.pupil
ncpa_amp = float(np.std(KAO.opd_offset.OPD[np.where(KAO.tel.pupil > 0)])) * 1e9
ncpa_amp_seg = float(np.std(offset_seg[np.where(KAO.tel.pupil > 0)])) * 1e9
#KAO.opd_offset.OPD = np.zeros((280,280))
# Closed-loop
KAO.wfs.is_geometric = False
KAO.tel.isPaired = True 
KAO.tel.resetOPD()
KAO.dm.coefs = 0
KAO.ttm.coefs = 0
KAO.opd_M1.OPD = M1_opd*0#offset_seg#
#KAO.wfs.reference_slopes_maps = KAO.cog_ncpa #applied ncpa
#KAO.opd_ncpa.OPD = -ncpa
KAO.ngs * KAO.tel
KAO.tel * KAO.ttm * KAO.dm * KAO.wfs
m1_amp = []
m1_z = float(np.std(KAO.opd_M1.OPD[np.where(KAO.tel.pupil > 0)])) * 1e9
m1_amp.append(m1_z)
total_M1_applied = 0
seg_0 = 0
slopes = 0
slopes10 = 0
slopes1d_10 = 0
slopes1d = 0
psf_list = []
psf_ini = []
psf_diff = []

ratio_time = int(KAO.science_detector.integrationTime/KAO.tel.samplingTime)

niter = 100
gainCL = 0.5
gainTTM = gainCL
leak = param['leak']

for k in range(ratio_time+1):
    KAO.tel.resetOPD()
    KAO.ngs * KAO.tel * KAO.ttm * KAO.dm * KAO.wfs 
    
    KAO.science * KAO.tel *KAO.science_detector # if cog offset KAO.opd_ncpa 
    psf = KAO.science_detector.frame[:]
    
    if k>0 and k%ratio_time==0:
        psf_diff.append(psf) 
       
# aberrated PSF
for k in range(ratio_time+1):
    KAO.tel.resetOPD()
    KAO.ngs * KAO.tel * KAO.opd_M1 * KAO.ttm * KAO.dm * KAO.wfs 
    
    KAO.science * KAO.tel * KAO.opd_offset *KAO.science_detector # if cog offset KAO.opd_ncpa 
    psf = KAO.science_detector.frame[:]
    
    if k>0 and k%ratio_time==0:
        psf_ini.append(psf) 
 
sr_ini = strehl(psf_ini[0],psf_diff[0],pos=[256,256])

for k in range(niter):
    KAO.tel.resetOPD()
    KAO.ngs * KAO.tel * KAO.opd_M1 * KAO.ttm * KAO.dm * KAO.wfs  
    wfsSignal = KAO.wfs.signal  
    
    KAO.science * KAO.tel * KAO.opd_offset *KAO.science_detector # if cog offset KAO.opd_ncpa 
    psf = KAO.science_detector.frame[:]
    
    if k%ratio_time==0:
        psf_list.append(psf)
    
    #ZWFS
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
    if k==0:
        seg_0 = seg_rec
        slopes = KAO.wfs.signal_2D
        slopes1d = KAO.wfs.signal
    if k==10:
        slopes10 = KAO.wfs.signal_2D
        slopes1d_10 = KAO.wfs.signal
        
    total_M1_applied += seg_rec
              
    KAO.opd_M1.OPD = KAO.opd_M1.OPD - seg_rec
    m1_z = float(np.std(KAO.opd_M1.OPD[np.where(KAO.tel.pupil > 0)])) * 1e9
    m1_amp.append(m1_z)
    
    #AO
    command = Rec @ wfsSignal
    command_tronc = command.copy()
    command_tronc[np.where(np.abs(command_tronc)<1e-10)]=0
    com_dm = command_tronc[:KAO.dm.nValidAct]
    com_ttm = command[KAO.dm.nValidAct:]
    KAO.dm.coefs = KAO.dm.coefs - gainCL * com_dm
    KAO.ttm.coefs = KAO.ttm.coefs - gainTTM * com_ttm
    #if k>10:
        #KAO.dm.coefs = leak*KAO.dm.coefs - gainCL * com_dm
        #KAO.ttm.coefs = KAO.ttm.coefs - gainTTM * com_ttm

dm_ = KAO.dm.OPD*KAO.tel.pupil
mean_dm = np.mean(dm_[np.where(KAO.tel.pupil>0)])
dm_shape = dm_-mean_dm

KAO.tel.resetOPD()
KAO.ngs * KAO.tel * KAO.opd_M1 * KAO.ttm * KAO.dm 
opd_final_wfs = KAO.tel.OPD
opd_final_wfs = (opd_final_wfs-np.mean(opd_final_wfs[np.where(KAO.tel.pupil>0)]))*KAO.tel.pupil

KAO.science * KAO.tel * KAO.opd_offset
opd_final_science = KAO.tel.OPD
opd_final_science = (opd_final_science-np.mean(opd_final_science[np.where(KAO.tel.pupil>0)]))*KAO.tel.pupil

# plots
deb = 512//2-50
fin = 512//2+50

path = '/home/mcisse/keckAOSim/keckSim/data/simple_data/'
m1 = f'{round(m1_amp[0])}nmRMS'
of = f'{round(ncpa_amp)}nmRMS'
fig_name = f'{path}M1_{m1}_pure_NCPA_{of}'

fig_wfs  = plt.figure(); ax_wfs  = plt.gca()
im = ax_wfs.imshow(opd_final_wfs*1e9);ax_wfs.set_title('WFS Output phase [nm]');fig_wfs.colorbar(im, ax=ax_wfs)
plt.show(block=False)

fig_s  = plt.figure(); ax_s  = plt.gca()
im = ax_s.imshow(opd_final_science*1e9);ax_s.set_title('Science Output phase [nm]');fig_s.colorbar(im, ax=ax_s)
plt.show(block=False)

fig_m1  = plt.figure(); ax_m1  = plt.gca()
im=ax_m1.imshow(KAO.opd_M1.OPD*1e9);ax_m1.set_title('M1 shape after CL [nm]');fig_m1.colorbar(im, ax=ax_m1)
plt.show(block=False)

fig_m1ini  = plt.figure(); ax_m1ini  = plt.gca()
im=ax_m1ini.imshow(M1_opd*1e9);ax_m1ini.set_title('Input M1 OPD [nm]');fig_m1ini.colorbar(im, ax=ax_m1ini)
plt.show(block=False)

fig_m1off  = plt.figure(); ax_m1off  = plt.gca()
im=ax_m1off.imshow((M1_opd-offset_seg)*1e9);ax_m1off.set_title('Input M1 OPD - offset projected on segments [nm]');fig_m1off.colorbar(im, ax=ax_m1off)
plt.show(block=False)

fig_off  = plt.figure(); ax_off  = plt.gca()
im=ax_off.imshow((offset_seg)*1e9);ax_off.set_title('Offset projected on segments [nm]');fig_off.colorbar(im, ax=ax_off)
plt.show(block=False)

fig_corr  = plt.figure(); ax_corr  = plt.gca()
im=ax_corr.imshow((total_M1_applied)*1e9);ax_corr.set_title('Total ZWFS correction [nm]');fig_corr.colorbar(im, ax=ax_corr)
plt.show(block=False)

fig_dm  = plt.figure(); ax_dm  = plt.gca()
im=ax_dm.imshow(dm_shape*KAO.tel.pupil*1e9);ax_dm.set_title('DM OPD [nm]');fig_dm.colorbar(im, ax=ax_dm)
plt.show(block=False)

fig_psf  = plt.figure(); ax_psf  = plt.gca()
im=ax_psf.imshow(psf_ini[0][deb:fin,deb:fin]**0.2);ax_psf.set_title(f'Initial PSF SR = {round(sr_ini*100)}% @H band');fig_psf.colorbar(im, ax=ax_psf)
plt.show(block=False)

psf_list = np.array(psf_list)
psf = psf_list[-1,:,:]
fig = plt.figure(); axes  = plt.gca()
fig.suptitle(f'PSF {k}', fontsize=14)
axes.imshow(psf[deb:fin,deb:fin]**0.2, cmap='viridis')
axes.axis('off')
plt.show(block=False)

sr_final = strehl(psf_list[-1,:,:],psf_diff[0],pos=[256,256])
'''
images = []
fig = plt.figure(); ax  = plt.gca()
for k in range(psf_list.shape[0]):
    aa = psf_list[k,:,:]
    ax.imshow(aa[deb:fin,deb:fin]**0.2)
    ax.set_title(f'PSF SR = {round(sr_final*100)}% @H band', fontsize=14)  # ? Add title here
       
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    images.append(imageio.imread(buf))
    buf.close()
    ax.clear()
plt.close(fig)
imageio.mimsave(f'{fig_name}_PSF_animation.gif', images, duration=2)

#save 

fig_wfs.savefig(f'{fig_name}_output_phase_wfs.png', dpi=300, bbox_inches='tight')  
fig_s.savefig(f'{fig_name}_output_phase_science.png', dpi=300, bbox_inches='tight')  
fig_m1.savefig(f'{fig_name}_Final_M1_shape.png', dpi=300, bbox_inches='tight')
fig_m1ini.savefig(f'{fig_name}_Initial_M1_shape.png', dpi=300, bbox_inches='tight')
fig_m1off.savefig(f'{fig_name}_Initial_M1_shape_plus_offset_segments.png', dpi=300, bbox_inches='tight')
fig_off.savefig(f'{fig_name}_total_ZWFS_correction_applied.png', dpi=300, bbox_inches='tight')
fig_corr.savefig(f'{fig_name}_total_ZWFS_correction_applied.png', dpi=300, bbox_inches='tight')
fig_dm.savefig(f'{fig_name}_DM_shape.png', dpi=300, bbox_inches='tight')
fig_psf.savefig(f'{fig_name}_Initial_PSF.png', dpi=300, bbox_inches='tight')

'''

















