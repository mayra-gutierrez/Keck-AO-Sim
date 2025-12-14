#%%
import numpy as np
import matplotlib.pyplot as plt

from OOPAO.Zernike import Zernike
from hcipy import *
from scipy.ndimage import center_of_mass

import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim/")
from keckTel import keckTel, keckStandard
from keckAtm import keckAtm

from simulations_codes.research_copy.KAO_parameter_file_research_copy import initializeParameterFile
from simulations_codes.ZWFS_toolbox.ZWFS_tools import *

#%% 
param = initializeParameterFile()
keck_object = keckTel.create('keck', resolution=param['resolution'], samplingTime=param['samplingTime'], return_segments=True)
tel = keck_object.tel
KSM = SegmentedDeformableMirror(keck_object.keck_segments)
s1 = np.array(KSM.segments)
seg_vect2D = s1.reshape(s1.shape[0],int(np.sqrt(s1.shape[1])),int(np.sqrt(s1.shape[1])))
flat = np.sum(seg_vect2D,axis=0)

my_basis = [
    1,   # Tilt x (you used 1 here)
    2,   # Tilt y
    4,   # Astig. oblique
    3,   # Defocus
    5,   # Astig. vertical
    8,   # Trefoil y
    7,   # Coma x
    6,   # Coma y
    9,   # Trefoil x
    14,  # Tetrafoil (n=4,m=-4)
    12,  # Secondary astig oblique
    10,  # Primary spherical (n=4,m=0)
    11,  # Secondary astig vertical
    13   # Tetrafoil (n=4,m=+4)
]


'''
nmin = 2
nmax = 4
nlist = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4]
nlist = np.array(nlist)
f1 = nlist[nlist>=nmin].astype(float)
psd = f1 ** (-2)
zk_coefficients = psd*np.random.randn(f1.shape[0])

noll_to_my_index_1based = {noll: idx+1 for idx, noll in enumerate(my_basis)}
amp_vect = np.array([0,0,0,-27,-15,66,0,-3,-20,0,5,-11,-4,6]) # nm RMS

#amp_vect = [-63,-62,-54,-35,-13,7,-10,2,-3,-17,-19]
phase = np.zeros((zk_res,zk_res))

for j in range(1,len(amp_vect)):
    ind = noll_to_my_index_1based[j]
    zk = z2p[:,ind].reshape(zk_res,zk_res)*zk_coefficients[j]
    phase = phase + zk

amp_phase = float(np.std(phase[np.where(s1 > 0)]))
plt.figure(), plt.imshow(-phase), plt.colorbar(), plt.show(block=False)

z_indexmin = np.where(nlist == nmin)[0]
z_indexmax = np.where(nlist == nmax)[0]
Zvect = z2p[:,z_indexmin[0]:z_indexmax[-1]+1]
phase_screen = Zvect * zk_coefficients
phase_screen = np.sum(phase_screen, axis=1)
phase_screen = phase_screen.reshape(zk_res,zk_res)
plt.figure(), plt.imshow(phase_screen), plt.colorbar(), plt.show(block=False)
opd_screen = phase_screen *640*1e-9/(2*np.pi)
amp_screen = np.std(opd_screen[np.where(s1 > 0)])
warping = opd_screen/amp_screen 

plt.figure(), plt.imshow(warping), plt.colorbar(), plt.show(block=False)
'''
#%% loop to created warped surface in all segments

nmin = 2
nmax = 4
nlist = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4]
nlist = np.array(nlist)
f1 = nlist[nlist>=nmin].astype(float)
psd = f1 ** (-2)
z_indexmin = np.where(nlist == nmin)[0]
z_indexmax = np.where(nlist == nmax)[0]

seg_vect2D_copy = seg_vect2D.copy()
Nseg = 36
for k in range(Nseg):
    seg_pup = seg_vect2D_copy[k,:,:].copy()
    y,x = center_of_mass(seg_pup)
    ll = 25
    deb_y = int(np.round(y))-ll; fin_y = int(np.round(y))+ll
    deb_x = int(np.round(x))-ll; fin_x = int(np.round(x))+ll
    if deb_x<0:
        deb_x = 0 
    if deb_y<0:
        deb_y = 0
    s1 = seg_pup[deb_y:fin_y,deb_x:fin_x]
    rows, cols = s1.shape
    if rows != cols:
        size = max(rows, cols)  # target square size
        s1_padded = np.full((size, size), 0)  # use np.nan or 0 if you prefer
        s1_padded[:rows, :cols] = s1
    else:
        s1_padded = s1
        
    z2p,p2z = zernikeBasis_nonCirc(20,s1_padded) #Noll Zk
    zk_res = s1_padded.shape[0]
    zk_coefficients = psd*np.random.randn(f1.shape[0])

    Zvect = z2p[:,z_indexmin[0]:z_indexmax[-1]+1]
    phase_screen = Zvect * zk_coefficients
    phase_screen = np.sum(phase_screen, axis=1)
    phase_screen = phase_screen.reshape(zk_res,zk_res)
    opd_screen = phase_screen *640*1e-9/(2*np.pi)
    amp_screen = np.std(opd_screen[np.where(s1 > 0)])
    warping = opd_screen/amp_screen 
    warping_cropped = warping[:rows, :cols]
    
    seg_pup[deb_y:fin_y,deb_x:fin_x] = warping_cropped
    seg_vect2D_copy[k,:,:] = seg_pup

amp_warp = 66*1e-9   
warpped_surf = np.sum(seg_vect2D_copy,axis=0)
amp_ = np.std(warpped_surf[np.where(flat != 0)])
warpped_surf = warpped_surf/amp_ * amp_warp

plt.figure(), plt.imshow(warpped_surf*1e9), plt.colorbar(), plt.title('Warpped surface in nm RMS'), plt.show(block=False)

# piston
for i in range(Nseg):
    aa = np.random.rand(3) # (piston in m, tip in rad, tilt in rad)
    KSM.set_segment_actuators(i, aa[0], aa[1]*0, aa[2]*0)

amp_piston = 100*1e-9        
opd_phasing = KSM.opd.shaped # OPD

cophase_amp = np.std(opd_phasing[np.where(flat != 0)])
piston_shape = opd_phasing/cophase_amp * amp_piston
piston_shape = (piston_shape - np.mean(piston_shape[np.where(flat!=0)]))*flat

M1_error = (piston_shape + warpped_surf)*tel.pupil
amp_M1 = np.std(M1_error[np.where(tel.pupil != 0)])

plt.figure(), plt.imshow(piston_shape*1e9), plt.colorbar(), plt.title('Piston OPD in nm RMS'), plt.show(block=False)

plt.figure(), plt.imshow(M1_error*1e9), plt.colorbar(), plt.title('M1 OPD in nm RMS'), plt.show(block=False)


