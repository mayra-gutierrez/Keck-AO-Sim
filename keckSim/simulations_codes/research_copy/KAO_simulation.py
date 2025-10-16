#%%
import time
import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from scipy.signal import convolve2d
from scipy.signal import welch

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.Source import Source
from OOPAO.Detector import Detector
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.calibration.getFittingError import *
from OOPAO.tools.displayTools import cl_plot, displayMap
from OOPAO.OPD_map import OPD_map

import os, sys
os.chdir("/home/mcisse/PycharmProjects/keckSim/keckSim")
from keckTel import keckTel
from keckAtm import keckAtm

from Jitter_TT import RandomTipTilt_Gaussian, SinusoidalTipTilt
from SH import ShackHartmann_modifNoise
from Imat_SHWFS import InteractionMatrix_test

#%% -----------------------     Keck Aperture   ----------------------------------
Nx = 240*2
Nseg=36
samp = 1/1000
#Create a Keck telescope with default parameters
keck_object = keckTel.create('keck', resolution=Nx,samplingTime=samp, return_segments=True)

#%% -----------------------     Segmented DM   ----------------------------------
#'''
segments_vector = np.zeros((Nx**2, Nseg))
for i in range(Nseg):
    segments_vector[:, i] = keck_object.keck_segments[i]

KSM = SegmentedDeformableMirror(keck_object.keck_segments)
KSM.set_segment_actuators(np.arange(Nseg), 1,0,0)
flat_ksm = KSM.opd.shaped

for i in range(Nseg):
    aa = np.random.rand(3) # (piston in m, tip in rad, tilt in rad)
    KSM.set_segment_actuators(i, aa[0], aa[1]*0, aa[2]*0)

opd_phasing = KSM.opd.shaped # OPD
surf = KSM.surface.shaped # Surface of the DM it is the OPD divided by 2
cophase_amp = np.std(opd_phasing[np.where(flat_ksm != 0)])

seg_amp = 100*1e-9
default_shape = opd_phasing/cophase_amp * seg_amp
aa = np.std(default_shape[np.where(flat_ksm != 0)])

plt.figure(), plt.imshow(default_shape), plt.colorbar(), plt.title('OPD Co-Phasing error'), plt.show()
plt.figure(), plt.imshow(flat_ksm), plt.colorbar(), plt.title('flat'), plt.show()

#'''

# %% -----------------------     Circular TELESCOPE   ----------------------------------
D = keck_object.tel.D
Dobs = 2.6 #obs central in m
obs = Dobs/D
tel_circ = Telescope(diameter=D,
            resolution=Nx,
            samplingTime=samp,
            centralObstruction=obs)

# %% -----------------------     NGS   ----------------------------------
mag = 5
ngs = Source(optBand='V',  # Optical band (see photometry.py)
             magnitude=mag)  # Source Magnitude

src = Source(optBand='K',  # Optical band (see photometry.py)
             magnitude=mag)  # Source Magnitude

LD_mas = src.wavelength/D * 206265 * 1000 # in mas

# propagate the NGS to the telescope object
ngs * keck_object.tel

keck_object.tel.computePSF(zeroPaddingFactor=4)
PSF = keck_object.tel.PSF
size_psf = PSF.shape[0]//2

plt.figure(), plt.imshow(PSF[size_psf-50:size_psf+50,size_psf-50:size_psf+50]**0.2,cmap='gray'), plt.colorbar(), plt.show()

# %% -----------------------     ATMOSPHERE   ----------------------------------

atm = keckAtm.create(telescope=keck_object.tel, atm_type='maunakea') # median condition 
#atm_type = 'custom' 3 to change the r0,L0,etc...
atm.atm.update()
# %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
nAct = 61

path = '/home/mcisse/PycharmProjects/data_pyao/research/ZWFS/data_simulation/'
binary_act_map = np.load(path+'binary_actuators_map.npy')
map_illuminated_act = np.load(path+'map_of_actuators_always_illuminated.npy')
if nAct == 21:
    mask = binary_act_map
else:
    print(f'HAKA version nact{nAct}')
    grid=np.mgrid[0:nAct,0:nAct]
    rgrid=np.sqrt((grid[0]-nAct/2+0.5)**2+(grid[1]-nAct/2+0.5)**2)
    mask = np.zeros((nAct,nAct)).astype(np.float32)
    mask[np.where(rgrid<nAct/2)]=1

x = np.linspace(-keck_object.tel.D/2, keck_object.tel.D/2, nAct)
valid = (mask == 1)
X, Y = np.meshgrid(x, x)
# Extract the (x,y) coordinates only where the mask is 1
xc = X[valid]
yc = Y[valid]
coords0 = np.column_stack((xc, yc))
pitch = keck_object.tel.D / nAct

coords_final = np.column_stack((xc, yc))
DM= DeformableMirror(telescope=keck_object.tel,
                     nSubap=nAct - 1,
                     mechCoupling=0.1458,
                     coordinates=coords_final,
                     pitch=pitch,
                     floating_precision=32)

dm_modes = DM.modes
projector_DM = np.linalg.pinv(dm_modes)

plt.figure()
plt.imshow(np.reshape(np.sum(DM.modes**5,axis=1),[keck_object.tel.resolution,keck_object.tel.resolution]).T +
            keck_object.tel.pupil,extent=[-keck_object.tel.D/2,keck_object.tel.D/2,-keck_object.tel.D/2,keck_object.tel.D/2])
plt.plot(DM.coordinates[:,0],DM.coordinates[:,1],'rx')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates keck')
plt.colorbar()
plt.show()

# %% -----------------------     Modal Basis - Zernike  ----------------------------------
Z = Zernike(keck_object.tel, 10)
Z.computeZernike(keck_object.tel)
M2C_zernike = np.linalg.pinv(np.squeeze(DM.modes[keck_object.tel.pupilLogical, :])) @ Z.modes

DM.coefs = M2C_zernike[:,:2]
keck_object.tel * DM
TT_modes = keck_object.tel.OPD.reshape(keck_object.tel.resolution ** 2, 2)

DM.coefs = M2C_zernike
keck_object.tel * DM
Zk_modes = keck_object.tel.OPD.reshape(keck_object.tel.resolution ** 2, M2C_zernike.shape[1])
projector_zk = np.linalg.pinv(Zk_modes)

# %% -----------------------     Modal Basis - KL Basis  ----------------------------------
M2C_KL = compute_KL_basis(keck_object.tel, atm, DM)

DM.coefs = M2C_KL
keck_object.tel * DM
KL_modes = keck_object.tel.OPD.reshape(keck_object.tel.resolution ** 2, M2C_KL.shape[1])
projector_kl = np.linalg.pinv(KL_modes)

# %% -----------------------     TTM   ----------------------------------

TTM_bis = DeformableMirror(telescope=keck_object.tel,
                            nSubap=2,
                            mechCoupling=0.1458,
                            coordinates=None,
                            pitch=pitch,
                            modes=TT_modes)
Zb = Zernike(tel_circ, 10)
Zb.computeZernike(tel_circ)
M2C_zernike_bis = np.linalg.pinv(np.squeeze(DM.modes[tel_circ.pupilLogical, :])) @ Zb.modes
DM.coefs = M2C_zernike_bis[:,:2]
tel_circ * DM
TT_modes_b = tel_circ.OPD.reshape(tel_circ.resolution ** 2, 2)

TTM_pur = DeformableMirror(telescope=tel_circ,
                            nSubap=2,
                            mechCoupling=0.1458,
                            coordinates=None,
                            pitch=pitch,
                            modes=TT_modes_b)

# %% -----------------------     SH WFS   ----------------------------------
nsubap = nAct-1
# make sure tel and atm are separated to initialize the PWFS
tel_circ.isPaired = False
tel_circ.resetOPD()
ngs * tel_circ
pix_scale = 0.8 # in arcsec
npix_sub = 4

# modification of the SH from OOPAO to fix a typo error in the noise
if nsubap==20:
    binary_act_subap_sh = np.load(path+'binary_sub_apertures_map.npy')
else:
    grid=np.mgrid[0:nsubap,0:nsubap]
    rgrid=np.sqrt((grid[0]-nsubap/2+0.5)**2+(grid[1]-nsubap/2+0.5)**2)
    mask_sub = np.zeros((nsubap,nsubap)).astype(np.float32)
    mask_sub[np.where(rgrid<nsubap/2)]=1

    nsubap_obs = int(nsubap*obs)
    rgrid=np.sqrt((grid[0]-nsubap/2+0.5)**2+(grid[1]-nsubap/2+0.5)**2)
    mask_obs = np.zeros((nsubap,nsubap)).astype(np.float32)
    mask_obs[np.where(rgrid<nsubap_obs/2)]=1

    binary_act_subap_sh = mask_sub-mask_obs

#perfect match
wfs = ShackHartmann_modifNoise(nSubap=nsubap,
                                telescope=tel_circ,
                                lightRatio=0.36,
                                pixel_scale=pix_scale,
                                shannon_sampling=False,
                                n_pixel_per_subaperture=npix_sub)

keck_object.tel.isPaired = False
keck_object.tel.resetOPD()
ngs * keck_object.tel
wfs_bis = ShackHartmann_modifNoise(nSubap=nsubap,
                                telescope=keck_object.tel,
                                lightRatio=0.36,
                                pixel_scale=pix_scale,
                                shannon_sampling=False,
                                n_pixel_per_subaperture=npix_sub)

aa = wfs.valid_subapertures
plt.figure(),plt.imshow(aa.astype(int)-binary_act_subap_sh), plt.colorbar()
bb = wfs_bis.valid_subapertures
plt.figure(),plt.imshow(bb.astype(int)-binary_act_subap_sh), plt.colorbar()
#change detector
'''
OCAM = Detector(nRes=wfs.cam.resolution,
                integrationTime=tel.samplingTime,
                bits=14,
                FWC=270000,
                gain=500,
                sensor='EMCCD',
                QE=0.92,
                binning=1,
                psf_sampling=1,
                darkCurrent=0,
                readoutNoise=0.3,
                photonNoise=True)
'''

# propagate the light to the Wave-Front Sensor

ngs *tel_circ * wfs
frame_keck = wfs.cam.frame
print(f'Signal shape keck {wfs.signal.shape}')

ngs *keck_object.tel * wfs_bis
frame_bis = wfs_bis.cam.frame
print(f'Signal shape keck {wfs_bis.signal.shape}')

#plots

plt.figure(), plt.imshow(frame_keck), plt.colorbar(), plt.title('WFS Camera Frame keck style'), plt.show()
plt.figure(), plt.imshow(wfs.signal_2D), plt.colorbar(), plt.title('WFS slopes keck style'), plt.show()

plt.figure(), plt.imshow(frame_bis), plt.colorbar(), plt.title('WFS Camera Frame OOPAO style'), plt.show()
plt.figure(), plt.imshow(wfs_bis.signal_2D), plt.colorbar(), plt.title('WFS slopes OOPAO style'), plt.show()


# %% -----------------------     Science detector  ----------------------------------
# science camera
#src_cam = Detector(1024)
import time
NIRC2 = Detector(nRes=1024,
                 integrationTime=keck_object.tel.samplingTime*10,
                 bits=14,
                 sensor='CCD',
                 QE=0.92,
                 binning=1,
                 psf_sampling=5,
                 darkCurrent=0,
                 readoutNoise=0.0,
                 photonNoise=False)
src_cam = NIRC2

# Science diffracted limited PSF
src_cam.integrationTime = 1#keck_object.tel.samplingTime
keck_object.tel.isPaired = False
keck_object.tel.resetOPD()
nn = int(src_cam.integrationTime/keck_object.tel.samplingTime)
PSF_diff = sum((src * keck_object.tel * src_cam, src_cam.frame)[1] for _ in range(nn))
aa = src_cam.resolution//2   

plt.figure(), plt.imshow(PSF_diff[aa-50:aa+50,aa-50:aa+50]**0.2), plt.colorbar(), plt.title('Diffraction limited PSF'), plt.show()
plt.figure(), plt.imshow(src_cam.frame[aa-50:aa+50,aa-50:aa+50]**0.2), plt.colorbar(), plt.title('Diffraction limited PSF frame'), plt.show()

#%% filter TTM 
Reg_TTM_DM = np.zeros((2, DM.modes.shape[1]))
eps = 0
for k in range(2):
    zi = Z.modesFullRes[:,:,k]
    dm_command = projector_DM @ zi.flatten()
    Reg_TTM_DM[k,:] = dm_command

Reg_TTM = Reg_TTM_DM @ Reg_TTM_DM.T
P_TT = Reg_TTM_DM.T @ np.linalg.pinv(Reg_TTM + eps * np.eye(Reg_TTM.shape[0])) @ Reg_TTM_DM
I = np.eye(P_TT.shape[0])
P_orth_TTM = I - P_TT  # Projects orthogonally to T/T

# %% Modal Interaction Matrix

stroke = 1e-9  # amplitude of the modes in m
M2C_zonal = np.eye(DM.nValidAct)
wfs.is_geometric = False
tel_circ.resetOPD() # to remove the default OPD of the co-phasing error

# zonal interaction matrix
calib_zonal = InteractionMatrix_test(ngs=ngs, \
                                    tel=tel_circ, \
                                    dm=DM, \
                                    wfs=wfs, \
                                    M2C=M2C_zonal, \
                                    stroke=stroke, \
                                    nMeasurements=100, \
                                    noise='off')

plt.figure(), plt.plot(calib_zonal.eigenValues,'rx'), plt.ylabel('eigenValues zonal'), plt.show()
plt.figure(), plt.imshow(calib_zonal.D), plt.colorbar(), plt.title('Zonal Imat'), plt.show()

wfs_bis.is_geometric = False
keck_object.tel.resetOPD() # to remove the default OPD of the co-phasing error
calib_zonal_bis = InteractionMatrix_test(ngs=ngs, \
                                    tel=keck_object.tel, \
                                    dm=DM, \
                                    wfs=wfs_bis, \
                                    M2C=M2C_zonal, \
                                    stroke=stroke, \
                                    nMeasurements=100, \
                                    noise='off')

# TTM calib
M2C_TTM = np.eye(2)
calib_TTM_bis = InteractionMatrix_test(ngs=ngs, \
                                    tel=keck_object.tel, \
                                    dm=TTM_bis, \
                                    wfs=wfs_bis, \
                                    M2C=M2C_TTM, \
                                    stroke=stroke, \
                                    nMeasurements=1, \
                                    noise='off')

plt.figure(), plt.plot(calib_TTM_bis.eigenValues,'bo'), plt.ylabel('eigenValues zonal'), plt.show()

# %% Modal interaction matrix
# KL
calib_KL = CalibrationVault(calib_zonal.D @ M2C_KL)
plt.figure(), plt.plot(calib_KL.eigenValues,'rx'), plt.ylabel('eigenValues KL'), plt.show()

# Zernike
calib_zernike = CalibrationVault(calib_zonal.D @ M2C_zernike)
plt.figure(), plt.plot(calib_zernike.eigenValues,'rx'), plt.ylabel('eigenValues Zk'), plt.show()

# apply keck pupil
tel.pupil = pupil # to modify the pupil
plt.figure(),plt.imshow(tel.pupil), plt.colorbar(), plt.show()

#%% reconstructor for Keck (no use of SVD, Least-square or MMSE reconstructor)
# reset the OPD
tel.isPaired = False
tel.resetOPD()
DM.coefs = 0
ngs * tel * DM * wfs

# computation of the weight of the subap with respect to their illumination
int_subap = wfs.maps_intensity
weight = np.sum(int_subap, axis=(1, 2))
weight = weight/weight.max()
W0 = np.diag(weight)
W1 = np.diag(weight) * 0
W = np.asarray(np.bmat([[W0, W1], [W1, W0]]))

#inversion W
ww = np.diag(W)
inv_diag = np.where(ww != 0, 1 / ww, 0)
inv_W = np.diag(inv_diag)

# SVD
U, s, Vt = np.linalg.svd(calib_zonal.D, full_matrices=False)
nTrunc = 10

S = np.diag(s)
D = U @ S @ Vt
'''
threshold = 0.999
energy = np.cumsum(s**2) / np.sum(S**2)
k = np.searchsorted(energy, threshold) + 1
'''
nEigenValues = len(s) - nTrunc
iS = np.diag(1 / s)
M = Vt.T @ iS @ U.T

iStrunc = np.diag(1 / s[:nEigenValues])
Utrunc = U[:, :nEigenValues]
Vttrunc = Vt[:nEigenValues, :]
Mtrunc = Vttrunc.T @ iStrunc @ Utrunc.T
Strunc = np.diag(s[:nEigenValues])
Dtrunc = Utrunc @ Strunc @ Vttrunc

cond=s[0]/s[(nEigenValues-1)]

# add TTM
#calib_zonal.nTrunc = 1
Dtrunc_ =  Dtrunc #calib_zonal.Dtrunc #
H = np.zeros((Dtrunc_.shape[0], Dtrunc_.shape[1]))
H[:, :Dtrunc_.shape[1]] = Dtrunc_
Mat_inter = H.transpose()@ inv_W @ H
A = np.linalg.inv(Mat_inter)
B = H.transpose()@ inv_W
Rec_int1 = A @ B

H = np.zeros((calib_TTM_pur.D.shape[0], calib_TTM_pur.D.shape[1]))
H[:, :calib_TTM_pur.D.shape[1]] = calib_TTM_pur.D
Mat_inter = H.transpose()@ inv_W @ H
A = np.linalg.inv(Mat_inter)
B = H.transpose()@ inv_W
Rec_int2 = A @ B
Rec_int = np.vstack([Rec_int1,Rec_int2])

DM_rec = P_orth_TTM @ Rec_int[:DM.nValidAct,:]
TTM_rec = Rec_int[DM.nValidAct:,:]
Rec = np.vstack([DM_rec,TTM_rec])

# %% switch to a diffractive SH-WFS

wfs.is_geometric = False


#%% reconstruction one mode

tel.isPaired = False
tel.resetOPD()
DM.coefs = 0
TTM_pur.coefs = 0

DM.coefs = M2C_zernike[:, 9]*2e-9# + M2C_zernike[:, 0]*1e-9 + M2C_zernike[:, 2]*1e-9
opd_input = OPD_map(telescope=tel)
opd_input.OPD = DM.OPD * tel.pupil
plt.figure(), plt.imshow(opd_input.OPD), plt.colorbar(), plt.title('Tel OPD'), plt.show()

tel.isPaired = True # DO NOT CHANGE THIS
tel.resetOPD()
DM.coefs = 0
TTM_pur.coefs = 0
ngs * tel * opd_input * TTM_pur * DM *wfs  
ini= tel.OPD 
coef_in = projector_zk@ini.flatten()

plt.figure(), plt.imshow(ini), plt.colorbar(), plt.title('Tel OPD'), plt.show()

signal = wfs.signal
plt.figure(), plt.imshow(wfs.signal_2D), plt.colorbar(), plt.title('slopes map'), plt.show()
command = Rec @ signal

DM.coefs = DM.coefs-command[:DM.nValidAct] #dm.coefs - command
plt.figure(), plt.imshow(DM.OPD*tel.pupil), plt.colorbar(), plt.title('DM OPD est'), plt.show()
coef_dm = projector_zk@DM.OPD.flatten()

gain_tt = np.array([3.5,7])*0
TTM_pur.coefs = TTM_pur.coefs-command[DM.nValidAct:] * gain_tt
plt.figure(), plt.imshow(TTM_pur.OPD*tel.pupil), plt.colorbar(), plt.title('TTM OPD est'), plt.show()
coef_tt = projector_zk@TTM_pur.OPD.flatten()

correction = (TTM_pur.OPD+DM.OPD)*tel.pupil
coef_est = projector_zk@correction.flatten()
plt.figure(), plt.imshow(correction), plt.colorbar(), plt.title('Total cor'), plt.show()


tel.resetOPD()
ngs * tel * opd_input * TTM_pur * DM
opd_res = tel.OPD

res = np.std(opd_res) * 1e9
coef_res = projector_zk@opd_res.flatten()
plt.figure(), plt.imshow(opd_res), plt.colorbar(), plt.title('residual OPD'), plt.show()

plt.figure()
plt.plot(coef_in, 'rx', label='input coefficients')
plt.plot(coef_res, 'bo', label='res coefficients')
plt.plot(coef_tt, 'g*', label='TTM coefficients')
plt.plot(coef_dm, 'k+', label='DM coefficients')
plt.xlabel('Mode number')
plt.ylabel('Coefficient value [m]')
plt.legend()
plt.show()

#%% AO loop

wfs.is_geometric = False
tel.isPaired = True # do not change this
tel.resetOPD()
DM.coefs = 0
TTM_pur.coefs = 0
gain_tt = np.array([3.5,7])

atm.generateNewPhaseScreen(seed=10)
tel+atm
ngs * tel
tel * TTM_pur * DM * wfs

nLoop = 200
gain = 0.5
latency = 2

SR = np.zeros(nLoop)
total = np.zeros(nLoop)
residual = np.zeros(nLoop)
fitting_rms = np.zeros(nLoop)
wfsSignal = np.arange(0, wfs.nSignal) * 0
signalBuffer = np.zeros((wfs.signal.shape[0], nLoop)) # buffer for the frame delay
SRC_PSF = []

#Rec = np.vstack([M2C_KL@calib_KL.M,TTM_rec])  #np.vstack([DM_rec,TTM_rec]) 
for i in range(nLoop):
    a = time.time()
    atm.update()
    ngs * tel 
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    tel * TTM_pur* DM * wfs

    OPD_fitting_2D, OPD_corr_2D, OPD_turb_2D = getFittingError(tel.OPD, projector_DM, dm_modes, display=False)
    fitting_rms[i] = np.std(OPD_fitting_2D[np.where(tel.pupil > 0)]) * 1e9

    src * tel * src_cam
    
    if i >= latency:
        command = np.matmul(Rec, signalBuffer[:, i-latency])
        DM.coefs = DM.coefs - gain * command[:DM.nValidAct] 
        TTM_pur.coefs = TTM_pur.coefs - gain * (command[DM.nValidAct:] * gain_tt)

    signalBuffer[:, i] = wfs.signal

    SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
    residual[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9

    if i%10 == 0:
        res = tel.OPD
        res = (res - np.mean(res[np.where(tel.pupil>0)]))*tel.pupil
        plt.figure(),plt.imshow(res), plt.colorbar(), plt.title('Residual OPD'), plt.show()
        time.sleep(1)
    if i>50:
        SRC_PSF.append(src_cam.frame)

    print('Loop' + str(i) + '/' + str(nLoop) + ' Turbulence: ' + str(total[i]) + ' -- Residual:' +
          str(residual[i]) + ' -- Fitting:' + str(fitting_rms[i]) + '\n')
    
    

plt.figure()
plt.plot(total)
plt.plot(residual)
plt.plot(fitting_rms)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')
plt.show()

plt.figure()
plt.plot(SR)
plt.xlabel('Time')
plt.ylabel('Strehl Ratio')
plt.show()

print('Average SR = %d\n', np.mean(SR[50:]*100))

PSF_LE = np.mean(SRC_PSF,axis=0)
PSF_LE_norm =PSF_LE/ np.max(PSF_LE)  # Normalize PSF
log_PSF = np.log10(np.abs(PSF_LE))
beg = src_cam.resolution//2 - 100
end = src_cam.resolution//2 + 100
plt.figure(), plt.imshow(log_PSF[beg:end,beg:end],cmap = 'gray'), plt.colorbar(), plt.show()


#%% tip tilt jitter

LD = src.wavelength/tel.D * 206265 * 1000 # in mas
tel.resetOPD()
DM.coefs = M2C_zernike[:,:2]
tel*DM

TT = tel.OPD.reshape(tel.resolution ** 2, 2)
TT_mode = TT.reshape((Nx,Nx,2))

TT_jitter = TT_mode[:,:,0]
TT_jitter_OPD = TT_jitter*ngs.wavelength/(2*np.pi)
amp = np.std(TT_jitter_OPD[np.where(tel.pupil > 0)]) * 1e9 # in nm
opd_tt = OPD_map(telescope=tel)

opd_tt.OPD = TT_jitter_OPD * tel.pupil
src * tel * opd_tt

n_frames = 1000
freq = 50 # in Hz
tempo = np.linspace(0,1,1001)
amp_screen = []
dt = 1 / n_frames  # e.g., 1 ms time step
tt_rms = 100  # nm RMS

'''
for k in range (tempo.shape[0]):
    tel.resetOPD()
    TT_jitter = (np.sin(2*np.pi*freq*tempo[k])+amp_jitter)*TT_mode[:,:,0] + (np.cos(2*np.pi*freq*tempo[k])+amp_jitter)*TT_mode[:,:,1]
    TT_jitter_OPD = TT_jitter*ngs.wavelength/(2*np.pi)
    amp_screen.append(np.std(TT_jitter_OPD[np.where(tel.pupil > 0)]))

    opd_tt.OPD = TT_jitter_OPD * tel.pupil
    ngs * tel * opd_tt

    tel.computePSF(zeroPaddingFactor=2)
    PSF = tel.PSF
    aa = PSF.shape[0]//2
    #plt.figure(), plt.imshow(PSF[aa-50:aa+50,aa-50:aa+50]**0.2), plt.colorbar(), plt.title('tip tilt jitter'), plt.show()

plt.plot(tempo, amp_screen,'rx')
'''

signal = tt_rms * np.sin(2*np.pi*freq*tempo)

tiptilt_gen = RandomTipTilt_Gaussian(freq=freq, dt=dt, n_frames=n_frames, nm_rms=tt_rms) 
tiptilt = SinusoidalTipTilt(freq=freq, dt=dt, n_frames=n_frames, nm_rms=tt_rms)


fs = 1 / 0.001  # Sampling frequency
f, Pxx = welch(signal, fs=fs, window='hann', nperseg=1024, scaling='density')
fsine, Pxx_sine = welch(tiptilt.x, fs=fs, window='hann', nperseg=1024, scaling='density')
fgaus, Pxx_gaus = welch(tiptilt_gen.x, fs=fs, window='hann', nperseg=1024, scaling='density')

# plot PSD
plt.figure(figsize=(8, 4))
plt.semilogy(f, Pxx)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [arcsec²/Hz]')
plt.xlim(0, fs//10)
plt.title('Power Spectral Density - mine')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.semilogy(fsine, Pxx_sine)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [arcsec²/Hz]')
plt.xlim(0, fs//10)
plt.title('Power Spectral Density - Sine')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
plt.semilogy(fgaus, Pxx_gaus)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [arcsec²/Hz]')
plt.xlim(0, fs//10)
plt.title('Power Spectral Density - Gauss')
plt.grid(True)
plt.tight_layout()
plt.show()



'''
# easier : convolve the AO PSF with a Gaussain Kernel of std = X mas

psf_image = PSF_diff/PSF_diff.max()  # Normalize PSF

pixel_size_mas = LD_mas/src_cam.psf_sampling
# Define Gaussian kernel
size_psf = psf_image.shape[0]
x, y = np.meshgrid(np.arange(-size_psf//2, size_psf//2), 
                   np.arange(-size_psf//2, size_psf//2))
sigma = pixel_size_mas/8 # in mas
kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
#kernel /= np.sum(kernel)  

# Fourier Transform convolution
PSF_fft = np.fft.fft2(psf_image)
kernel_fft = np.fft.fft2(kernel)
prod_fft = PSF_fft * kernel_fft
convolved_psf = np.fft.fftshift(np.fft.ifft2(prod_fft)).real
convolved_psf = convolved_psf/np.sum(kernel)    # Normalize PSF

SR_jitter = computeSR(PSF_diff, convolved_psf)

# Plot results
plt.figure(), plt.imshow(convolved_psf[aa-50:aa+50,aa-50:aa+50]**0.2), plt.colorbar(), plt.title('Convolved PSF'), plt.show()
plt.figure(), plt.imshow(psf_image[aa-50:aa+50,aa-50:aa+50]**0.2), plt.colorbar(), plt.title('PSF'), plt.show()
plt.figure(), plt.imshow(kernel[aa-50:aa+50,aa-50:aa+50]), plt.colorbar(), plt.title('Kernel'), plt.show()

plt.figure()
plt.plot(psf_image[psf_image.shape[0]//2, :], 'r--', label='PSF')
plt.plot(convolved_psf[convolved_psf.shape[0]//2, :], 'b', label='Convolved PSF')
plt.hlines(0.5, xmin = 0, xmax = psf_image.shape[0], color='k', linestyle='--')
plt.xlim([490, 530])
plt.legend()
plt.show()
'''


#%%  NCPA

amp_ncpa = 150e-9
num_modes = 15
f = np.arange(1, num_modes+1, dtype=float)
psd = f ** (-2)
zk_coefficients = np.sqrt(psd) * np.random.randn(num_modes)
tel.resetOPD()
DM.coefs = 0
DM.coefs = M2C_zernike[:,2:num_modes+2]
tel*DM

Z_comb = tel.OPD.reshape(tel.resolution ** 2, num_modes)
phase_screen = Z_comb * zk_coefficients
phase_screen = np.sum(phase_screen, axis=1)
phase_screen = phase_screen.reshape(tel.resolution, tel.resolution)
opd_screen = phase_screen * src.wavelength/(2*np.pi)

amp_screen = np.std(opd_screen[np.where(tel.pupil > 0)])
ncpa = opd_screen/amp_screen * amp_ncpa

plt.figure(), plt.imshow(ncpa), plt.colorbar(), plt.title('NCPA'), plt.show()

# NCPA psf
src_cam.integrationTime = tel.samplingTime
opd_ncpa = OPD_map(telescope=tel)
opd_ncpa.OPD = ncpa

tel.isPaired = False
tel.resetOPD()
src * tel* opd_ncpa* src_cam
PSF_ncpa = src_cam.frame
plt.figure(), plt.imshow(PSF_ncpa[aa-50:aa+50,aa-50:aa+50]**0.2), plt.colorbar(), plt.title('NCPA PSF'), plt.show()

PSF_diff_norm = PSF_diff/np.sum(PSF_diff)
PSF_ncpa_norm = PSF_ncpa/np.sum(PSF_ncpa)
ss2 = PSF_ncpa_norm.max()/ PSF_diff_norm.max()
ss = computeSR(PSF_diff, PSF_ncpa)
ss3 = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))

plt.figure()
plt.plot(PSF_diff[PSF_diff.shape[0]//2, :], 'r', label='PSF diffraction cam')
plt.plot(PSF_ncpa[PSF_ncpa.shape[0]//2, :], 'b--', label='PSF NCPA cam')
plt.xlim([480, 530])
plt.legend()
plt.show()

#%% test add piston (verification between the science and the sensing)
'''
tel.isPaired = False
tel.resetOPD()
tel + atm
DM.coefs = 0
TTM.coefs = 0
ngs * tel * opd * opd_tt
tel * TTM * DM * wfs
OPD_vis = tel.OPD
total= np.std(OPD_vis[np.where(tel.pupil > 0)]) * 1e9

plt.figure(), plt.imshow(OPD_vis), plt.colorbar(), plt.title('Input (atm + piston)'), plt.show()
plt.figure(), plt.imshow(atm.OPD), plt.colorbar(), plt.title('Input (atm)'), plt.show()
plt.figure(), plt.imshow((OPD_vis-atm.OPD)*tel.pupil), plt.colorbar(), plt.title('Input (piston)'), plt.show()

# science star
opd_sum = opd_tt.OPD + opd.OPD + opd_ncpa.OPD
src * tel * opd_ncpa
OPD_ir = tel.OPD
delta = OPD_ir-OPD_vis
delta_bis = delta -opd_sum
total_sc = np.std(OPD_ir[np.where(tel.pupil > 0)]) * 1e9
plt.figure(), plt.imshow(OPD_ir), plt.colorbar(), plt.title('Input (atm + piston + ncpa + jitter) IR'), plt.show()
plt.figure(), plt.imshow(delta), plt.colorbar(), plt.title('$\Delta OPD (VIS -IR)$'), plt.show()
plt.figure(), plt.imshow(delta_bis), plt.colorbar(), plt.title('$\Delta_{bis}$'), plt.show()

'''

# %% AO loop
#'''
opd_jitter = OPD_map(telescope=tel)
tel.isPaired = False
tel.resetOPD()
DM.coefs = 0
TTM.coefs = 0
ngs.magnitude = mag
src.magnitude = 10
src.coordinates = [0.0, 0]

src_cam.integrationTime = tel.samplingTime
tel.isPaired = False
tel.resetOPD()
src * tel * src_cam
PSF_diff_new = src_cam.frame

# Atmosphere propagation
atm.generateNewPhaseScreen(seed=10)
tel+atm
ngs * tel * opd
tel * TTM * DM * wfs
# These are the calibration data used to close the loop
calib_CL = calib_zonal
M2C_CL = M2C_zonal
calib_CL.nTrunc = 0
reconstructor = Rec#M2C_CL @ calib_CL.Mtrunc #

# loop parameters
ratio = (1/tel.samplingTime)/(freq)
gainCL = 0.4
latency = 2
nLoop = 100

wfs.cam.photonNoise = True # True to add photon noise only, to add RON add the line wfs.cam.readoutNoise = 0.5 for example
if wfs.cam.photonNoise == True:
    wfs.cam.readoutNoise = ron
    wfs.cam.darkCurrent = dark_current

src_cam.integrationTime = tel.samplingTime*10

# allocate memory to save data
SR = np.zeros(nLoop)
SR_PSF = []
total = np.zeros(nLoop)
residual = np.zeros(nLoop)
input_jitter = np.zeros(nLoop)
residual_OPD = []
fitting_rms = np.zeros(nLoop)
wfsSignal = np.arange(0, wfs.nSignal) * 0
signalBuffer = np.zeros((wfs.signal.shape[0], nLoop)) # buffer for the frame delay
SRC_PSF = []

for i in range(nLoop):
    a = time.time()
    atm.update()

    if i%ratio == 0 or i==0:
        opd_jitter.OPD = (tiptilt.x[i+1] * TT_modes[:,0].reshape(Nx,Nx)  + tiptilt.y[i+1] * TT_modes[:,1].reshape(Nx,Nx))*1e-9
        input_jitter[i] = np.std(opd_jitter.OPD[np.where(tel.pupil > 0)]) * 1e9
    
    ngs * tel * opd * opd_jitter
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9

    tel * TTM * DM * wfs

    # fitting
    OPD_fitting_2D, OPD_corr_2D, OPD_turb_2D = getFittingError(tel.OPD, projector, KL_modes, display=False)
    fitting_rms[i] = np.std(OPD_fitting_2D[np.where(tel.pupil > 0)]) * 1e9

    src * tel * opd_ncpa * src_cam

    if i >= latency:
        command = np.matmul(reconstructor, signalBuffer[:, i-latency])
        DM.coefs = DM.coefs - gainCL * command[:DM.nValidAct]
        TTM.coefs = TTM.coefs - gainCL * M2C_Zk_TTM @ command[DM.nValidAct:]

    signalBuffer[:, i] = wfs.signal

    SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
    residual[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9

    if i%10 == 0:
        res = tel.OPD
        res = (res - np.mean(res[np.where(tel.pupil>0)]))*tel.pupil
        residual_OPD.append(res)

    if i>50:
        ao_psf = src_cam.frame
        SRC_PSF.append(ao_psf)
        SR_PSF.append(ss)

    print('Loop' + str(i) + '/' + str(nLoop) + ' Turbulence: ' + str(total[i]) + ' -- Residual:' +
          str(residual[i]) + ' -- Fitting:' + str(fitting_rms[i]) + '\n')

# mean SR
SR_av = np.mean(SR[50:])
# plot the residual and the SR in the science band
plt.figure()
plt.plot(total)
plt.plot(residual)
plt.plot(fitting_rms)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')
plt.show()

plt.figure()
plt.plot(SR, 'b', label='SR from $\phi_{res}$')
plt.hlines(SR_av, xmin = 0, xmax = nLoop, color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Strehl Ratio')
plt.legend()
plt.show()

print('Average SR from residual =\n', SR_av*100)

PSF_LE = np.mean(SRC_PSF,axis=0)
PSF_LE_norm =PSF_LE/ np.max(PSF_LE)  # Normalize PSF
log_psf_norm = np.log10(np.abs(PSF_LE_norm))

(plt.figure(), plt.imshow(log_psf_norm[src_cam.resolution//2-100:src_cam.resolution//2+100,src_cam.resolution//2-100:src_cam.resolution//2+100],
                         extent=[-src_cam.fov_arcsec/2,src_cam.fov_arcsec/2,-src_cam.fov_arcsec/2,src_cam.fov_arcsec/2],cmap = 'gray'),
 plt.colorbar(), plt.title('AO PSF'), plt.show())

#%% Add TT jitter 
# Define Gaussian kernel
from scipy.ndimage import gaussian_filter
size_psf = PSF_LE_norm.shape[0]
x, y = np.meshgrid(np.arange(-size_psf//2, size_psf//2), 
                   np.arange(-size_psf//2, size_psf//2))
sigma = pixel_size_mas/8 # in mas
kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Fourier Transform convolution
PSF_LE_fft = np.fft.fft2(PSF_LE)
kernel_fft = np.fft.fft2(kernel)
prod_LE_fft = PSF_LE_fft * kernel_fft
convolved_psf_LE = np.fft.fftshift(np.fft.ifft2(prod_LE_fft)).real
convolved_psf_LE = convolved_psf_LE /np.sum(kernel)    # Normalize PSF

SR_jitter = computeSR(PSF_diff, convolved_psf)
ii = gaussian_filter(PSF_LE, sigma=sigma)
log_jitter = np.log10(np.abs(convolved_psf_LE))

# Plot results
(plt.figure(), plt.imshow(log_jitter[src_cam.resolution//2-100:src_cam.resolution//2+100,src_cam.resolution//2-100:src_cam.resolution//2+100],
                         extent=[-src_cam.fov_arcsec/2,src_cam.fov_arcsec/2,-src_cam.fov_arcsec/2,src_cam.fov_arcsec/2],cmap = 'gray'),
 plt.colorbar(), plt.title('AO PSF + Jitter'), plt.show())

plt.figure(),plt.imshow(np.log10(np.abs(ii))[src_cam.resolution//2-100:src_cam.resolution//2+100,src_cam.resolution//2-100:src_cam.resolution//2+100],
                         extent=[-src_cam.fov_arcsec/2,src_cam.fov_arcsec/2,-src_cam.fov_arcsec/2,src_cam.fov_arcsec/2],cmap = 'gray'), plt.colorbar(), 


plt.figure()
plt.plot(PSF_LE[PSF_LE.shape[0]//2, :], 'r--', label='PSF')
plt.plot(convolved_psf_LE[convolved_psf_LE.shape[0]//2, :], 'b', label='Jitter PSF')
plt.plot(ii[ii.shape[0]//2, :], 'g--', label='Jitter PSF scipy')
plt.xlim([400, 600])
plt.legend()
plt.show()


#'''
#%% Statistical evaluation of the perf for one mag

'''
path = '/home/mcisse/PycharmProjects/data_pyao/research/ZWFS/simulations_codes/analisis_SCAO_NGS/'
Ns = 5

# These are the calibration data used to close the loop
calib_CL = calib_zonal
M2C_CL = M2C_zonal
calib_CL.nTrunc = 0
reconstructor = Rec#M2C_CL @ calib_CL.Mtrunc #

# loop parameters
gainCL = 0.4
latency = 2
nLoop = 500
wfs.cam.photonNoise = True # True to add photon noise only, to add RON add the line wfs.cam.readoutNoise = 0.5 for example

if wfs.cam.photonNoise == True:
    wfs.cam.readoutNoise = ron
    wfs.cam.darkCurrent = dark_current

magnitude = [6]#np.arange(6, 18, 2)
SR_mag = np.zeros(len(magnitude))
PSF_LE_mag = np.zeros((len(magnitude), src_cam.resolution, src_cam.resolution))
Photon_count = np.zeros(len(magnitude))
c = 0
for mag in magnitude:
    ngs.magnitude = mag
    tel.isPaired = False
    tel.resetOPD()
    ngs * tel
    Photon_count[c] = tel.src.nPhoton
    SR_av = np.zeros(Ns)
    PSF_LE = np.zeros((Ns, src_cam.resolution, src_cam.resolution))
    for s in range(Ns):
        
        tel.isPaired = False
        tel.resetOPD()
        DM.coefs = 0
        TTM.coefs = 0

        # Atmosphere propagation
        atm.generateNewPhaseScreen(seed=s)
        tel + atm
        ngs * tel #* opd
        tel * TTM * DM * wfs

        # allocate memory to save data
        SR = np.zeros(nLoop)
        total = np.zeros(nLoop)
        residual = np.zeros(nLoop)
        fitting_rms = np.zeros(nLoop)
        wfsSignal = np.arange(0, wfs.nSignal) * 0
        signalBuffer = np.zeros((wfs.signal.shape[0], nLoop))  # buffer for the frame delay
        SRC_PSF = []
        command_list = np.zeros((reconstructor.shape[0], nLoop))
        coef_list = np.zeros((DM.nValidAct, nLoop))
        coef_TTM_list = np.zeros((TTM.nValidAct, nLoop))
        for i in range(nLoop):
            a = time.time()
            atm.update()

            total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
            turbPhase = tel.src.phase
            ngs * tel #* opd
            tel * TTM * DM * wfs

            # fitting
            OPD_fitting_2D, OPD_corr_2D, OPD_turb_2D = getFittingError(tel.OPD, projector, KL_modes, display=False)
            fitting_rms[i] = np.std(OPD_fitting_2D[np.where(tel.pupil > 0)]) * 1e9

            src * tel * src_cam

            if i >= latency:
                command = np.matmul(reconstructor, signalBuffer[:, i-latency])
                DM.coefs = DM.coefs - gainCL * command[:DM.nValidAct]
                TTM.coefs = TTM.coefs - gainCL * M2C_Zk_TTM @ command[DM.nValidAct:]

                command_list[:,i] = command 
                coef_list[:,i] = DM.coefs
                coef_TTM_list[:,i] = TTM.coefs

            signalBuffer[:, i] = wfs.signal

            SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
            residual[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
            OPD = tel.OPD[np.where(tel.pupil > 0)]
            if i > 50:
                SRC_PSF.append(src_cam.frame)

        SR_av[s] = np.mean(SR[100:])*100
        PSF_LE[s, :, :] = np.mean(SRC_PSF, axis=0)
        print('Seed' + str(s) + '/ -- Strehl:' + str(SR_av[s]) + '\n')

        name = 'Without_piston_mag'+str(mag)+'_r0_'+str(int(r0*100))+'cm_seed_' + str(s)
        np.save(path+name+'_command.npy', command_list)
        np.save(path+name+'_coef_DM.npy', coef_list)
        np.save(path+name+'_coef_TTM.npy', coef_TTM_list)
        
    SR_mag[c] = np.mean(SR_av)
    PSF_LE_mag[c, :, :] = np.mean(PSF_LE, axis=0)
    c += 1



# plot
plt.figure(), plt.plot(Photon_count,SR_mag, 'rx'), plt.xlabel('Number of photon'), plt.ylabel('Strehl Ratio'), plt.show()
plt.figure(), plt.semilogx(Photon_count,SR_mag, 'rx'), plt.xlabel('Number of photon'), plt.ylabel('Strehl Ratio'), plt.show()

'''

#%% save statistic performance
'''
path = '/home/mcisse/PycharmProjects/data_pyao/research/ZWFS/simulations_codes/analisis_SCAO_NGS/'
np.save(path+'Photon_count_TTM_DM_SH_pistonM1.npy', Photon_count)
np.save(path+'SR_TTM_DM_SH_pistonM1.npy', SR_mag)
np.save(path+'PSE_LE_TTM_DM_SH_pistonM1.npy', PSF_LE_mag)
'''

#%% load and plot data
'''
path = '/home/mcisse/PycharmProjects/data_pyao/research/ZWFS/simulations_codes/analisis_SCAO_NGS/'
photon_piston = np.load(path+'Photon_count_TTM_DM_SH_pistonM1.npy')
SR_mag_piston = np.load(path+'SR_TTM_DM_SH_pistonM1.npy')
PSF_mag_piston = np.load(path+'PSE_LE_TTM_DM_SH_pistonM1.npy')

photon = np.load(path+'Photon_count_simple_DM_SH.npy')
SR_mag = np.load(path+'SR_simple_DM_SH.npy')

# photons per frame and pixel
#tel.pupilReflectivity = 1 so
photon_per_frame = photon_piston * tel.samplingTime*(tel.D/tel.resolution)**2

plt.figure()
plt.semilogx(photon_per_frame,SR_mag_piston, 'rx', label='With M1 Piston')
plt.semilogx(photon_per_frame,SR_mag, 'bo', label='Without M1 Piston')
plt.xlabel('Number of photon per frame and per pixels')
plt.ylabel('Strehl Ratio')
plt.show()


for k in range(PSF_mag_piston.shape[0]):

    stri = 'NGS magnitude = ' + str(6+2*k)
    log_PSF = np.log10(np.abs(PSF_mag_piston[k,:,:]))
    ina = log_PSF.shape[1]//2-100
    inb = log_PSF.shape[1]//2+100
    plt.figure(), plt.imshow(log_PSF[ina:inb, ina:inb], cmap='gray'), plt.colorbar(), plt.title(stri), plt.show()
'''

# histogram command
'''

path = '/home/mcisse/PycharmProjects/data_pyao/research/ZWFS/simulations_codes/analisis_SCAO_NGS/'
command = np.load(path+'With_piston_mag6_r0_16cm_seed_3_command.npy')
coef_DM = np.load(path+'Without_piston_mag6_r0_16cm_seed_3_coef_DM.npy')
coef_TTM = np.load(path+'Without_piston_mag6_r0_16cm_seed_3_coef_TTM.npy')

dm_commands_flat = coef_DM.flatten()
command_flat = command.flatten()
ttm_commands_flat = coef_TTM.flatten()
# Plot the histogram
plt.figure(figsize=(8, 5))
plt.hist(dm_commands_flat, bins=50, density=True, alpha=0.7, color='b')
plt.xlabel("DM Command Value")
plt.ylabel("Density")
plt.title("Histogram of DM coefficients")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(command_flat, bins=50, density=True, alpha=0.7, color='r')
plt.xlabel("Command Value")
plt.ylabel("Density")
plt.title("Histogram of the residuals commands")
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(ttm_commands_flat, bins=50, density=True, alpha=0.7, color='g')
plt.xlabel("TTM Command Value")
plt.ylabel("Density")
plt.title("Histogram of TTM coefficients")
plt.grid(True)
plt.show()

'''



