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

from simulations_codes.Jitter_TT import RandomTipTilt_Gaussian, SinusoidalTipTilt
from simulations_codes.SH import ShackHartmann_modifNoise
from simulations_codes.Imat_SHWFS import InteractionMatrix_test

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

# %% -----------------------     Circular TELESCOPE   ----------------------------------
D = keck_object.tel.D
Dobs = 2.6 #obs central in m
obs = Dobs/D
tel_calib = Telescope(diameter=D,
            resolution=Nx,
            samplingTime=samp,
            centralObstruction=obs)
#%% 
type_simu = 'OOPAO'

if type_simu == 'keck':
    tel = tel_calib
else:
    tel = keck_object.tel
# %% -----------------------     NGS   ----------------------------------
mag = 5
ngs = Source(optBand='V',  # Optical band (see photometry.py)
             magnitude=mag)  # Source Magnitude

src = Source(optBand='K',  # Optical band (see photometry.py)
             magnitude=mag)  # Source Magnitude

LD_mas = src.wavelength/D * 206265 * 1000 # in mas

# propagate the NGS to the telescope object
ngs * tel

tel.computePSF(zeroPaddingFactor=4)
PSF = tel.PSF
size_psf = PSF.shape[0]//2

plt.figure(), plt.imshow(PSF[size_psf-50:size_psf+50,size_psf-50:size_psf+50]**0.2,cmap='gray'), plt.colorbar(), plt.show()


# %% -----------------------     ATMOSPHERE   ----------------------------------

atm = keckAtm.create(telescope=tel, atm_type='maunakea') # median condition 
#atm_type = 'custom' to change the r0,L0,etc...
atm.atm.update()

# %% -----------------------     DEFORMABLE MIRROR   ----------------------------------
nAct = 61
print(f'HAKA version nact{nAct}')
pitch = keck_object.tel.D / nAct
'''
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


coords_final = np.column_stack((xc, yc))
DM= DeformableMirror(telescope=tel,
                     nSubap=nAct - 1,
                     mechCoupling=0.1458,
                     coordinates=coords_final,
                     pitch=pitch,
                     floating_precision=32)
'''
dm = DeformableMirror(telescope=tel,
                nSubap=nAct-1,
                mechCoupling=0.1458,
                pitch=pitch,
                floating_precision=32)

pup = tel.pupil.flatten()#np.transpose(tel.pupil).flatten()#
IF = dm.modes.astype(np.float32) * pup[:, np.newaxis]
IF_sum = np.sum(IF, axis=0)
IF_max = IF_sum.max()
valid_index = np.where(IF_sum >= 0.1*IF_max)[0]
IF_filtered = IF[:, valid_index]
dm_modes = IF_filtered
dm_modes = dm_modes.astype(np.float32)
projector_DM = np.linalg.pinv(dm_modes)

DM = DeformableMirror(telescope=tel,
                nSubap=nAct-1,
                mechCoupling=0.1458,
                pitch=pitch, 
                modes=IF_filtered)
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

Z = Zernike(tel, 10)
Z.computeZernike(tel)
M2C_zernike = np.linalg.pinv(np.squeeze(DM.modes[tel.pupilLogical, :])) @ Z.modes

DM.coefs = M2C_zernike[:,:2]
TT_modes = Z.modesFullRes[:,:,:2].reshape((tel.resolution**2,2))

Zk_modes = Z.modesFullRes.reshape(tel.resolution ** 2, Z.modesFullRes.shape[-1])
projector_zk = np.linalg.pinv(Zk_modes)

TTM_pur = DeformableMirror(telescope=tel,
                            nSubap=2,
                            mechCoupling=0.1458,
                            coordinates=None,
                            pitch=pitch,
                            modes=TT_modes)

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

# %% -----------------------     SH WFS define the SH with circular apperature  ----------------------------------
nsubap = nAct-1
# make sure tel and atm are separated to initialize the PWFS
tel.isPaired = False
tel.resetOPD()
ngs * tel
pix_scale = 0.8 # in arcsec
npix_sub = 4

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
                                telescope=tel,
                                lightRatio=0.36,
                                pixel_scale=pix_scale,
                                shannon_sampling=False,
                                n_pixel_per_subaperture=npix_sub)

ngs *tel * wfs
frame_keck = wfs.cam.frame
print(f'Signal shape keck {wfs.signal.shape}')
plt.figure(), plt.imshow(frame_keck), plt.colorbar(), plt.title('WFS Camera Frame keck style'), plt.show()
plt.figure(), plt.imshow(wfs.signal_2D), plt.colorbar(), plt.title('WFS slopes keck style'), plt.show()

# %% -----------------------     Science detector  ----------------------------------
NIRC2 = Detector(nRes=1024,
                 integrationTime=tel.samplingTime*10,
                 bits=14,
                 sensor='CCD',
                 QE=0.92,
                 binning=1,
                 psf_sampling=4,
                 darkCurrent=0,
                 readoutNoise=0.0,
                 photonNoise=False)
src_cam = NIRC2
src_cam.integrationTime = tel.samplingTime
tel.isPaired = False
tel.resetOPD()
src * tel * src_cam
PSF_diff = src_cam.frame

size_psf = PSF_diff.shape[0]//2
plt.figure(), plt.imshow(PSF_diff[size_psf-50:size_psf+50,size_psf-50:size_psf+50]**0.2), plt.colorbar(), plt.title('Diffraction limited PSF'), plt.show()

# %% -----------------------     IMAT  ----------------------------------

stroke = 1e-9  # amplitude of the modes in m
M2C_zonal = np.eye(DM.nValidAct)
wfs.is_geometric = False
tel.resetOPD() # to remove the default OPD of the co-phasing error

# zonal interaction matrix
calib_zonal = InteractionMatrix_test(ngs=ngs, \
                                    tel=tel, \
                                    dm=DM, \
                                    wfs=wfs, \
                                    M2C=M2C_zonal, \
                                    stroke=stroke, \
                                    nMeasurements=100, \
                                    noise='off')

plt.figure(), plt.plot(calib_zonal.eigenValues,'rx'), plt.ylabel('eigenValues zonal'), plt.show()
plt.figure(), plt.imshow(calib_zonal.D), plt.colorbar(), plt.title('Zonal Imat'), plt.show()

# TTM calib
M2C_TTM = np.eye(2)
calib_TTM = InteractionMatrix_test(ngs=ngs, \
                                    tel=tel, \
                                    dm=TTM_pur, \
                                    wfs=wfs, \
                                    M2C=M2C_TTM, \
                                    stroke=stroke, \
                                    nMeasurements=1, \
                                    noise='off')

plt.figure(), plt.plot(calib_TTM.eigenValues,'bo'), plt.ylabel('eigenValues zonal'), plt.show()

# %% -----------------------     KECK reconstructor  ----------------------------------
# reset the OPD
tel.isPaired = False
tel.resetOPD()
DM.coefs = 0
ngs * tel * DM * wfs
plt.figure(), plt.imshow(wfs.signal_2D), plt.colorbar(), plt.title('WFS new slopes'), plt.show()

# computation of the weight of the subap with respect to their illumination
int_subap = wfs.maps_intensity
weight = np.sum(int_subap, axis=(1, 2))
weight = weight/weight.max()
W0 = np.diag(weight)
W1 = np.diag(weight) * 0
W = np.asarray(np.bmat([[W0, W1], [W1, W0]]))
plt.figure(), plt.plot(weight), plt.title('Weight'), plt.show()

ww = np.diag(W)
inv_diag = np.where(ww != 0, 1 / ww, 0)
inv_W = np.diag(inv_diag)

calib_zonal.nTrunc = 0
Dtrunc_ =  calib_zonal.Dtrunc #
H = np.zeros((Dtrunc_.shape[0], Dtrunc_.shape[1]))
H[:, :Dtrunc_.shape[1]] = Dtrunc_
Mat_inter = H.transpose()@ inv_W @ H
A = np.linalg.inv(Mat_inter)
B = H.transpose()@ inv_W
Rec_int1 = A @ B

H = np.zeros((calib_TTM.D.shape[0], calib_TTM.D.shape[1]))
H[:, :calib_TTM.D.shape[1]] = calib_TTM.D
Mat_inter = H.transpose()@ inv_W @ H
A = np.linalg.inv(Mat_inter)
B = H.transpose()@ inv_W
Rec_int2 = A @ B
Rec_int = np.vstack([Rec_int1,Rec_int2])

DM_rec = P_orth_TTM @ Rec_int[:DM.nValidAct,:]
Rec = np.vstack([DM_rec,Rec_int[DM.nValidAct:,:]])

#%% reconstruction one mode

tel.isPaired = False
tel.resetOPD()
DM.coefs = 0
TTM_pur.coefs = 0

DM.coefs = M2C_zernike[:,1]*2e-9 + M2C_zernike[:, 0]*1e-9 + M2C_zernike[:, 2]*1e-9
opd_input = OPD_map(telescope=tel)
opd_input.OPD = DM.OPD * tel.pupil

tel.isPaired = True # DO NOT CHANGE THIS
tel.resetOPD()
DM.coefs = 0
TTM_pur.coefs = 0
ngs * tel * opd_input * TTM_pur * DM *wfs  
ini= tel.OPD 
coef_in = projector_zk@ini.flatten()
plt.figure(), plt.imshow(ini), plt.colorbar(), plt.title('Tel OPD'), plt.show()

signal = wfs.signal
command = Rec @ signal
DM.coefs = DM.coefs-command[:DM.nValidAct]
TTM_pur.coefs = TTM_pur.coefs-command[DM.nValidAct:]

coef_dm = projector_zk@DM.OPD.flatten()
coef_tt = projector_zk@TTM_pur.OPD.flatten()

plt.figure(), plt.imshow(wfs.signal_2D), plt.colorbar(), plt.title('slopes map'), plt.show()
plt.figure(), plt.imshow(DM.OPD*tel.pupil), plt.colorbar(), plt.title('DM OPD rec'), plt.show()
plt.figure(), plt.imshow(TTM_pur.OPD*tel.pupil), plt.colorbar(), plt.title('TTM OPD est'), plt.show()

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

# %% -----------------------     AO loop  ----------------------------------
wfs.is_geometric = False
tel.isPaired = False
tel.resetOPD()
DM.coefs = 0
TTM_pur.coefs = 0

wfs.cam.photonNoise = True # True to add photon noise only, to add RON add the line wfs.cam.readoutNoise = 0.5 for example
if wfs.cam.photonNoise == True:
    wfs.cam.readoutNoise = ron
    wfs.cam.darkCurrent = dark_current

src_cam.integrationTime = tel.samplingTime
atm.atm.generateNewPhaseScreen(seed=10)
tel+atm.atm
ngs * tel
tel * TTM_pur * DM * wfs

nLoop = 200
gain = 0.3
latency = 2

SR = np.zeros(nLoop)
total = np.zeros(nLoop)
residual = np.zeros(nLoop)
fitting_rms = np.zeros(nLoop)
wfsSignal = np.arange(0, wfs.nSignal) * 0
signalBuffer = np.zeros((wfs.signal.shape[0], nLoop)) # buffer for the frame delay
SRC_PSF = []

for i in range(nLoop):
    a = time.time()
    atm.atm.update()
    ngs * tel 
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    tel * TTM_pur* DM * wfs

    OPD_fitting_2D, OPD_corr_2D, OPD_turb_2D = getFittingError(tel.OPD, projector_DM, dm_modes, display=False)
    fitting_rms[i] = np.std(OPD_fitting_2D[np.where(tel.pupil > 0)]) * 1e9

    src * tel * src_cam
    
    if i >= latency:
        command = np.matmul(Rec, signalBuffer[:, i-latency])
        DM.coefs = DM.coefs - gain * command[:DM.nValidAct] 
        TTM_pur.coefs = TTM_pur.coefs - gain * (command[DM.nValidAct:])

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

# %%
