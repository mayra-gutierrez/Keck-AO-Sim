import time

import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from scipy.ndimage import center_of_mass
from scipy.ndimage import gaussian_filter
from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror as DM_OOPAO
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.MisRegistration import MisRegistration
from OOPAO.Source import Source
from OOPAO.Detector import Detector as Det_OOPAO
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.calibration.getFittingError import *
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.OPD_map import OPD_map
from OOPAO.tools.displayTools import cl_plot, displayMap
import matplotlib.gridspec as gridspec

import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim/")
from keckTel import keckTel, keckStandard
from keckAtm import keckAtm

from simulations_codes.Jitter_TT import RandomTipTilt_Gaussian, SinusoidalTipTilt
from simulations_codes.SH import ShackHartmann_modifNoise
from simulations_codes.Imat_SHWFS import InteractionMatrix_test
from simulations_codes.research_copy.KAO_parameter_file_research_copy import initializeParameterFile

from simulations_codes.ZWFS_toolbox.wfSensors import *

#%% Parameters
param = initializeParameterFile()

#%% Telescope and atmosphere
keck_object = keckTel.create('keck', resolution=param['resolution'],samplingTime=param['samplingTime'], return_segments=True)
tel = keck_object.tel

KSM = SegmentedDeformableMirror(keck_object.keck_segments)
s1 = np.array(KSM.segments)
seg_vect2D = s1.reshape(s1.shape[0],int(np.sqrt(s1.shape[1])),int(np.sqrt(s1.shape[1])))
segments_vect1D = seg_vect2D.reshape(seg_vect2D.shape[0],seg_vect2D.shape[1]**2)
proj_seg = np.linalg.pinv(segments_vect1D)

seg = np.arange(36)
seg_amp = np.zeros(36)
KSM.set_segment_actuators(seg, 0.5*1e-9,0,0)
flat_ksm = KSM.opd.shaped

KSM.set_segment_actuators(10,250*1e-9,0,0)
opd_phasing = KSM.opd.shaped
cophase_amp = np.std(opd_phasing[np.where(flat_ksm != 0)])
opd_seg = OPD_map(telescope=tel)
opd_seg.OPD = opd_phasing * tel.pupil
plt.figure(), plt.imshow(opd_seg.OPD), plt.title('One Segment'), plt.colorbar(), plt.show(block=False)

opd_M1 = OPD_map(telescope=tel)
opd_M1.OPD = opd_seg.OPD    

Diam = keck_object.tel.D
Dobs = 2.6 #obs central in m
obs = Dobs/Diam
tel_circ = Telescope(diameter=Diam,
            resolution=param['resolution'],
            samplingTime=param['samplingTime'],
            centralObstruction=obs)
                    
ngs=Source(optBand   = param['opticalBand_guide'],\
               magnitude = param['magnitude_guide'], \
               coordinates=param['ngs_coordinate'])
               
src = Source(optBand='K', 
             magnitude=param['science_magnitude'])  
                 
ngs*tel
ngs*tel_circ

atm_ = keckAtm.create(telescope=tel,\
                            atm_type='custom',\
                            r0 = param['r0'],\
                            L0 = param['L0'],\
                            wind_speeds = param['windSpeed'],\
                            fractional_r0 = param['fractionnalR0'],\
                            wind_directions = param['windDirection'],\
                            altitudes = param['altitude']) 
atm = atm_.atm
atm.update()

# DM 
nAct = param['nActuator']
pitch = param['diameter']/(param['nActuator']-1)

grid=np.mgrid[0:nAct,0:nAct]
rgrid=np.sqrt((grid[0]-nAct/2+0.5)**2+(grid[1]-nAct/2+0.5)**2)
mask = np.zeros((nAct,nAct)).astype(np.float32)
mask[np.where(rgrid<nAct/2)]=1

x = np.linspace(-tel.D/2, tel.D/2, nAct) # in meter
valid = (mask == 1)
X, Y = np.meshgrid(x, x)
# Extract the (x,y) coordinates only where the mask is 1
xc = X[valid]
yc = Y[valid]
coords_final = np.column_stack((xc, yc))
#param['mechanicalCoupling']
DM = DM_OOPAO(telescope=tel,
              nSubap=nAct-1,
              mechCoupling=param['mechanicalCoupling'],
              coordinates=coords_final,
              pitch=pitch)
dm_modes= DM.modes
projector_DM = np.linalg.pinv(dm_modes)

plt.figure()
plt.imshow(np.reshape(np.sum(dm_modes**5,axis=1),[tel.resolution,tel.resolution]).T +
            tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.plot(DM.coordinates[:,0],DM.coordinates[:,1],'rx')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates keck')
plt.colorbar()
plt.show(block=False)    

# Zernike
nm = 10
Z = Zernike(tel, nm)
Z.computeZernike(tel)
M2C = np.linalg.pinv(np.squeeze(DM.modes[tel.pupilLogical, :])) @ Z.modes
C2M = np.linalg.pinv(M2C)

DM.coefs = M2C
tel * DM
Zk_modes = tel.OPD.reshape(tel.resolution ** 2, M2C.shape[1])
projector_zk = np.linalg.pinv(Zk_modes)
Zk_2D = Zk_modes.reshape(param['resolution'],param['resolution'],nm)

tel.resetOPD()
DM.coefs = 0
M2C_KL_ = compute_KL_basis(tel, atm, DM)
M2C_KL = M2C_KL_
DM.coefs = M2C_KL
tel * DM
KL = tel.OPD.reshape(tel.resolution ** 2, M2C_KL.shape[1])
projector_kl = np.linalg.pinv(KL)
KL_2D = KL.reshape((param['resolution'],param['resolution'],KL.shape[1]))

#%% TTM 
TT_modes_ = Z.modesFullRes[:,:,:2].reshape((param['resolution']**2,2))
TT_modes = TT_modes_ * ngs.wavelength/(2*np.pi)
TTM = DM_OOPAO(telescope=tel,
                    nSubap=2,
                    mechCoupling=param['mechanicalCoupling'],
                    modes=TT_modes)
                    
# Kolmogorov covariance matrix

# act_pos: (N,2) actuator (x,y) positions in meters
d = np.linalg.norm(coords_final[:,None,:] - coords_final[None,:,:], axis=2)
Dphi = 6.88 * (d / atm.r0)**(5/3)
Cphi = -0.5*Dphi
Cphi -= np.mean(Cphi)
Cphi /=1e1

invphi = np.linalg.inv(Cphi)

path = '/home/mcisse/keckAOSim/keckSim/'
filename_cphi = os.path.join(path, "invcov.lsp")
filename_w = os.path.join(path, "ping0.map")

invcov = np.fromfile(filename_cphi, dtype='>f4', count=349*349)
invcov = invcov.reshape((349, 349))

cov = np.linalg.inv(invcov)

#%% load matrix 


weight0 = np.fromfile(filename_w, dtype='uint8', count=304)

#%% SHWFS
path_bench = '/home/mcisse/keckAOSim/keckSim/data_bench/'
bin_map = np.load(f'{path_bench}binary_sub_apertures_map.npy')
bin_map =  bin_map.astype(bool)

nsub = param['nSubaperture']
grid=np.mgrid[0:nsub,0:nsub]
rgrid=np.sqrt((grid[0]-nsub/2+0.5)**2+(grid[1]-nsub/2+0.5)**2)
mask = np.zeros((nsub,nsub)).astype(np.float32)
mask[np.where(rgrid<nsub/2)]=1

mask2 = np.zeros((nsub,nsub)).astype(np.float32)
mask2[np.where(rgrid<2)]=1

sub_mask = mask-mask2

wfs = ShackHartmann_modifNoise(nSubap=param['nSubaperture'],
                                   telescope=tel_circ,
                                   lightRatio=0.5,
                                   valid_subapertures = bin_map,
                                   pixel_scale=param['plateScale'],
                                   shannon_sampling=param['shannon'],
                                   n_pixel_per_subaperture=param['nPixelPerSubap'])
                                   
map_ph = wfs.photon_per_subaperture_2D
map_val = wfs.valid_subapertures

tel + atm
ngs*tel*wfs
plt.figure(), plt.imshow(wfs.cam.frame), plt.colorbar(), plt.title('SHWFS frame'),plt.show(block=False)
plt.figure(), plt.imshow(wfs.signal_2D), plt.colorbar(), plt.title('SHWFS slope'),plt.show(block=False)
                                   
# Science detector
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
src_cam = NIRC2

#%% Interaction matrix
tel.isPaired = False
tel.resetOPD()
DM.coefs = 0
TTM.coefs = 0
ngs * tel * TTM * DM * wfs
M2C_zonal = np.eye(DM.nValidAct)
M2C_TTM = np.eye(TTM.nValidAct)
stroke = param['stroke']

# single_pass = False to force the push pull. it cleans the IM from the outer ring of the slopes mask

calib_zonal = InteractionMatrix_test(ngs=ngs, \
                                        tel=tel, \
                                        dm=DM, \
                                        wfs=wfs, \
                                        M2C=M2C_zonal, \
                                        stroke=stroke, \
                                        nMeasurements=100, \
                                        noise='off',
                                        single_pass = False)

calib_TTM = InteractionMatrix_test(ngs=ngs, \
                                        tel=tel, \
                                        dm=TTM, \
                                        wfs=wfs, \
                                        M2C=M2C_TTM, \
                                        stroke=stroke, \
                                        nMeasurements=1, \
                                        noise='off',
                                        display_slopes = False,
                                        single_pass = False)
                                        
# filtering TT

g1 = np.zeros((wfs.nSignal,1))
g2 = np.zeros((wfs.nSignal,1))

g1[:wfs.nSignal//2,:]=1
g2[wfs.nSignal//2:,:]=1

Gt = np.hstack([g1,g2])
Gt = Gt/np.linalg.norm(Gt)
reg = Gt.T @ Gt
reg /= 10 
PGt = Gt @ np.linalg.inv(reg) @ Gt.T #Gt @ Gt.T

# piston
bp = np.ones((DM.nValidAct,1))
bp /=np.sqrt(DM.nValidAct)
reg_p = bp.T @ bp
Pp = bp @ np.linalg.inv(reg_p) @ bp.T

D_noTT = (np.eye(PGt.shape[0])-PGt) @ calib_zonal.D @ (np.eye(Pp.shape[0])-Pp)

#%% Reconstructor

#computation of the weight of the subap with respect to their illumination
tel.isPaired = False
tel.resetOPD()
DM.coefs = 0
ngs * tel * DM * wfs
int_subap = wfs.maps_intensity
weight = np.sum(int_subap, axis=(1, 2))
max_weight = weight.max()
weight = weight/max_weight
weight[np.where(weight<1e-3)] = 0

weight_map = np.zeros_like(map_val, dtype=float)
weight_map[map_val] = weight

w = weight
W0 = np.diag(w)
W1 = np.diag(w) * 0
W = np.asarray(np.bmat([[W0, W1], [W1, W0]]))

alpha = 5
eps_mat = 1
#TTM
H_ttm = np.zeros((calib_TTM.D.shape[0], calib_TTM.D.shape[1]))
H_ttm[:, :calib_TTM.D.shape[1]] = calib_TTM.D
Dw_ttm = W @ H_ttm
Mat_inter_ttm = H_ttm.transpose()@ Dw_ttm
Mat_inter_ttm += eps_mat * np.eye(Mat_inter_ttm.shape[0])
Attm = np.linalg.pinv(Mat_inter_ttm)
Bttm = H_ttm.transpose()@ W

Rec_int2 = Attm @ Bttm

# DM
Hnt = np.zeros((D_noTT.shape[0], D_noTT.shape[1]))
Hnt[:, :D_noTT.shape[1]] = D_noTT
Dw_nt = W @ Hnt
Mat_inter_nt = Hnt.T @ Dw_nt
Mat_inter_nt += eps_mat * np.eye(Mat_inter_nt.shape[0])
cond = np.linalg.cond(Mat_inter_nt)
print("Mat_inter cond:", cond)

A_nt = np.linalg.pinv(Mat_inter_nt)
Aphi_nt = np.linalg.pinv(Mat_inter_nt+alpha*invphi)

B_nt = Hnt.transpose()@ W
Rec_int_nt = A_nt @ B_nt
Rec_int_phi1_nt = Aphi_nt @ B_nt

#Rec_nt = np.vstack([Rec_int_nt,Rec_int2])
Rec_phi_nt = np.vstack([Rec_int_phi1_nt,Rec_int2])

# plots & check
#check slopes
kk = 10
slope_map = np.zeros_like(map_val, dtype=float)
slope_map[map_val] = Dw_nt[:304,kk]

slope_map_ini = np.zeros_like(map_val, dtype=float)
slope_map_ini[map_val] = calib_zonal.D[:304,kk]

plt.figure()
im = plt.imshow(weight_map, cmap="viridis")
plt.colorbar(im, label="Weight")
valid_inds = np.argwhere(map_val)
# Go through each valid subap and annotate
for idx, (i, j) in enumerate(valid_inds):
    val = weight_map[i, j]
    plt.text(j, i, f"{idx}: {val:.1f}",
        ha="center", va="center",
        color="w", fontsize=6)

plt.title("Weight Map with Subap Index & Values")
plt.show(block=False)

#%% Reconstruction one screen
#'''
basis = Zk_2D
proj = projector_zk

tel.isPaired = False
tel.resetOPD()
DM.coefs = 0
TTM.coefs = 0
opd_input = OPD_map(telescope=tel)
screen = (basis[:,:,1]*2+basis[:,:,0]*0.5+basis[:,:,7]*1)*0
opd_input.OPD = screen*ngs.wavelength/(2*np.pi)

tel.isPaired = True # DO NOT CHANGE THIS
tel.resetOPD()
tel+atm
DM.coefs = 0
TTM.coefs = 0
ngs * tel * opd_input * TTM * DM *wfs  
opd_ini= tel.OPD 
amp_ini = np.std(opd_ini) * 1e9
coef_ini = proj @ opd_ini.flatten()

plt.figure(), plt.imshow(opd_ini), plt.colorbar(), plt.title('Initial OPD'), plt.show(block=False)

signal = wfs.signal
slope_z1 = wfs.signal
plt.figure(), plt.imshow(wfs.signal_2D), plt.colorbar(), plt.title('slopes map screen'), plt.show(block=False)

command = Rec_phi_nt @ signal
cc_dm = command[:DM.nValidAct] 
DM.coefs = DM.coefs-cc_dm
TTM.coefs = TTM.coefs-command[DM.nValidAct:]
#tel.resetOPD()
ngs * tel * opd_input * TTM * DM
opd_res = tel.OPD

res = np.std(opd_res) * 1e9
coef_res = proj @ opd_res.flatten()
coef_tt = proj @ TTM.OPD.flatten()
coef_tt_dm = proj @ DM.OPD.flatten()
plt.figure(), plt.imshow(opd_res), plt.colorbar(), plt.title('residual OPD'), plt.show(block=False)
plt.figure(), plt.imshow(DM.OPD*tel.pupil), plt.colorbar(), plt.title('DM OPD'), plt.show(block=False)
plt.figure(), plt.imshow(TTM.OPD), plt.colorbar(), plt.title('TTM'), plt.show(block=False)

plt.figure()
plt.plot(coef_ini[:10]*1e9,'rx',label=f'Input')
plt.plot(coef_res[:10]*1e9,'bo',label=f'Residual')
plt.plot(coef_tt[:10]*1e9,'kd',label=f'TT est')
plt.plot(coef_tt_dm[:10]*1e9,'mo',label=f'DM est')
plt.legend()
plt.show(block=False)

plt.figure()
im = plt.imshow(DM.OPD*tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.colorbar(im, label="[nm]")
valid_act = DM.coordinates
# Go through each valid subap and annotate
for idx, (i, j) in enumerate(valid_act):
    val = DM.coefs[idx]
    plt.text(j, i, f"{idx}",
        ha="center", va="center",
        color="w", fontsize=6)
plt.plot(DM.coordinates[:,0],DM.coordinates[:,1],'rx')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates keck')
plt.show(block=False) 

print("Input tilt :", amp_ini)
print("Residual tilt :", res)

#'''
'''
plt.close('all')
wfs.is_geometric = False
tel.isPaired = False 
tel.resetOPD()
DM.coefs = 0
TTM.coefs = 0

atm.generateNewPhaseScreen(seed=10)
tel+atm
ngs * tel
tel * TTM * DM * wfs
opd_M1.OPD = opd_seg.OPD  
KSM.set_segment_actuators(seg,0,0,0)  
for k in range(10):
    atm.update()
    ngs * tel * opd_M1 * TTM * DM * wfs
    
    tel_OPD = tel.OPD
    M1_loop = tel_OPD-atm.OPD-DM.OPD
    plt.figure(), plt.imshow(M1_loop*1e9),plt.colorbar(), plt.show(block=False)

    flat_ksm = KSM.opd.shaped

    KSM.set_segment_actuators(1,50*1e-9,0,0)
    opd_phasing = KSM.opd.shaped
    cophase_amp = np.std(opd_phasing[np.where(flat_ksm != 0)])
    opd_M1.OPD = opd_M1.OPD - opd_phasing * tel.pupil
    DM.coefs = DM.coefs-np.random.rand(DM.nValidAct)*1e-9
    
'''
#%% AO loop
'''
plt.close('all')
wfs.is_geometric = False
tel.isPaired = True # do not change this
tel.resetOPD()
DM.coefs = 0
TTM.coefs = 0

atm.generateNewPhaseScreen(seed=10)
tel+atm
ngs * tel
tel * TTM * DM * wfs

nLoop = 2000
gain = 0.3
latency = 1
leak=0.985

SR = np.zeros(nLoop)
total = np.zeros(nLoop)
residual = np.zeros(nLoop)
fitting_rms = np.zeros(nLoop)
wfsSignal = np.arange(0, wfs.nSignal) * 0
signalBuffer = np.zeros((wfs.signal.shape[0], nLoop)) # buffer for the frame delay
SRC_PSF = []

plot_obj = cl_plot(list_fig  = [atm.OPD,
                                tel.mean_removed_OPD,
                                wfs.cam.frame,
                                DM.OPD,
                                TTM.OPD,
                                [[0,0],[0,0]],
                                np.log10(src_cam.frame)],
                   type_fig          = ['imshow','imshow','imshow','imshow','imshow','plot','imshow'],
                   list_title        = ['Turbulence [nm]','NGS residual [m]','WFS Detector','DM OPD [nm]','TTM OPD [nm]',None,'PSF'],
                   list_legend       = [None,None,None,None,None,['SRC@'+str(src.coordinates[0])+'"','NGS@'+str(ngs.coordinates[0])+'"'],None,None],
                   list_label        = [None,None,None,None,None,['Time','WFE [nm]'],['Science PSF','']],
                   n_subplot         = [4,2],
                   list_display_axis = [None,None,None,None,None,True,None],
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)


for i in range(nLoop):
    a = time.time()
    atm.update()
    total[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    ngs * tel * TTM * DM * wfs

    OPD_fitting_2D, OPD_corr_2D, OPD_turb_2D = getFittingError(tel.OPD, projector_DM, dm_modes, display=False)
    fitting_rms[i] = np.std(OPD_fitting_2D[np.where(tel.pupil > 0)]) * 1e9

    src * tel * src_cam
    
    if latency ==1:        
        wfsSignal = AO_sys.wfs.signal

    command = np.matmul(Rec_phi_nt, signalBuffer[:, i-latency])
    DM.coefs = leak*DM.coefs - gain * command[:DM.nValidAct]
    TTM.coefs = TTM.coefs - gain * command[DM.nValidAct:]
    
    if latency ==2:        
        wfsSignal = AO_sys.wfs.signal

    SR[i] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil == 1)]))
    residual[i] = np.std(tel.OPD[np.where(tel.pupil > 0)]) * 1e9
    
    psf = src_cam.frame[:]
    if i>1:
        pp = np.abs(psf)
        pp[pp<=0] = np.nan
        psf_plot = np.log10(pp)
            
        plot_obj.list_lim = [None,None,None,None,None,None,[psf_plot.min(), psf_plot.max()],None]        
        # update title
        plot_obj.list_title = ['Turbulence '+str(np.round(total[i]))+'[nm]',
                               'AO residual '+str(np.round(residual[i]))+'[nm]',
                               'WFS Detector',
                               'DM OPD',
                               'TTM OPD',
                               None,
                               'PSF']

        cl_plot(list_fig   = [1e9*atm.OPD,1e9*tel.OPD,wfs.cam.frame,DM.OPD*tel.pupil*1e9, TTM.OPD*tel.pupil*1e9, [np.arange(i+1),residual[:i+1]],psf_plot],plt_obj = plot_obj)
            
        plt.pause(0.01)
        if plot_obj.keep_going is False:
            break
            
    if i>50:
        SRC_PSF.append(psf)

    print('Loop' + str(i) + '/' + str(nLoop) + ' Turbulence: ' + str(total[i]) + ' -- Residual:' +
          str(residual[i]) + ' -- Fitting:' + str(fitting_rms[i]) + '\n')

plt.figure()
plt.plot(total)
plt.plot(residual)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')
plt.show(block=False)

plt.figure()
plt.plot(SR)
plt.xlabel('Time')
plt.ylabel('Strehl Ratio')
plt.show(block=False)

print('Average SR = %d\n', np.mean(SR[50:]*100))

PSF_LE = np.mean(SRC_PSF,axis=0)
PSF_LE_norm =PSF_LE/ np.max(PSF_LE)  # Normalize PSF
log_PSF = np.log10(np.abs(PSF_LE))
beg = src_cam.resolution//2 - 100
end = src_cam.resolution//2 + 100
plt.figure(), plt.imshow(log_PSF[beg:end,beg:end],cmap = 'gray'), plt.colorbar(), plt.show(block=False)

    

#'''









