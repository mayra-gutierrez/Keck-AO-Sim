import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pfits
from scipy import sparse
from scipy.ndimage import binary_erosion
from scipy.ndimage import center_of_mass
import scipy.sparse
import pickle

from hcipy import *
from scipy.ndimage import center_of_mass

from OOPAO.DeformableMirror import DeformableMirror as DM_OOPAO
from OOPAO.MisRegistration import MisRegistration
from OOPAO.ShackHartmann import ShackHartmann
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.Source import Source
from OOPAO.Detector import Detector as Det_OOPAO
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from OOPAO.calibration.getFittingError import *
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.OPD_map import OPD_map
from OOPAO.ShackHartmann import ShackHartmann

import os, sys
os.chdir("/Users/mayragutierrez/home/lab/maygut/keckAOSim2/keckSim")
from keckTel import keckTel,keckStandard
from keckAtm import keckAtm

from simulations_codes.SH import ShackHartmann_modifNoise
from simulations_codes.Imat_SHWFS import InteractionMatrix_test

from simulations_codes.ZWFS_toolbox.wfSensors import *

def initialize_AO_hardware(param):

    # -----------------------  Keck apperture ----------------------------------
    if param['nSubaperture'] == 20:
        param['resolution'] = 280
        print('Xinetics: Putting the telescope resolution to 280 pixels accross the pupil to match the number of subaperture')
    else:
        param['resolution'] = 336
        print('HAKA: Putting the telescope resolution to 336 pixels accross the pupil to match the number of subaperture')
        
    keck_object = keckTel.create('keck', resolution=param['resolution'],samplingTime=param['samplingTime'], return_segments=True)
    
    keck_object_bis = keckStandard.make_keck_aperture(with_spiders=False,return_segments=False)
    grid = make_pupil_grid(param['resolution'], diameter=keck_object.tel.D)
    pupil_full = keck_object_bis(grid).shaped.astype(bool)
    pup_crop = binary_erosion(pupil_full, iterations=1)

    # -----------------------      KSM       ----------------------------------
    KSM = SegmentedDeformableMirror(keck_object.keck_segments)
    s1 = np.array(KSM.segments)
    seg_vect2D = s1.reshape(s1.shape[0],int(np.sqrt(s1.shape[1])),int(np.sqrt(s1.shape[1])))
    segments_vect1D = seg_vect2D.reshape(seg_vect2D.shape[0],seg_vect2D.shape[1]**2)
    proj_seg = np.linalg.pinv(segments_vect1D)

    # -----------------------     TELESCOPES   ----------------------------------
    tel = keck_object.tel
    
    Diam = keck_object.tel.D
    Dobs = 2.6 #obs central in m
    obs = Dobs/Diam
    tel_circ = Telescope(diameter=Diam,
            resolution=param['resolution'],
            samplingTime=param['samplingTime'],
            centralObstruction=obs)
    # -----------------------     NGS        ----------------------------------
    ngs=Source(optBand   = param['opticalBand_guide'],\
               magnitude = param['magnitude_guide'], \
               coordinates=param['ngs_coordinate'])
    
    science=Source(optBand   = param['science_opticalBand'],\
                   magnitude = param['science_magnitude'],\
                   coordinates = param['science_coordinate'])
    
    LD_mas = science.wavelength/tel.D * 206265 * 1000 # in mas
    ngs*tel

    # -----------------------     Atmosphere   ----------------------------------
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
    # -----------------------     DEFORMABLE MIRRORS   ----------------------------------
    nAct = param['nActuator']
    pitch = param['diameter']/(param['nActuator']-1)
    # mis-registrations object
    misReg = MisRegistration(param)

    if nAct >21:
        print('HAKA AO')
        
    else:
        print('Current AO')
    
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

    DM = DM_OOPAO(telescope=tel,
              nSubap=nAct-1,
              mechCoupling=param['mechanicalCoupling'],
              misReg=misReg,
              coordinates=coords_final,
              pitch=pitch)
    dm_modes= DM.modes
    projector_DM = np.linalg.pinv(dm_modes)    

    #  -----------------------     TTM basis  ----------------------------------
    Z = Zernike(tel, param['nb_Zpolynomials'])
    Z.computeZernike(tel)
    TT_modes = Z.modesFullRes[:,:,:2].reshape((param['resolution']**2,2))

    TTM = DM_OOPAO(telescope=tel,
                           nSubap=2,
                           mechCoupling=param['mechanicalCoupling'],
                           modes=TT_modes)
    
    # -----------------------     SH   ----------------------------------
    tel.isPaired = False
    tel.resetOPD()
    ngs * tel
    ngs*tel_circ
    
    nsub = param['nSubaperture']
    grid=np.mgrid[0:nsub,0:nsub]
    rgrid=np.sqrt((grid[0]-nsub/2+0.5)**2+(grid[1]-nsub/2+0.5)**2)
    mask_sub = np.zeros((nsub,nsub)).astype(np.float32)
    mask_sub[np.where(rgrid<nsub/2)]=1

    mask_obs = np.zeros((nsub,nsub)).astype(np.float32)
    mask_obs[np.where(rgrid<2)]=1

    map_subap = mask_sub-mask_obs
    map_subap = map_subap.astype(bool)
    
    wfs = ShackHartmann_modifNoise(nSubap=param['nSubaperture'],
                                   telescope=tel_circ,
                                   lightRatio=param['lightThreshold'],
                                   valid_subapertures = map_subap,
                                   pixel_scale=param['plateScale'],
                                   shannon_sampling=param['shannon'],
                                   n_pixel_per_subaperture=param['nPixelPerSubap'])
    
    if param['detector_wfs'] == 'OCAM':
        OCAM = Det_OOPAO(nRes=wfs.cam.resolution,
                        integrationTime=tel.samplingTime,
                        bits=14,
                        FWC=270000,
                        gain=param['em_gain'],
                        sensor='EMCCD',
                        QE=0.92,
                        binning=1,
                        psf_sampling=1,
                        darkCurrent=0,
                        readoutNoise=0.3,
                        photonNoise=True)
        wfs.cam = OCAM
        detector_type = 'EMCCD'
    else: 
        detector_type = 'CCD'
        wfs.cam.photonNoise = param['photonNoise']
        wfs.cam.readoutNoise = param['ron']
        wfs.cam.QE = param['QE']
        wfs.cam.darkCurrent = param['darkCurrent']

    #  -----------------------     Science detector  ----------------------------------
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

    #  -----------------------     Modal basis  ---------------------------------- 
    
    nameFolder = param['pathInput']
    initName = f'M2C_{param["modal_basis"]}'
    nameFile = f'{initName}_telescope_res{param["resolution"]}_DM_nact_{param["nActuator"]}_keck'
    file_path = f"{nameFolder}{nameFile}.npz"
    
    loaded_M2C = False
    M2C_ = None
    # --- Try to load precomputed basis ---
    try:
        M2C_sparse = sparse.load_npz(file_path)
        M2C_ = M2C_sparse.toarray()
        loaded_M2C = True
        print(f"Loaded {param['modal_basis']} basis from {file_path}")
        
    except (FileNotFoundError, OSError):
        print(f"No {param['modal_basis']} basis found at {file_path}, computing new one...")
    
        if param['modal_basis'] =='KL':
            param['save_basis'] = True
            M2C_ = compute_KL_basis(tel, atm, DM, n_batch = 10)

        elif param['modal_basis'] == 'Zernike':
                 
            Z = Zernike(tel, param['nModes'])
            Z.computeZernike(tel)
            M2C_ = np.linalg.pinv(np.squeeze(DM.modes[tel.pupilLogical, :])) @ Z.modes
    
    # --- Truncate to number of modes & Build projector---
    if param['nModes'] is not None:
        M2C = M2C_[:, :param['nModes']] if param['modal_basis'] == 'KL' else M2C_.copy()
        print('Truncate modal basis')
    else:
        M2C = M2C_.copy()
        print('No truncate modal basis')
        
    DM.coefs = M2C
    tel * DM
    modes = tel.OPD.reshape(tel.resolution ** 2, M2C.shape[1])       
    projector = np.linalg.pinv(modes)
        
    # making sparse matrix of the modal base
    
    if param['save_basis'] and not loaded_M2C:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        M2C_sparse = sparse.csr_matrix(M2C_)
        sparse.save_npz(file_path, M2C_sparse)
        print(f"Saving {param['modal_basis']} basis as a sparse matrix at {file_path}")

    #  -----------------------     Calibrations  ----------------------------------
    stroke = param['stroke']  # amplitude of the modes in m
    M2C_zonal = np.eye(DM.nValidAct)
    M2C_TTM = np.eye(2)
    wfs.is_geometric = False
    tel.resetOPD() # to remove the default OPD of the co-phasing error

    calib_zonal = InteractionMatrix_test(ngs=ngs,
                                        tel=tel,
                                        dm=DM,
                                        wfs=wfs,
                                        M2C=M2C_zonal,
                                        stroke=stroke,
                                        nMeasurements=100,
                                        noise='off',
                                        display_slopes = False,
                                        single_pass = False)

    calib_modal = InteractionMatrix_test(ngs=ngs,
                                        tel=tel,
                                        dm=DM, 
                                        wfs=wfs, 
                                        M2C=M2C, 
                                        stroke=stroke, 
                                        nMeasurements=100, 
                                        noise='off',
                                        display_slopes = False,
                                        single_pass = False)
    # TTM calib
    calib_TTM = InteractionMatrix_test(ngs=ngs, 
                                        tel=tel, 
                                        dm=TTM, 
                                        wfs=wfs, 
                                        M2C=M2C_TTM, 
                                        stroke=stroke, 
                                        nMeasurements=1, 
                                        noise='off',
                                        single_pass = False)

    #  -----------------------     Filter modes Tip/Tilt and Piston ----------------------------------
    g1 = np.zeros((wfs.nSignal,1))
    g2 = np.zeros((wfs.nSignal,1))

    g1[:wfs.nSignal//2,:]=1
    g2[wfs.nSignal//2:,:]=1

    Gt = np.hstack([g1,g2])
    Gt = Gt/np.linalg.norm(Gt)
    reg = Gt.T @ Gt
    reg /= 10 # debug 
    PGt = Gt @ np.linalg.inv(reg) @ Gt.T
    
    bp = np.ones((DM.nValidAct,1))
    bp /=np.sqrt(DM.nValidAct)
    reg_p = bp.T @ bp
    Pp = bp @ np.linalg.inv(reg_p) @ bp.T

    if param['filter_TT'] == True: 
        print('Filtering TT from the DM reconstructor')
        Dt = (np.eye(PGt.shape[0])-PGt) @ calib_zonal.D @ (np.eye(Pp.shape[0])-Pp) #filter tt and piston
        Dt_modal = (np.eye(PGt.shape[0])-PGt) @ calib_modal.D
    else:
        calib_zonal.nTrunc = param['SVD_thr']
        Dt = calib_zonal.Dtrunc @ (np.eye(Pp.shape[0])-Pp) #filter piston

    #  -----------------------     Phase covariance  ----------------------------------        
    d = np.linalg.norm(coords_final[:,None,:] - coords_final[None,:,:], axis=2)
    Dphi = 6.88 * (d / atm.r0)**(5/3)
    Cphi = -0.5*Dphi
    Cphi -= np.mean(Cphi)
    Cphi /=1e1

    invphi = np.linalg.inv(Cphi)
            
    #  -----------------------     Keck reconstructor  ----------------------------------
    alpha = param['alpha']
    tel.isPaired = False
    tel.resetOPD()
    DM.coefs = 0
    ngs * tel * DM * wfs
    
    # computation of the weight of the subap with respect to their illumination
    int_subap = wfs.maps_intensity
    weight = np.sum(int_subap, axis=(1, 2))
    max_weight = weight.max()
    weight = weight/max_weight#**param['weight_power']
    weight[np.where(weight<1e-3)] = 0

    W0 = np.diag(weight) 
    W1 = np.diag(weight) * 0
    W = np.asarray(np.bmat([[W0, W1], [W1, W0]]))

    H = np.zeros((Dt.shape[0], Dt.shape[1]))
    H[:, :Dt.shape[1]] = Dt
    Dw = W @ H
    Mat_inter = H.T @ Dw
    eps_mat = param['eps_iMat']
    #Mat_inter += eps_mat * np.eye(Mat_inter.shape[0])
    #A = np.linalg.pinv(Mat_inter+alpha*invphi) 
    A = np.linalg.inv(Mat_inter +alpha*invphi)
    B = H.T @ W
    Rec_int1 = A @ B  

    H = np.zeros((calib_TTM.D.shape[0], calib_TTM.D.shape[1]))
    H[:, :calib_TTM.D.shape[1]] = calib_TTM.D
    Dw_ttm = W @ H
    Mat_inter_ttm = H.T @ Dw_ttm
    Mat_inter_ttm += eps_mat * np.eye(Mat_inter_ttm.shape[0])
    A = np.linalg.pinv(Mat_inter_ttm)
    B = H.T @ W
    Rec_int2 = A @ B
    
    Rec = np.vstack([Rec_int1,Rec_int2])
    
    #Rec modal
    Rec_modal_1 = np.linalg.inv(Dt_modal.T @ W @ Dt_modal) @ Dt_modal.T @ W
    Rec_modal_2 = M2C @ Rec_modal_1
    Rec_modal = np.vstack([Rec_modal_2, Rec_int2])
    
    print('Keck rec build!')   
 
    #  -----------------------     M1 aberrations, NCPA & Jitter  ----------------------------------    
    # Add co-phasing error
    Nseg = param['numberSegments']
    KSM.set_segment_actuators(np.arange(Nseg), 1,0,0)
    flat_ksm = KSM.opd.shaped

    if param['M1_segments_pistons']==True and param['M1_segments_tilts']==False:
        print(param['M1_segments_pistons'])
        for i in range(Nseg):
            aa = np.random.rand(3) # (piston in m, tip in rad, tilt in rad)
            KSM.set_segment_actuators(i, aa[0], aa[1]*0, aa[2]*0)
        
        opd_phasing = KSM.opd.shaped # OPD
        surf = KSM.surface.shaped # Surface of the DM it is the OPD divided by 2
        cophase_amp = np.std(opd_phasing[np.where(flat_ksm != 0)])

        seg_amp = param['M1_OPD_amplitude']
        default_shape = opd_phasing/cophase_amp * seg_amp

        opd_M1 = OPD_map(telescope=tel)
        opd_M1.OPD = default_shape * tel.pupil
        print('M1 piston!')

    elif param['M1_segments_tilts']==True and param['M1_segments_pistons']==True:
        print(param['M1_segments_pistons'])
        for i in range(Nseg):
            aa = np.random.rand(3) # (piston in m, tip in rad, tilt in rad)
            KSM.set_segment_actuators(i, aa[0], aa[1], aa[2])
        
        opd_phasing = KSM.opd.shaped # OPD
        surf = KSM.surface.shaped # Surface of the DM it is the OPD divided by 2
        cophase_amp = np.std(opd_phasing[np.where(flat_ksm != 0)])

        seg_amp = param['M1_OPD_amplitude']
        default_shape = opd_phasing/cophase_amp * seg_amp

        opd_M1 = OPD_map(telescope=tel)
        opd_M1.OPD = default_shape * tel.pupil
        print('M1 piston and TT!')
        
    else:
        print('No segments tilts or pistons')
        print(param['M1_segments_pistons'])
        KSM.set_segment_actuators(np.arange(Nseg), 1,0,0)
        default_shape = KSM.opd.shaped * 0
        print('M1 flat!')

    default_shape = (default_shape - np.mean(default_shape[np.where(tel.pupil>0)])) * tel.pupil
    opd_M1 = OPD_map(telescope=tel)
    opd_M1.OPD = default_shape 

    if param['warping_M1']:
        nmin = 2
        nmax = 4
        nlist = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4]
        nlist = np.array(nlist)
        f1 = nlist[nlist>=nmin].astype(float)
        psd = f1 ** (-2)
        z_indexmin = np.where(nlist == nmin)[0]
        z_indexmax = np.where(nlist == nmax)[0]
        seg_vect2D_copy = seg_vect2D.copy()
        
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

        amp_warp = param['warped_ampplitude'] 
        warpped_surf = np.sum(seg_vect2D_copy,axis=0)
        amp_ = np.std(warpped_surf[np.where(flat != 0)])
        warpped_surf = warpped_surf/amp_ * amp_warp
        
        default_shape = (default_shape + warpped_surf)* tel.pupil
        opd_M1.OPD = default_shape   
    
    
    # Add NCPA
    if param['NCPA']:
        print('Computing NCPA OPD')
        
        amp_ncpa = param['NCPA_amplitude']
        nmin = param['NCPA_nmin']
        nmax = param['NCPA_nmax']

        nlist = []
        for i in range(Z.modes.shape[1]):
            ind = Z.zernIndex(i+1)
            nlist.append(ind[0])

        nlist = np.array(nlist)
        f1 = nlist[nlist>=nmin].astype(float)
        f2 = f1[f1<=nmax]
        psd = f2 ** (-2)
        zk_coefficients = psd*np.random.randn(f2.shape[0])

        z_indexmin = np.where(nlist == nmin)[0]
        z_indexmax = np.where(nlist == nmax)[0]

        Zvect = Z.modesFullRes[:,:,z_indexmin[0]:z_indexmax[-1]+1]
        Z_comb = Zvect.reshape(param['resolution']**2, Zvect.shape[2])
        phase_screen = Z_comb * zk_coefficients
        phase_screen = np.sum(phase_screen, axis=1)
        phase_screen = phase_screen.reshape(param['resolution'], param['resolution'])
        phase_screen = (phase_screen-np.mean(phase_screen[np.where(tel.pupil>0)]))*tel.pupil
        opd_screen = phase_screen * ngs.wavelength/(2*np.pi)

        amp_screen = np.std(opd_screen[np.where(tel.pupil > 0)])
        ncpa = opd_screen/amp_screen * amp_ncpa

    else:
        ncpa = np.zeros((param['resolution'], param['resolution']))

    #creating the COG offset for the NCPA map
    opd_ncpa = OPD_map(telescope=tel)
    opd_ncpa.OPD = ncpa
    tel.isPaired = True # DO NOT CHANGE THIS
    tel.resetOPD()
    TTM.coefs = 0
    DM.coefs = 0
    
    science * tel * opd_ncpa * TTM * DM * wfs 
    zeros_slopes = wfs.reference_slopes_maps
    ncpa_slopes = wfs.signal_2D
    ncpa_cog = ncpa_slopes* wfs.slopes_units+zeros_slopes

    # Pure NCPA
    amp_pure_ncpa = param['Pure_NCPA_amplitude']
    nmin = param['Pure_NCPA_nmin']
    nmax = param['Pure_NCPA_nmax']

    nlist = []
    for i in range(Z.modes.shape[1]):
        ind = Z.zernIndex(i+1)
        nlist.append(ind[0])

    nlist = np.array(nlist)
    f1 = nlist[nlist>=nmin].astype(float)
    f2 = f1[f1<=nmax]
    psd = f2 ** (-2)
    zk_coefficients = psd*np.random.randn(f2.shape[0])

    z_indexmin = np.where(nlist == nmin)[0]
    z_indexmax = np.where(nlist == nmax)[0]

    Zvect = Z.modesFullRes[:,:,z_indexmin[0]:z_indexmax[-1]+1]
    Z_comb = Zvect.reshape(param['resolution']**2, Zvect.shape[2])
    phase_screen = Z_comb * zk_coefficients
    phase_screen = np.sum(phase_screen, axis=1)
    phase_screen = phase_screen.reshape(param['resolution'], param['resolution'])
    opd_screen = phase_screen * ngs.wavelength/(2*np.pi)

    amp_screen = np.std(opd_screen[np.where(tel.pupil > 0)])
    offset = opd_screen/amp_screen * amp_pure_ncpa
    
    opd_offset = OPD_map(telescope=tel)
    opd_offset.OPD = offset
    
    offset_seg_coefs = np.dot(opd_offset.OPD.flatten(),proj_seg)
    offset_seg = np.dot(offset_seg_coefs,segments_vect1D).reshape(param['resolution'],param['resolution'])
    
    # PSF
    src_cam.integrationTime = tel.samplingTime
    tel.isPaired = False
    tel.resetOPD()
    science * tel * src_cam
    PSF_diff = src_cam.frame[:]

    tel.isPaired = False
    tel.resetOPD()
    science * tel* opd_ncpa* src_cam
    PSF_ncpa = src_cam.frame[:]

    src_cam.integrationTime = param['science_integrationTime']

    # Jitter
    opd_jitter = OPD_map(telescope=tel) 
    frames = param['Jitter_nFrames']
    dt = 1.0 / frames
    fc = param['Jitter_freq']   # 50 Hz
    
    #need to think about mas to nm conversion
    mas_to_rad = 1e-3*(np.pi/(180*3600))
    amp_rad = param['Jitter_amp'] * mas_to_rad
    amp_m = amp_rad * ngs.wavelength / (2*np.pi)
    
    a = np.exp(-2 * np.pi * fc * dt)
    b = param['Jitter_amp'] * np.sqrt(1 - a**2)
    jitter = [a,b] # x(k+1)= ax(k)+bn(k) 
   
    #  -----------------------     ZWFS  ----------------------------------

    pupil_stack = np.stack((tel.pupil, tel.pupil))
    cyl, cxl = center_of_mass(np.array(tel.pupil)) # check I think it's cy,cx
    cyr = cyl
    cxr = cxl + tel.pupil.shape[0]//2 

    img_size_x = tel.pupil.shape[0]*2
    img_size_y = tel.pupil.shape[0]
    coord = [int(cxl),int(cyl),int(cxr),int(cyr), img_size_x, img_size_y]
    dimple_pixels = param['ZWFS_dimple_pixels']
    wavelength = param['ZWFS_wavelength'] # nm # H band OOPAO
    diameter = param['ZWFS_diameter']
    depth = param['ZWFS_depth']
    ZWFS = zernikeWFS(pupil_stack,coord,diameter,depth,dimple_pixels,wavelength)

    ZWFS.nIterRec = param['ZWFS_nIterRec']
    ZWFS.doUnwrap = param['ZWFS_doUnwrap']
    if param['ZWFS_algo'] is not None:
        ZWFS.algo=param['ZWFS_algo']
        
    ZWFS.pupilRec = param['ZWFS_pupil_both']

    # seg IDs
    matching_inds = {str(i+1): i for i in range(0, param['numberSegments'])}
    plt.show(block=False)
    plt.close('all')
    print('Everything has been loaded and the calibration are done!')

    class emptyClass():
        pass

    # save output as sub-classes
    simulationObject = emptyClass()

    simulationObject.ksm = KSM
    simulationObject.seg2D = seg_vect2D
    simulationObject.seg1D = segments_vect1D
    simulationObject.proj_seg = proj_seg

    simulationObject.tel    = tel
    simulationObject.pup_crop    = pup_crop
    simulationObject.atm    = atm

    simulationObject.ngs    = ngs
    simulationObject.dm     = DM
    simulationObject.ttm    = TTM
    simulationObject.wfs    = wfs
    simulationObject.detector_type = detector_type

    simulationObject.param  = param
    simulationObject.keck_reconstructor = Rec
    simulationObject.keck_reconstructor_modal = Rec_modal
    simulationObject.calib_zonal  = calib_zonal
    simulationObject.calib_modal  = calib_modal
    simulationObject.calib_TTM = calib_TTM
    simulationObject.projector_modal = projector
    simulationObject.name_basis = param['modal_basis']
    simulationObject.basis = modes
    simulationObject.M2C   = M2C
    simulationObject.M2C_TTM  = M2C_TTM
    simulationObject.projector_dm = projector_DM
    simulationObject.IF = dm_modes
    simulationObject.filter_TTM = PGt
    simulationObject.w = W
    simulationObject.invphi = invphi
    
    simulationObject.science = science
    simulationObject.science_detector = src_cam

    simulationObject.opd_M1 = opd_M1
    simulationObject.opd_ncpa = opd_ncpa
    simulationObject.opd_jitter = opd_jitter
    simulationObject.opd_offset = opd_offset
    simulationObject.offset_seg_proj = offset_seg

    simulationObject.PSF_diff = PSF_diff
    simulationObject.PSF_ncpa = PSF_ncpa
    simulationObject.jitter = jitter
    
    simulationObject.cog_ncpa = ncpa_cog
    simulationObject.cog_zeros = zeros_slopes
    
    simulationObject.zwfs = ZWFS
    simulationObject.matching_inds = matching_inds

    return simulationObject
