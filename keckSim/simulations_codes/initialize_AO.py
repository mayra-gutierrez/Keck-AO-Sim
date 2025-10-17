import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pfits
from scipy import sparse
import scipy.sparse
import pickle

from hcipy import *
from scipy.ndimage import center_of_mass

from OOPAO.DeformableMirror import DeformableMirror as DM_OOPAO
from OOPAO.MisRegistration import MisRegistration
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
#os.chdir("/home/mcisse/keckAOSim/keckSim")

# Add the parent directory (keckSim) to Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from keckTel import keckTel  # now Python will find this module
from keckAtm import keckAtm

from simulations_codes.Jitter_TT import RandomTipTilt_Gaussian, SinusoidalTipTilt
from simulations_codes.SH import ShackHartmann_modifNoise
from simulations_codes.Imat_SHWFS import InteractionMatrix_test

from simulations_codes.ZWFS_toolbox.wfSensors import *

def initialize_AO_hardware(param):

    # -----------------------  Keck apperture ----------------------------------
    keck_object = keckTel.create('keck', resolution=param['resolution'],samplingTime=param['samplingTime'], return_segments=True)

    # -----------------------      KSM       ----------------------------------
    KSM = SegmentedDeformableMirror(keck_object.keck_segments)
    s1 = np.array(KSM.segments)
    seg_vect2D = s1.reshape(s1.shape[0],int(np.sqrt(s1.shape[1])),int(np.sqrt(s1.shape[1])))
    segments_vect1D = seg_vect2D.reshape(seg_vect2D.shape[0],seg_vect2D.shape[1]**2)
    proj_seg = np.linalg.pinv(segments_vect1D)

    # -----------------------     TELESCOPE   ----------------------------------
    tel = keck_object.tel
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
    # -----------------------     DEFORMABLE MIRRORs   ----------------------------------
    nAct = param['nActuator']
    pitch = param['diameter']/(param['nActuator']-1)
    # mis-registrations object
    misReg = MisRegistration(param)
    
    dm = DM_OOPAO(telescope=tel,
                          nSubap=nAct-1,
                          mechCoupling=param['mechanicalCoupling'],
                          misReg=misReg,
                          coordinates=param['dm_coordinates'],
                          pitch=pitch,
                          floating_precision=32)
    
    pup = tel.pupil.flatten()
    IF = dm.modes.astype(np.float32) * pup[:, np.newaxis]
    IF_sum = np.sum(IF, axis=0)
    IF_max = IF_sum.max()
    valid_index = np.where(IF_sum >= param['validActTreshold']*IF_max)[0]
    IF_filtered = IF[:, valid_index]
    dm_modes = IF_filtered
    
    DM = DM_OOPAO(telescope=tel,
                          nSubap=nAct-1,
                          mechCoupling=param['mechanicalCoupling'],
                          misReg=misReg,
                          coordinates=param['dm_coordinates'],
                          pitch=pitch, 
                          modes=IF_filtered)
    dm_modes = DM.modes
    projector_DM = np.linalg.pinv(dm_modes)
    #  -----------------------     TTM basis  ----------------------------------
    Z = Zernike(tel, param['nb_Zpolynomials'])
    Z.computeZernike(tel)
    TT_modes = Z.modesFullRes[:,:,:2].reshape(param['resolution']**2, Z.modesFullRes[:,:,:2].shape[-1])

    TTM = DM_OOPAO(telescope=tel,
                           nSubap=2,
                           mechCoupling=param['mechanicalCoupling'],
                           modes=TT_modes)
    
    #  -----------------------     Filter modes  ----------------------------------
    # TT
    Reg_TTM_DM = np.zeros((2, DM.modes.shape[1]))
    eps = param['regularization_TT']
    for k in range(2):
        zi = Z.modesFullRes[:,:,k]
        dm_command = projector_DM @ zi.flatten()
        Reg_TTM_DM[k,:] = dm_command

    Reg_TTM = Reg_TTM_DM @ Reg_TTM_DM.T
    P_TT = Reg_TTM_DM.T @ np.linalg.pinv(Reg_TTM + eps * np.eye(Reg_TTM.shape[0])) @ Reg_TTM_DM
    I = np.eye(P_TT.shape[0])
    P_orth_TTM = I - P_TT  # Projects orthogonally to T/T

    #Segments
    Reg_M1_DM = np.zeros((seg_vect2D.shape[0], DM.modes.shape[1]))

    for k in range(seg_vect2D.shape[0]):
        si = seg_vect2D[k,:,:]
        dm_command = projector_DM @ si.flatten()
        Reg_M1_DM[k,:] = dm_command

    Reg = Reg_M1_DM @ Reg_M1_DM.T
    eps = param['regularization_seg']
    P_seg = Reg_M1_DM.T @ np.linalg.pinv(Reg + eps*np.eye(Reg.shape[0])) @ Reg_M1_DM
    I = np.eye(P_seg.shape[0])
    P_orth_seg = I - P_seg  

    # -----------------------     SH   ----------------------------------
    tel.isPaired = False
    tel.resetOPD()
    ngs * tel
    wfs = ShackHartmann_modifNoise(nSubap=param['nSubaperture'],
                                   telescope=tel,
                                   lightRatio=param['lightThreshold'],
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
    nameFile = f'{initName}_telescope_res{param["resolution"]}_DM_nact_{param["nActuator"]}'
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
        M2C_sparse = sparse.csr_matrix(M2C_)
        sparse.save_npz(file_path, M2C_sparse)
        print(f"Saving {param['modal_basis']} basis as a sparse matrix at {file_path}")

    #  -----------------------     Calibrations  ----------------------------------
    stroke = param['stroke']  # amplitude of the modes in m
    M2C_zonal = np.eye(DM.nValidAct)
    M2C_TTM = np.eye(2)
    wfs.is_geometric = False
    tel.resetOPD() # to remove the default OPD of the co-phasing error

    calib_zonal = InteractionMatrix_test(ngs=ngs, \
                                        tel=tel, \
                                        dm=DM, \
                                        wfs=wfs, \
                                        M2C=M2C_zonal, \
                                        stroke=stroke, \
                                        nMeasurements=100, \
                                        noise='off')
    
    #calib_modal = CalibrationVault(calib_zonal.D @ M2C)
    calib_modal = InteractionMatrix_test(ngs=ngs, \
                                        tel=tel, \
                                        dm=DM, \
                                        wfs=wfs, \
                                        M2C=M2C, \
                                        stroke=stroke, \
                                        nMeasurements=100, \
                                        noise='off')
    # TTM calib
    calib_TTM = InteractionMatrix_test(ngs=ngs, \
                                        tel=tel, \
                                        dm=TTM, \
                                        wfs=wfs, \
                                        M2C=M2C_TTM, \
                                        stroke=stroke, \
                                        nMeasurements=1, \
                                        noise='off')

    #  -----------------------     Keck reconstructor  ----------------------------------
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
    
    ww = np.diag(W)
    inv_diag = np.where(ww != 0, 1 / ww, 0)
    inv_W = np.diag(inv_diag)

    calib_zonal.nTrunc = param['SVD_thr']
    Dtrunc_ =  calib_zonal.Dtrunc 
    #calib_modal.nTrunc = param['SVD_thr']
    #Dtrunc_ =  calib_modal.Dtrunc 
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

    if param['filter_segments'] == True and param['filter_TT'] == True:
        DM_rec = P_orth_TTM @ P_orth_seg @ Rec_int[:DM.nValidAct,:]
        TTM_rec = Rec_int[DM.nValidAct:,:]
        Rec = np.vstack([DM_rec,TTM_rec])
    elif param['filter_segments'] == False and param['filter_TT'] == True:
        DM_rec = P_orth_TTM @ Rec_int[:DM.nValidAct,:]
        TTM_rec = Rec_int[DM.nValidAct:,:]
        Rec = np.vstack([DM_rec,TTM_rec])
    else:
        Rec = Rec_int
    print('Keck rec build!')

    #  -----------------------     M1 aberrations, NCPA & Jitter  ----------------------------------    
    # Add co-phasing error
    Nseg = param['numberSegments']
    KSM.set_segment_actuators(np.arange(Nseg), 1,0,0)
    flat_ksm = KSM.opd.shaped

    if param['M1_segments_pistons']==True and param['M1_segments_tilts']==False:
        print(param['M1_segments_pistons'])

        with open('/home/lab/maygut/keckAOSim/keckSim/simulations_codes/OPD_120nm_file.pkl', 'rb') as f:
            opd_from_file = pickle.load(f)

        print(f"Loaded OPD file with shape: {np.shape(opd_from_file)}")
        if hasattr(opd_from_file, 'OPD'):
            opd_from_file = opd_from_file.OPD  # extract the actual array

        '''
        for i in range(Nseg):
            aa = np.random.rand(3) # (piston in m, tip in rad, tilt in rad)
            KSM.set_segment_actuators(i, aa[0], aa[1]*0, aa[2]*0)
        
        opd_phasing = KSM.opd.shaped # OPD
        surf = KSM.surface.shaped # Surface of the DM it is the OPD divided by 2
        cophase_amp = np.std(opd_phasing[np.where(flat_ksm != 0)])
        '''
        #seg_amp = param['M1_OPD_amplitude']
        #default_shape = opd_phasing/cophase_amp * seg_amp

        opd_from_file = opd_from_file * tel.pupil

        opd_M1 = OPD_map(telescope=tel)
        opd_M1.OPD = opd_from_file #default_shape * tel.pupil
        default_shape = opd_from_file.copy()
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
    
    #try to under stand this section.
    #default_shape = (default_shape - np.mean(default_shape[np.where(tel.pupil>0)])) * tel.pupil
    opd_M1 = OPD_map(telescope=tel)
    opd_M1.OPD = default_shape 
    
    
    # Add NCPA
    if param['NCPA'] == True:
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
        opd_screen = phase_screen * ngs.wavelength/(2*np.pi)

        amp_screen = np.std(opd_screen[np.where(tel.pupil > 0)])
        ncpa = opd_screen/amp_screen * amp_ncpa

    else:
        ncpa = np.zeros((param['resolution'], param['resolution']))

    src_cam.integrationTime = tel.samplingTime
    opd_ncpa = OPD_map(telescope=tel)
    opd_ncpa.OPD = ncpa

    # PSF
    src_cam.integrationTime = tel.samplingTime
    tel.isPaired = False
    tel.resetOPD()
    science * tel * src_cam
    PSF_diff = src_cam.frame

    tel.isPaired = False
    tel.resetOPD()
    science * tel* opd_ncpa* src_cam
    PSF_ncpa = src_cam.frame

    src_cam.integrationTime = param['science_integrationTime']

    if param['Jitter'] == True:
        dt = 1/ param['Jitter_nFrames']
        jitter = SinusoidalTipTilt(freq=param['Jitter_freq'], dt=dt, n_frames=param['Jitter_nFrames'], nm_rms=param['Jitter_amp'])#RandomTipTilt_Gaussian(freq=param['Jitter_freq'], dt=dt, n_frames=param['Jitter_nFrames'], nm_rms=param['Jitter_amp'])
    else :
        dt = 1/ param['Jitter_nFrames']
        jitter = SinusoidalTipTilt(freq=param['Jitter_freq'], dt=dt, n_frames=param['Jitter_nFrames'], nm_rms=param['Jitter_amp'])#RandomTipTilt_Gaussian(freq=param['Jitter_freq'], dt=dt, n_frames=param['Jitter_nFrames'], nm_rms=param['Jitter_amp'])
        jitter.x = jitter.x * 0
        jitter.y = jitter.y * 0

    opd_jitter = OPD_map(telescope=tel)
    
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
    ZWFS.algo=param['ZWFS_algo']
    ZWFS.pupilRec = param['ZWFS_pupil_both']

    # seg IDs
    matching_inds = {str(i+1): i for i in range(0, param['numberSegments'])}
    plt.show(block=False)
    
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
    simulationObject.atm    = atm

    simulationObject.ngs    = ngs
    simulationObject.dm     = DM
    simulationObject.ttm    = TTM
    simulationObject.wfs    = wfs
    simulationObject.detector_type = detector_type

    simulationObject.param  = param
    simulationObject.keck_reconstructor = Rec
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
    simulationObject.filter_TTM = P_orth_TTM
    simulationObject.filter_Piston = P_orth_seg
    
    simulationObject.science = science
    simulationObject.science_detector = src_cam

    simulationObject.opd_M1 = opd_M1
    simulationObject.opd_ncpa = opd_ncpa
    simulationObject.opd_jitter = opd_jitter

    simulationObject.PSF_diff = PSF_diff
    simulationObject.PSF_ncpa = PSF_ncpa
    simulationObject.jitter = jitter
    
    simulationObject.zwfs = ZWFS
    simulationObject.matching_inds = matching_inds

    return simulationObject
