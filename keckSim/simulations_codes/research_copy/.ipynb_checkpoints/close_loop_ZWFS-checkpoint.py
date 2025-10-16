import numpy as np
import time
import matplotlib.pyplot as plt
import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim/")
sys.path.insert(0, os.getcwd())
from simulations_codes.ZWFS_toolbox.ZWFS_tools import *
from simulations_codes.ZWFS_toolbox.tools import *
from simulations_codes.ZWFS_toolbox.wfSensors import *

from OOPAO.calibration.getFittingError import *

def close_loop(AO_sys,ZWFS_param, atm_seed=0):
    param = AO_sys.param
    type_rec = param['type_rec'] 
    display = param['display_loop']
    
    AO_sys.wfs.is_geometric = False
    AO_sys.tel.isPaired = False
    AO_sys.tel.resetOPD()
    AO_sys.dm.coefs = 0
    AO_sys.ttm.coefs = 0
    AO_sys.ngs.magnitude = param['magnitude_guide']
    AO_sys.science.magnitude = param['science_magnitude']

    # Atmosphere propagation
    AO_sys.atm.generateNewPhaseScreen(seed=atm_seed)
    AO_sys.tel+AO_sys.atm
    AO_sys.ngs * AO_sys.tel * AO_sys.opd_M1
    AO_sys.tel * AO_sys.ttm * AO_sys.dm * AO_sys.wfs

    if type_rec == 'keck':
        M2C_CL = np.eye(AO_sys.dm.nValidAct)
        Rec = AO_sys.keck_reconstructor
        modes_proj = AO_sys.dm.modes
        proj = AO_sys.projector_dm
    elif type == 'zonal':
        # for zonal control
        # need to handle the filtering of the TTM not implemented yet
        calib_CL = AO_sys.calib_zonal
        M2C_CL = np.eye(AO_sys.dm.nValidAct)
        calib_CL.nTrunc = 0
        Rec1 = M2C_CL@calib_CL.Mtrunc
        if param['filter_TT'] == True:
            Rec1 = AO_sys.filter_TTM @ calib_CL.Mtrunc
        
        Rec2 = AO_sys.M2C_TTM @ AO_sys.calib_TTM.M
        Rec = np.vstack([Rec1, Rec2])

        modes_proj = AO_sys.dm.modes
        proj = AO_sys.projector_dm
    else:
        # for modal control 
        M2C_CL = AO_sys.M2C[:,2:] # remove TT
        calib_CL = AO_sys.calib_modal
        calib_CL.nTrunc = param['modal_trunc']
        plt.show(block=False)
        plt.close('all')
        Rec1 = M2C_CL@calib_CL.Mtrunc
        if param['filter_TT'] == True:
            Rec1 = AO_sys.filter_TTM @ calib_CL.Mtrunc

        Rec2 = AO_sys.M2C_TTM @ AO_sys.calib_TTM.M
        Rec = np.vstack([Rec1, Rec2])
        modes_proj = AO_sys.basis
        proj = AO_sys.projector_modal


    TT_modes = AO_sys.ttm.modes
    ratio = int((1/AO_sys.tel.samplingTime)/(param['Jitter_freq']))
    
    # ZWFS param 
    ZWFS_active = ZWFS_param['activate']
    ZWFS_iter_max = ZWFS_param['max_iter']
    gain_Z = ZWFS_param['gain']
    nrep_max = ZWFS_param['mean_img']
    freq_zwfs = ZWFS_param['freq']
    maxZ = ZWFS_param['maxZ']
    subGZ = ZWFS_param['subGZ']
    maxGlobalZ = ZWFS_param['maxGlobalZ']
    #expo = ZWFS_param['total_exposure']
    
    nrep = 0
    ZWFS_iter = 0
    ImgBuffer_ZWFS = []
    samp_zwfs = (1/freq_zwfs)
    speed_factor = int(samp_zwfs/AO_sys.tel.samplingTime)#int((1/AO_sys.tel.samplingTime)/(freq_zwfs))   #
    #Nimgs = int(freq_zwfs * expo)
    coefs_seg_list = []
    
    m1_amp = []
    m1_amp.append(float(np.std(AO_sys.opd_M1.OPD[np.where(AO_sys.tel.pupil > 0)]) * 1e9))
    
    # KSM
    seg_vect2D = AO_sys.seg2D
    segments = AO_sys.seg1D
    proj_seg = AO_sys.proj_seg
    Nseg = param['numberSegments']
    pupil_spider = AO_sys.tel.pupil
    z2p, p2z = zernikeBasis_nonCirc(maxGlobalZ, pupil_spider)
    
    if ZWFS_active:
        print(f'ZWFS parameters nIterRec: {AO_sys.zwfs.nIterRec},doUnwrap:{AO_sys.zwfs.doUnwrap}, algo:{AO_sys.zwfs.algo}, pupil: {AO_sys.zwfs.pupilRec}')
    
    # loop parameters
    gainCL = param['gainCL']
    gainTTM = gainCL
    latency = param['latency']
    nLoop = param['nLoop']

    AO_sys.wfs.cam.photonNoise = True # True to add photon noise only, to add RON add the line wfs.cam.readoutNoise = 0.5 for example
    if AO_sys.wfs.cam.photonNoise == True:
        AO_sys.wfs.cam.readoutNoise = param['ron']
        AO_sys.wfs.cam.darkCurrent = param['darkCurrent']

    # allocate memory to save data
    SR = np.zeros(nLoop)
    total = np.zeros(nLoop)
    residual = np.zeros(nLoop)
    residual_OPD = []
    fitting_rms = np.zeros(nLoop)
    signalBuffer = np.zeros((AO_sys.wfs.signal.shape[0], nLoop)) # buffer for the frame delay
    SRC_PSF = []
    flux_psf = np.zeros(nLoop)

    for i in range(nLoop):
        a = time.time()
        AO_sys.atm.update()

        # Add Jitter
        if param['Jitter']:
            if i%ratio == 0 or i==0:
                AO_sys.opd_jitter.OPD = (AO_sys.jitter.x[i+1] * TT_modes[:,0].reshape(param['resolution'],param['resolution'])  + AO_sys.jitter.y[i+1] * TT_modes[:,1].reshape(param['resolution'],param['resolution']))*1e-9
        else: 
            AO_sys.opd_jitter.OPD = np.zeros((param['resolution'],param['resolution']))
        
        # Add M1 phase, NCPA and jitter
        AO_sys.ngs * AO_sys.tel * AO_sys.opd_M1 * AO_sys.opd_jitter 
        total[i] = np.std(AO_sys.tel.OPD[np.where(AO_sys.tel.pupil > 0)]) * 1e9

        AO_sys.tel * AO_sys.ttm * AO_sys.dm * AO_sys.wfs

        # fitting
        OPD_fitting_2D, OPD_corr_2D, OPD_turb_2D = getFittingError(AO_sys.tel.OPD, proj, modes_proj, display=False)
        fitting_rms[i] = np.std(OPD_fitting_2D[np.where(AO_sys.tel.pupil > 0)]) * 1e9

        AO_sys.science * AO_sys.tel * AO_sys.opd_ncpa * AO_sys.science_detector 
        ImgBuffer_ZWFS.append(AO_sys.zwfs.getImageSimu(AO_sys.tel.src.phase))

    
        if ZWFS_active:
            if i > 99: # wait after bootstrap
            
                if i%speed_factor == 0 and ZWFS_iter <ZWFS_iter_max:
                    # one measurement at XHz
                    imgs = np.array(ImgBuffer_ZWFS)[i-speed_factor:i,:,:]
                    
                    img_mean =  np.mean(imgs, axis = 0)
                    
                    _, opd_wttf, seg_Zmode_coeffs = ReconPhase_Segments(img_mean,z2p, p2z,pupil_spider,
                                            AO_sys.matching_inds,seg_vect2D,
                                            AO_sys.zwfs,maxZ=maxZ,maxSeg=Nseg, subGZ=subGZ)
                
                    coef_ucsc = np.squeeze(seg_Zmode_coeffs)*1e-9
                    coefs_seg_list.append(coef_ucsc)
                    #print(f'Piston estimation {nrep}')
                    nrep += 1

                if nrep == nrep_max: # apply correction
                    coefs_seg_list = np.array(coefs_seg_list)
                    mean_seg_coef = np.mean(coefs_seg_list, axis = 0)
                    seg_rec = np.dot(mean_seg_coef,segments).reshape(param['resolution'],param['resolution'])
                    seg_rec = seg_rec * AO_sys.tel.pupil
                    seg_rec = seg_rec - np.mean(seg_rec[np.where(AO_sys.tel.pupil>0)])
                
                    AO_sys.opd_M1.OPD = AO_sys.opd_M1.OPD - gain_Z * seg_rec
                    m1_amp.append(float(np.std(AO_sys.opd_M1.OPD[np.where(AO_sys.tel.pupil > 0)]) * 1e9))

                    if display:
                        plt.figure(), plt.imshow(seg_rec * AO_sys.tel.pupil), plt.colorbar(), plt.title('M1 rec'), plt.show(block=False)
                        plt.figure(), plt.imshow(AO_sys.opd_M1.OPD * AO_sys.tel.pupil), plt.colorbar(), plt.title('M1 new shape'), plt.show(block=False)
                    if param['print_display']:
                        print('Correction ZWFS applied Loop' + str(i) + 
                            '/' + str(nLoop) + ' M1 oppd: ' + str(m1_amp[-1]) + 'nm RMS Ziter:' + str(ZWFS_iter)+ '\n')
                    ZWFS_iter +=1
                    nrep = 0
                    coefs_seg_list = []

        if i >= latency:
            command = np.matmul(Rec, signalBuffer[:, i-latency])
            AO_sys.dm.coefs = AO_sys.dm.coefs - gainCL * command[:AO_sys.dm.nValidAct]
            AO_sys.ttm.coefs = AO_sys.ttm.coefs - gainTTM * command[AO_sys.dm.nValidAct:]

        signalBuffer[:, i] = AO_sys.wfs.signal

        SR[i] = np.exp(-np.var(AO_sys.tel.src.phase[np.where(AO_sys.tel.pupil == 1)]))
        residual[i] = np.std(AO_sys.tel.OPD[np.where(AO_sys.tel.pupil > 0)]) * 1e9

        if i%10 == 0:
            res = AO_sys.tel.OPD
            res = (res - np.mean(res[np.where(AO_sys.tel.pupil>0)]))*AO_sys.tel.pupil
            if display:
                plt.figure(), plt.imshow(res), plt.colorbar(), plt.title('Residual OPD'), plt.show()
                time.sleep(1)

            residual_OPD.append(res)

        if i>50:
            psf = AO_sys.science_detector.frame[:]
            SRC_PSF.append(psf)
            flux_psf[i]=np.sum(psf)
        if param['print_display']:
            print('Loop' + str(i) + '/' + str(nLoop) + ' Turbulence: ' + str(total[i]) + ' -- Residual:' +
                str(residual[i]) + ' -- Fitting:' + str(fitting_rms[i]) + '\n')

    # plot the residual and the SR in the science band
    #print('Average SR = %d\n', np.mean(SR[50:]*100))

    PSF_LE = np.mean(SRC_PSF,axis=0)
    PSF_LE_norm =PSF_LE/ np.max(PSF_LE)  # Normalize PSF
    #log_PSF = np.log10(np.abs(PSF_LE))
    beg = AO_sys.science_detector.resolution//2 - 100
    end = AO_sys.science_detector.resolution//2 + 100

    if display:
        # plot the residual and the SR in the science band
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

        plt.figure(), plt.imshow(PSF_LE_norm[beg:end,beg:end]**0.2,cmap = 'gray'), plt.colorbar(), plt.show()
    
    if ZWFS_active:
        output = {'PSF_LE': PSF_LE, 'SR': SR, 
              'residual': residual, 'M1_OPD':m1_amp,'flux':flux_psf}
        
    else:
        output = {'PSF_LE': PSF_LE, 'SR': SR, 
              'residual': residual,'flux':flux_psf}      

    return output
