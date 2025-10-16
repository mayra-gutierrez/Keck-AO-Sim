import numpy as np
import time
import matplotlib.pyplot as plt

from OOPAO.calibration.getFittingError import *

def close_loop(AO_sys, atm_seed=0):
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
        plt.show(block=False)
        plt.close('all')
        
        if param['filter_TT'] == True:
            Rec1 = AO_sys.filter_TTM @ Rec1      

        Rec2 = AO_sys.M2C_TTM @ AO_sys.calib_TTM.M
        Rec = np.vstack([Rec1, Rec2])
        
        modes_proj = AO_sys.dm.modes
        proj = AO_sys.projector_dm

    else:
        # for modal control 
        # need to handle the filtering of the TTM not implemented yet
        M2C_CL = AO_sys.M2C
        calib_CL = AO_sys.calib_modal
        calib_CL.nTrunc = param['modal_trunc']
        Rec1 = M2C_CL@calib_CL.Mtrunc
        plt.show(block=False)
        plt.close('all')
        
        if param['filter_TT'] == True:
            Rec1 = AO_sys.filter_TTM @ Rec1

        Rec2 = AO_sys.M2C_TTM @ AO_sys.calib_TTM.M
        Rec = np.vstack([Rec1, Rec2])
        
        modes_proj = AO_sys.basis
        proj = AO_sys.projector_modal

    TT_modes = AO_sys.ttm.modes
    ratio = int((1/AO_sys.tel.samplingTime)/(param['Jitter_freq']))

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
        if param['print_display']:
            print('Loop' + str(i) + '/' + str(nLoop) + ' Turbulence: ' + str(total[i]) + ' -- Residual:' +
                str(residual[i]) + ' -- Fitting:' + str(fitting_rms[i]) + '\n')

    # plot the residual and the SR in the science band
    if param['print_display']:
        print('Average SR = %d\n', np.mean(SR[50:]*100))

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
 
    output = {'PSF_LE': PSF_LE, 'SR': SR, 
              'residual': residual}

    return output
