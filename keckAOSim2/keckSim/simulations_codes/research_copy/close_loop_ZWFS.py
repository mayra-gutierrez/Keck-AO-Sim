import numpy as np
import time
import matplotlib.pyplot as plt
import random
import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim/")
sys.path.insert(0, os.getcwd())
from simulations_codes.ZWFS_toolbox.ZWFS_tools import *
from simulations_codes.ZWFS_toolbox.tools import *
from simulations_codes.ZWFS_toolbox.wfSensors import *

from OOPAO.calibration.getFittingError import *
from OOPAO.tools.displayTools import cl_plot, displayMap
import matplotlib.gridspec as gridspec
    
def close_loop(AO_sys,ZWFS_param, atm_seed=0):
    param = AO_sys.param
    type_rec = param['type_rec'] 
    display = param['display_loop']
    
    if display:
        plt.ion()
    
    # ----- basic AO init -----
    AO_sys.wfs.is_geometric = False
    AO_sys.tel.isPaired = False
    AO_sys.tel.resetOPD()
    AO_sys.dm.coefs = 0
    AO_sys.ttm.coefs = 0
    AO_sys.ngs.magnitude = param['magnitude_guide']
    AO_sys.science.magnitude = param['science_magnitude']

    # ---- reconstructor selection  ----
    if type_rec == 'keck':
        M2C_CL = np.eye(AO_sys.dm.nValidAct)
        Rec = AO_sys.keck_reconstructor
        modes_proj = AO_sys.dm.modes
        proj = AO_sys.projector_dm
        
    elif type == 'zonal':

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

    # ZWFS param 
    LuckyImaging = param['ZWFS_luckyImg']
    center_r = param['center_r']
    keepFraction = param['keepFraction']
    
    ZWFS_active = ZWFS_param['activate']
    ZWFS_iter_max = ZWFS_param['max_CLiter']
    gain_Z = ZWFS_param['gain']
    nrep_max = ZWFS_param['n_average']
    freq_zwfs = ZWFS_param['freq']
    maxZ = ZWFS_param['maxZ']
    subGZ = ZWFS_param['subGZ']
    maxGlobalZ = ZWFS_param['maxGlobalZ']
    avg_time = ZWFS_param['avg_time'] 
    
     # Derived timing
    expo_time = (1/freq_zwfs)   
    frames_per_img = int(expo_time / AO_sys.tel.samplingTime) # AO loop iterations per image (exposure / AO dt)    
    Nimgs = int(freq_zwfs * avg_time) # ZWFS exposures to average for one measurement
 
    # ---- ZWFS running accumulators  ----    
    inst_expo = []
    CRED2_image = []
    frame_count = 0
    CRED2_count = 0
    
    nrep = 0
    ZWFS_iter = 0
    coefs_seg_list = []

    # M1 amplitude history
    m1_z = float(np.std(AO_sys.opd_M1.OPD[np.where(AO_sys.tel.pupil > 0)])) * 1e9
    m1_amp = [m1_z]
    
    # KSM info
    seg_vect2D = AO_sys.seg2D
    segments = AO_sys.seg1D
    proj_seg = AO_sys.proj_seg
    Nseg = param['numberSegments']
    pupil_spider = AO_sys.tel.pupil
    z2p, p2z = zernikeBasis_nonCirc(maxGlobalZ, pupil_spider)
    
    # loop parameters
    gainCL = param['gainCL']
    gainTTM = param['gainTTM']
    leak = param['leak']
    latency = param['latency']
    nLoop = param['nLoop']
    bootstrap = 50

    # camera noise
    AO_sys.wfs.cam.photonNoise = True 
    if AO_sys.wfs.cam.photonNoise == True:
        AO_sys.wfs.cam.readoutNoise = param['ron']
        AO_sys.wfs.cam.darkCurrent = param['darkCurrent']
    
    # reference slopes
    if param['NCPA']:
        print('Applying the cog offset for the NCPA')
        AO_sys.wfs.reference_slopes_maps = AO_sys.cog_ncpa
        ncpa = AO_sys.opd_ncpa.OPD
        AO_sys.opd_ncpa.OPD = -ncpa
    else:
        print('No NCPA considered')
        AO_sys.wfs.reference_slopes_maps = AO_sys.cog_zeros
        ncpa = AO_sys.opd_ncpa.OPD*0
        AO_sys.opd_ncpa.OPD = -ncpa
    
    # Atmosphere propagation
    AO_sys.atm.generateNewPhaseScreen(seed=atm_seed)
    AO_sys.tel+AO_sys.atm
    AO_sys.ngs * AO_sys.tel * AO_sys.opd_M1
    AO_sys.tel * AO_sys.ttm * AO_sys.dm * AO_sys.wfs
    
    if ZWFS_active:
        ACS_iter = frames_per_img * Nimgs * nrep_max
        max_ACS = nLoop/ACS_iter
        print(f'ZWFS parameters nIterRec: {AO_sys.zwfs.nIterRec},doUnwrap:{AO_sys.zwfs.doUnwrap}, algo:{AO_sys.zwfs.algo}, pupil: {AO_sys.zwfs.pupilRec}')
        print(f"ZWFS setting for the simulations:\n ZWFS camera running at {freq_zwfs}Hz.")
        print(f"One ZWFS image is {frames_per_img} iterations of the AO loop.")
        print(f"Stack of {Nimgs} images to compute the segments coefficients.")
        print(f"One ACS command is sent after averaging {nrep_max} segment coefficients.")
        print(f"In total {ACS_iter} AO iterations are needed for ONE ACS command")
        print(f'You are running the simulation for {nLoop} iterations so a maximum of {max_ACS} ACS commands')

        if max_ACS <= ZWFS_iter_max:
            nLoop = ACS_iter*ZWFS_iter_max + int(1/AO_sys.tel.samplingTime *3)
            print(f'Update the number of iterations to {nLoop} for 5 ACS commands')

    # allocate memory to save data
    SR = np.zeros(nLoop, dtype=float)
    residual = np.zeros(nLoop, dtype=float)
    SRC_PSF_sum = np.zeros_like(AO_sys.science_detector.frame)   
    wfsSignal = np.arange(0,AO_sys.wfs.nSignal)*0
    total_M1_applied = 0

    psf_zwfs_trigger = False   # flag to know when to collect PSFs
    psf_count = 0                  # number of PSFs collected in current batch
    psf_batches = []               # will hold all batches (list of arrays)
    current_psf_batch = []         # temporary storage for 10 PSFs
    collecting_first10 = True
    psf_first10 = []
    psf_last10 = []
    ratio_time = int(AO_sys.science_detector.integrationTime/AO_sys.tel.samplingTime)

    # live plot
    if display:
        plot_obj = cl_plot(list_fig  = [AO_sys.atm.OPD,
                                AO_sys.tel.mean_removed_OPD,
                                AO_sys.wfs.cam.frame,
                                AO_sys.dm.OPD,
                                AO_sys.zwfs.getImageSimu(AO_sys.tel.src.phase),
                                AO_sys.opd_M1.OPD,
                                [[0,0],[0,0]],
                                AO_sys.zwfs.getImageSimu(AO_sys.tel.src.phase)],
                   type_fig          = ['imshow','imshow','imshow','imshow','imshow','imshow','plot','imshow'],
                   list_title        = ['Turbulence [nm]','NGS residual [m]','WFS Detector','DM OPD [nm]','ZWFS image','M1 OPD [nm]',None,'LE CRED2'],
                   list_legend       = [None,None,None,None,None,None,['SRC@'+str(AO_sys.science.coordinates[0])+'"','NGS@'+str(AO_sys.ngs.coordinates[0])+'"'],None,None],
                   list_label        = [None,None,None,None,None,None,['Iterations','WFE [nm]'],['Science PSF','']],
                   n_subplot         = [4,2],
                   list_display_axis = [None,None,None,None,None,None,True,None],
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)



    for i in range(nLoop):
        a = time.time()
        AO_sys.atm.update()
        
        # Add M1 phase and jitter
        AO_sys.ngs * AO_sys.tel * AO_sys.opd_M1
        tot = np.std(AO_sys.tel.OPD[np.where(AO_sys.tel.pupil > 0)]) * 1e9
        
        # Apply correction and go through WFS
        AO_sys.tel * AO_sys.ttm * AO_sys.dm * AO_sys.wfs

        # Compute fitting
        OPD_fitting_2D, OPD_corr_2D, OPD_turb_2D = getFittingError(AO_sys.tel.OPD, proj, modes_proj, display=False)
        fitt = np.std(OPD_fitting_2D[np.where(AO_sys.tel.pupil > 0)]) * 1e9
        
        #Science path
        AO_sys.science * AO_sys.tel * AO_sys.opd_ncpa * AO_sys.opd_offset * AO_sys.science_detector
        
        if ZWFS_active and i > bootstrap: # wait after bootstrap
            
            phase =  AO_sys.tel.OPD*2*np.pi/AO_sys.science.wavelength
            img_ao = AO_sys.zwfs.getImageSimu(phase) # take ZWFS @ AO loop frequency
            # accumulate AO-rate frames into one ZWFS exposure 
            inst_expo.append(img_ao)
            frame_count += 1
                        
            if frame_count == frames_per_img and ZWFS_iter < ZWFS_iter_max:
                inst_expo = np.array(inst_expo)
                zwfs_frame = np.mean(inst_expo, axis=0) # create the ZWFS image at the desired fps
                #reset
                inst_expo = []
                frame_count = 0
                    
                # accumulate ZWFS exposures to form the avg over avg_time
                CRED2_image.append(zwfs_frame)
                CRED2_count += 1
                print(f'Exposure {CRED2_count} at fps {freq_zwfs} (average of {frames_per_img})')
                               
                # when we have Nimgs exposures, compute img_mean and reconstruct
                if CRED2_count == Nimgs:
                    img_cube = np.array(CRED2_image)
                    #reset
                    CRED2_image = []
                    CRED2_count = 0
                    
                    # reconstruct piston per segment
                    img_mean = doLuckyImaging(img_cube,LuckyImaging,AO_sys.zwfs,center_r,keepFraction)
                    _, opd_wttf, seg_Zmode_coeffs = ReconPhase_Segments(img_mean,z2p,
                    p2z,pupil_spider,AO_sys.matching_inds,
                    seg_vect2D,AO_sys.zwfs,maxZ=maxZ,maxSeg=Nseg, subGZ=subGZ)
                
                    coef_ucsc = np.squeeze(seg_Zmode_coeffs)*1e-9
                    coefs_seg_list.append(coef_ucsc)
                    print(f'Piston estimation {nrep}')
                    nrep += 1

                # apply averaged correction when enough segment estimates collected
                if nrep == nrep_max: 
                    coefs_seg_list = np.array(coefs_seg_list)
                    mean_seg_coef = np.mean(coefs_seg_list, axis = 0)
                    seg_rec_ = np.dot(mean_seg_coef,segments).reshape(param['resolution'],param['resolution'])
                    seg_rec_ = seg_rec_ * AO_sys.tel.pupil
                    seg_rec = seg_rec_ - np.mean(seg_rec_[np.where(AO_sys.tel.pupil>0)])
                    total_M1_applied += seg_rec
                
                    AO_sys.opd_M1.OPD = AO_sys.opd_M1.OPD - gain_Z * seg_rec
                    m1_z = float(np.std(AO_sys.opd_M1.OPD[np.where(AO_sys.tel.pupil > 0)])) * 1e9
                    m1_amp.append(m1_z)
                    
                    if param['print_display']:
                        print('Correction ZWFS applied Loop' + str(i) + 
                            '/' + str(nLoop) + ' M1 oppd: ' + str(m1_amp[-1]) + 'nm RMS Ziter:' + str(ZWFS_iter)+ '\n')
                    
                    print(f'ACS command number {ZWFS_iter}')
                    ZWFS_iter +=1
                    #reset
                    nrep = 0                   
                    coefs_seg_list = []
                    
                    # ---- Trigger PSF collection ----
                    psf_zwfs_trigger = True
                    psf_count = 0
                    current_psf_batch = []


        # ---------------- AO control command (WFS -> DM/TT) ----------------        
        if latency ==1:        
            wfsSignal = AO_sys.wfs.signal
        
        command = Rec @ wfsSignal
        command_tronc = command.copy()
        command_tronc[np.where(np.abs(command_tronc)<1e-10)]=0
        com_dm = command_tronc[:AO_sys.dm.nValidAct]
        com_ttm = command[AO_sys.dm.nValidAct:]
    
        AO_sys.dm.coefs = leak*AO_sys.dm.coefs - gainCL * com_dm
        dm_opd = AO_sys.dm.OPD*AO_sys.tel.pupil
        AO_sys.ttm.coefs = AO_sys.ttm.coefs - gainTTM * com_ttm
        
        if latency ==2:        
            wfsSignal = AO_sys.wfs.signal
        
        # ---------------- diagnostics storage  ----------------
        sr = np.exp(-np.var(AO_sys.tel.src.phase[np.where(AO_sys.tel.pupil == 1)]))
        SR[i] = sr
        OPD_res = AO_sys.tel.OPD
        res = np.std(OPD_res[np.where(AO_sys.tel.pupil > 0)]) * 1e9
        residual[i] = res
        
        # Science image    
        psf = AO_sys.science_detector.frame[:]
        if i>=bootstrap and i%ratio_time==0:
            SRC_PSF_sum  += psf 
            
            if collecting_first10 and i > bootstrap:
                psf_first10.append(psf)
                if len(psf_first10) == 10:
                    collecting_first10 = False  # stop collecting after 10
                    psf_batches.append(np.array(psf_first10))
                    psf_first10 = []
                    print(f"Stored First PSF batch #{len(psf_batches)} (10 images)")
           
            if i>nLoop-ratio_time*11:
                psf_last10.append(psf)
                if len(psf_last10) == 10:
                    psf_batches.append(np.array(psf_last10))
                        
            # --- If we are in PSF collection mode, store 10 PSFs ---
            if psf_zwfs_trigger:
                current_psf_batch.append(psf.copy())
                if len(current_psf_batch) == 10:  # collected 10 PSFs
                    psf_batches.append(np.array(current_psf_batch))
                    psf_zwfs_trigger = False
                    print(f"Stored PSF batch #{len(psf_batches)} (10 images)")
            
        if display and i>bootstrap:
            pp = np.abs(psf)
            pp[pp<=0] = np.nan
            psf_plot = np.log10(pp)
            
            plot_obj.list_lim = [None,None,None,None,None,None,None,[None, None]]        
            # update title
            plot_obj.list_title = ['Turbulence '+str(np.round(tot))+'[nm]',
                               'AO residual '+str(np.round(res))+'[nm]',
                               'WFS Detector',
                               'DM OPD',
                               'ZWFS image',
                               'M1 OPD'+str(np.round(m1_amp[-1]))+'[nm]',
                               None,
                               'CRED2 LE']
            if ZWFS_active:           
                if 'img_mean' in locals() and isinstance(img_mean, np.ndarray):
                    LE_CRED2 = img_mean
                else:
                    LE_CRED2 = np.zeros_like(img_ao)
            else:
                img_ao = AO_sys.zwfs.getImageSimu(AO_sys.tel.src.phase) *0
                LE_CRED2 = np.zeros_like(img_ao)
                
            cl_plot(list_fig   = [1e9*AO_sys.atm.OPD,1e9*AO_sys.tel.OPD*AO_sys.pup_crop,AO_sys.wfs.cam.frame,AO_sys.dm.OPD*AO_sys.tel.pupil*1e9, img_ao, AO_sys.opd_M1.OPD*1e9, [np.arange(i+1),residual[:i+1]],LE_CRED2],plt_obj = plot_obj)
            
            plt.pause(0.01)
            if plot_obj.keep_going is False:
                break
                
        if param['print_display']:
            print('Loop' + str(i) + '/' + str(nLoop) + ' Turbulence: ' + str(tot) + ' -- Residual:' +
                str(res) + ' -- Fitting:' + str(fitt) + '\n')

    # plot the residual and the SR in the science band
    print('Average SR = ', np.mean(SR[bootstrap:i])*100)

    PSF_LE = SRC_PSF_sum/float(nLoop-bootstrap)
    PSF_LE_norm =PSF_LE/ np.max(PSF_LE)  # Normalize PSF
    psf_array = np.array(psf_batches, dtype=np.float32)

    if ZWFS_active:
        
        output = { 'PSF_LE': PSF_LE, 'PSF_batches':psf_array,'SR': SR, 'residual': residual, 'M1_OPD':m1_amp, 'total_M1_applied':total_M1_applied, 'M1_final_shape':AO_sys.opd_M1.OPD}
        
    else:
        output = {'PSF_LE': PSF_LE,'PSF_batches':psf_array, 'SR': SR,'residual': residual}     

    AO_sys.tel-AO_sys.atm               

    return output
