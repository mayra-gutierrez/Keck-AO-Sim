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
    
def close_loop(AO_sys, atm_seed=0):
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
    
    basis = AO_sys.basis
    tip = basis[:,0].reshape(AO_sys.param['resolution'],AO_sys.param['resolution'])*AO_sys.ngs.wavelength/(2*np.pi)
    tilt = basis[:,1].reshape(AO_sys.param['resolution'],AO_sys.param['resolution'])*AO_sys.ngs.wavelength/(2*np.pi)
    # State for tip and tilt
    jitter_x = 0.0
    jitter_y = 0.0
    a = AO_sys.jitter[0]
    b = AO_sys.jitter[1]

    # ---- reconstructor selection  ----
    if type_rec == 'zonal':
        M2C_CL = np.eye(AO_sys.dm.nValidAct)
        Rec = AO_sys.keck_reconstructor
        modes_proj = AO_sys.dm.modes
        proj = AO_sys.projector_dm
        print('Keck Zonal Reconstruction')
        
    elif type_rec == 'modal':   
        Rec = AO_sys.keck_reconstructor_modal        
        modes_proj = AO_sys.basis
        proj = AO_sys.projector_modal
        print('Keck Modal Reconstruction')
    else: 
        Rec = AO_sys.rec_modal_SVD
        modes_proj = AO_sys.basis
        proj = AO_sys.projector_modal
        print('SVD Modal Reconstruction')
        
    # loop parameters
    gainCL = param['gainCL']
    gainTTM = param['gainTTM']
    leak = param['leak']
    latency = param['latency']
    nLoop = param['nLoop']
    bootstrap = 10

    # camera noise
    AO_sys.wfs.cam.photonNoise = True 
    if AO_sys.wfs.cam.photonNoise == True:
        AO_sys.wfs.cam.readoutNoise = param['ron']
        AO_sys.wfs.cam.darkCurrent = param['darkCurrent']
    
    
    # Atmosphere propagation
    AO_sys.atm.generateNewPhaseScreen(seed=atm_seed)
    AO_sys.tel+AO_sys.atm
    AO_sys.ngs * AO_sys.tel * AO_sys.opd_M1
    AO_sys.tel * AO_sys.ttm * AO_sys.dm * AO_sys.wfs
    
  
    # allocate memory to save data
    SR = np.zeros(nLoop, dtype=float)
    residual = np.zeros(nLoop, dtype=float)
    SRC_PSF_sum = np.zeros_like(AO_sys.science_detector.frame)   
    wfsSignal = np.arange(0,AO_sys.wfs.nSignal)*0
    total_M1_applied = 0
    amp_jitter = []

    ratio_time = int(AO_sys.science_detector.integrationTime/AO_sys.tel.samplingTime)

    # live plot
    if display:
        plot_obj = cl_plot(list_fig  = [AO_sys.atm.OPD,
                                AO_sys.tel.mean_removed_OPD,
                                AO_sys.wfs.cam.frame,
                                AO_sys.dm.OPD,
                                AO_sys.wfs.focal_plane_camera.frame,
                                AO_sys.ttm.OPD,
                                [[0,0],[0,0]],
                                AO_sys.opd_M1.OPD*0],
                   type_fig          = ['imshow','imshow','imshow','imshow','imshow','imshow','plot','imshow'],
                   list_title        = ['Turbulence [nm]','NGS residual [m]','WFS Detector','DM OPD [nm]','Focal plane image','TTM OPD [nm]',None,None],
                   list_legend       = [None,None,None,None,None,None,['SRC@'+str(AO_sys.science.coordinates[0])+'"','NGS@'+str(AO_sys.ngs.coordinates[0])+'"'],None,None],
                   list_label        = [None,None,None,None,None,None,['Iterations','WFE [nm]'],['Science PSF','']],
                   n_subplot         = [4,2],
                   list_display_axis = [None,None,None,None,None,None,True,None],
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)

    for i in range(nLoop):
        AO_sys.atm.update()
        
        # Add M1 phase and Jitter 
        if param['Jitter']:
            jitter_x = a * jitter_x + b*np.random.randn()
            jitter_y = a * jitter_y + b*np.random.randn()
            AO_sys.opd_jitter.OPD = jitter_x*tip+jitter_y*tilt
        else:
            AO_sys.opd_jitter.OPD = np.zeros((param['resolution'],param['resolution']))
            
        amp_jitter.append(np.std(AO_sys.opd_jitter.OPD[np.where(AO_sys.tel.pupil > 0)]) * 1e9)
             
        AO_sys.ngs * AO_sys.tel * AO_sys.opd_M1 * AO_sys.opd_jitter
        tot = np.std(AO_sys.tel.OPD[np.where(AO_sys.tel.pupil > 0)]) * 1e9
        
        # Apply correction and go through WFS
        AO_sys.tel * AO_sys.ttm * AO_sys.dm * AO_sys.wfs
        AO_sys.wfs*AO_sys.wfs.focal_plane_camera

        # Compute fitting
        OPD_fitting_2D, OPD_corr_2D, OPD_turb_2D = getFittingError(AO_sys.tel.OPD, proj, modes_proj, display=False)
        fitt = np.std(OPD_fitting_2D[np.where(AO_sys.tel.pupil > 0)]) * 1e9
        
        #Science path
        AO_sys.science * AO_sys.tel * AO_sys.opd_ncpa * AO_sys.science_detector

        # ---------------- AO control command (WFS -> DM/TT) ----------------        
        if latency ==1:        
            wfsSignal = AO_sys.wfs.signal
        
        command = Rec @ wfsSignal
        command_tronc = command.copy()
        com_dm = command_tronc[:AO_sys.M2C.shape[1]]
        com_dm[:2] = 0
        com_dm_apply = AO_sys.M2C @ com_dm
        com_ttm = command[AO_sys.M2C.shape[1]:]
    
        AO_sys.dm.coefs = leak*AO_sys.dm.coefs - gainCL * com_dm_apply
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
                               'Focal plane image image',
                               'TTM OPD',
                               None,
                               None]
                
            cl_plot(list_fig   = [1e9*AO_sys.atm.OPD,1e9*AO_sys.tel.OPD,AO_sys.wfs.cam.frame,AO_sys.dm.OPD*AO_sys.tel.pupil*1e9, AO_sys.wfs.focal_plane_camera.frame, AO_sys.ttm.OPD*1e9, [np.arange(i+1),residual[:i+1]],AO_sys.opd_M1.OPD*1e9*0],plt_obj = plot_obj)
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

    output = {'PSF_LE': PSF_LE, 'SR': SR,'residual': residual,'jitter':amp_jitter}     

    AO_sys.tel-AO_sys.atm               

    return output
