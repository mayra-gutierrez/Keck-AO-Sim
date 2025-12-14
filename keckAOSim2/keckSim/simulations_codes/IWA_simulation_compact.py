#%%
import time
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import welch

from scipy import sparse
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
from OOPAO.Pyramid import Pyramid

import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim")
from keckTel import keckTel,keckStandard
from keckAtm import keckAtm

import os, sys
os.chdir("/home/mcisse/keckAOSim/keckSim/")
sys.path.insert(0, os.getcwd())
from simulations_codes.IWA_parameter_file import initializeParameterFile
from simulations_codes.initialize_IWA import initialize_AO_hardware
from simulations_codes.close_loop_IWA import close_loop

plt.ion()
# %% -----------------------     Import system  ----------------------------------

path = '/home/mcisse/keckAOSim/keckSim/data/'
param = initializeParameterFile()
IWA = initialize_AO_hardware(param)

#%% 
'''

D = IWA.calib_modal.D
M = IWA.calib_modal.M

DT = IWA.calib_TTM.D
MT = IWA.calib_TTM.M

Rec_svd = np.vstack([M,MT])
Rec = IWA.rec_modal_SVD

IWA.tel-IWA.atm
IWA.tel.isPaired = True
IWA.tel.resetOPD()
IWA.dm.coefs = IWA.M2C[:,0]*1e-7
IWA.ttm.coefs = 0
IWA.ttm.coefs = np.array([1e-7,0])

opd_in = IWA.ttm.OPD+IWA.dm.OPD
opd_map = OPD_map(telescope = IWA.tel)
opd_map.OPD = opd_in

IWA.ttm.coefs = 0
IWA.dm.coefs = 0
IWA.ngs*IWA.tel*opd_map*IWA.ttm*IWA.dm*IWA.wfs
plt.figure(), plt.imshow(IWA.tel.OPD*1e9), plt.colorbar(), plt.title('Tel shape nm')

wfsSignal=IWA.wfs.signal
plt.figure(), plt.imshow(IWA.wfs.signal_2D), plt.colorbar()
com = np.matmul(Rec,wfsSignal) 
com_dm = com[:IWA.dm.nValidAct]
com_ttm = com[IWA.dm.nValidAct:]
IWA.dm.coefs = IWA.dm.coefs-com_dm
IWA.ttm.coefs = IWA.ttm.coefs-com_ttm
plt.figure(), plt.imshow(IWA.dm.OPD*IWA.tel.pupil*1e9), plt.colorbar(), plt.title('DM shape nm')
plt.figure(), plt.imshow(IWA.ttm.OPD*IWA.tel.pupil*1e9), plt.colorbar(), plt.title('TTM shape nm')

IWA.tel.resetOPD()
IWA.ngs*IWA.tel*opd_map*IWA.ttm*IWA.dm*IWA.wfs
plt.figure(), plt.imshow(IWA.tel.OPD*1e9), plt.colorbar(), plt.title('Tel shape nm output')

'''
param['display_loop'] = True
param['print_display'] = True
param['gainCL'] = 0.4
param['gainTTM']= 0.2
param['type_rec'] = 'SVD'
param['nLoop' ]=500
AO_data = close_loop(IWA)


'''
import time

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
import matplotlib.gridspec as gridspec


# %%
plt.ion()
# number of subaperture for the WFS
n_subaperture = 60


#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 6*n_subaperture,                         
                diameter             = 10.95,                                    
                samplingTime         = 1/1000,                                  
                centralObstruction   = 0.1,                                      
                display_optical_path = False,                                    
                fov                  = 10 )                                    

# display current pupil
plt.figure()
plt.imshow(tel.pupil)

#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

ngs = Source(optBand     = 'H',         
             magnitude   = 8,            
             coordinates = [0,0])         
ngs*tel

src = Source(optBand     = 'K',           
             magnitude   = 8,             
             coordinates = [1,0])        
src*tel


#%% -----------------------     ATMOSPHERE   ----------------------------------
from OOPAO.Atmosphere import Atmosphere
           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               
                 r0            = 0.15,                              
                 L0            = 25,                               
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], 
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], 
                 windDirection = [0    ,72   ,144  ,216   ,288   ], 
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ])
                 
atm.initializeAtmosphere(tel)
atm.update()

# display the atm.OPD = resulting OPD 
plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration

nAct = 56+1#n_subaperture+1
    
dm = DeformableMirror(telescope  = tel,                       
                    nSubap       = nAct-1,                     
                    mechCoupling = 0.35, 
                    coordinates  = None,                      
                    pitch        = tel.D/nAct)               

# plot the dm actuators coordinates with respect to the pupil
plt.figure()
plt.imshow(np.reshape(np.sum(dm.modes**5,axis=1),[tel.resolution,tel.resolution]).T + tel.pupil,extent=[-tel.D/2,tel.D/2,-tel.D/2,tel.D/2])
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'rx')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')


#%% -----------------------     PYRAMID WFS   ----------------------------------
from OOPAO.Pyramid import Pyramid

# make sure tel and atm are separated to initialize the PWFS
tel.isPaired = False
tel.resetOPD()

wfs = Pyramid(nSubap            = n_subaperture,               
              telescope         = tel,                          
              lightRatio        = 0.5,                          
              modulation        = 3,                            
              binning           = 1,                           
              n_pix_separation  = 4,                           
              n_pix_edge        = 2,                            
              postProcessing    = 'fullFrame_incidence_flux')  # slopesMaps,
#slopesMaps_incidence_flux #fullFrame_incidence_flux
tel*wfs

plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame')
plt.colorbar()

plt.figure()
plt.imshow(wfs.signal_2D)
plt.title('WFS Signal')
plt.colorbar()

# The photon Noise of the detector can be disabled the same way than for a Detector class
wfs.cam.photonNoise = True

ngs*tel*wfs

plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame - Without Noise')
plt.colorbar()

wfs.cam.photonNoise = False
ngs*tel*wfs
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title('WFS Camera Frame - With Noise')
plt.colorbar()

#%% -----------------------     Modal Basis - KL Basis  ----------------------------------

from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
M2C_KL = compute_KL_basis(tel, atm, dm,lim = 1e-2,n_batch = 10) 

dm.coefs = M2C_KL[:,:10]
ngs*tel*dm
displayMap(tel.OPD)
#%% -----------------------     Calibration: Interaction Matrix  ----------------------------------

stroke=ngs.wavelength/16
M2C_zonal = np.eye(dm.nValidAct)
M2C_modal = M2C_KL[:,:300]

tel-atm
calib_modal = InteractionMatrix(ngs            = ngs,
                                atm            = atm,
                                tel            = tel,
                                dm             = dm,
                                wfs            = wfs,   
                                M2C            = M2C_modal, 
                                stroke         = stroke,    
                                nMeasurements  = 6,       
                                noise          = 'off',    
                                display        = True,     
                                single_pass    = True)     

plt.figure()
plt.plot(np.std(calib_modal.D,axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')

#%% Define instrument and WFS path detectors
from OOPAO.Detector import Detector

# instrument path
src_cam = Detector(tel.resolution*4)
src_cam.psf_sampling = 4
src_cam.integrationTime = tel.samplingTime*1
# put the scientific target off-axis to simulate anisoplanetism (set to  [0,0] to remove anisoplanetism)
src.coordinates = [0.4,0]

# WFS path
ngs_cam = Detector(tel.resolution)
ngs_cam.psf_sampling = 4
ngs_cam.integrationTime = tel.samplingTime

# initialize Telescope DM commands
tel.resetOPD()
dm.coefs=0
ngs*tel*dm*wfs
wfs*wfs.focal_plane_camera
# Update the r0 parameter, generate a new phase screen for the atmosphere and combine it with the Telescope
# atm.r0 = 0.15
atm.generateNewPhaseScreen(seed = 10)
tel+atm

tel.computePSF(4)
#plt.close('all')
    
# These are the calibration data used to close the loop
calib_CL    = calib_modal
M2C_CL      = M2C_modal


# combine telescope with atmosphere
tel+atm

# initialize DM commands
atm*ngs*tel*ngs_cam
atm*src*tel*src_cam

plt.show()

nLoop = 500
# allocate memory to save data
SR_NGS                      = np.zeros(nLoop)
SR_SRC                      = np.zeros(nLoop)
total                       = np.zeros(nLoop)
residual_SRC                = np.zeros(nLoop)
residual_NGS                = np.zeros(nLoop)

wfsSignal               = np.arange(0,wfs.nSignal)*0

plot_obj = cl_plot(list_fig          = [atm.OPD,
                                        tel.mean_removed_OPD,
                                        tel.mean_removed_OPD,
                                        [[0,0],[0,0],[0,0]],
                                        wfs.cam.frame,
                                        wfs.focal_plane_camera.frame,
                                        np.log10(tel.PSF),
                                        np.log10(tel.PSF)],
                   type_fig          = ['imshow',
                                        'imshow',
                                        'imshow',
                                        'plot',
                                        'imshow',
                                        'imshow',
                                        'imshow',
                                        'imshow'],
                   list_title        = ['Turbulence [nm]',
                                        'NGS@'+str(ngs.coordinates[0])+'" WFE [nm]',
                                        'SRC@'+str(src.coordinates[0])+'" WFE [nm]',
                                        None,
                                        'WFS Detector',
                                        'WFS Focal Plane Camera',
                                        None,
                                        None],
                   list_legend       = [None,None,None,['SRC@'+str(src.coordinates[0])+'"','NGS@'+str(ngs.coordinates[0])+'"'],None,None,None,None],
                   list_label        = [None,None,None,['Time','WFE [nm]'],None,None,['NGS PSF@'+str(ngs.coordinates[0])+'" -- FOV: '+str(np.round(ngs_cam.fov_arcsec,2)) +'"',''],['SRC PSF@'+str(src.coordinates[0])+'" -- FOV: '+str(np.round(src_cam.fov_arcsec,2)) +'"','']],
                   n_subplot         = [4,2],
                   list_display_axis = [None,None,None,True,None,None,None,None],
                   list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)

# loop parameters
gainCL                  = 0.4
wfs.cam.photonNoise     = False
display                 = True
frame_delay             = 2
reconstructor = M2C_CL@calib_CL.M

for i in range(nLoop):
    a=time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # propagate light from the NGS through the atmosphere, telescope, DM to the WFS and NGS camera with the CL commands applied
    atm*ngs*tel*dm*wfs*ngs_cam
    wfs*wfs.focal_plane_camera
    # save residuals corresponding to the NGS
    residual_NGS[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD_NGS         = tel.mean_removed_OPD.copy()

    if display==True:        
        NGS_PSF = np.log10(np.abs(ngs_cam.frame))
    
    # propagate light from the SRC through the atmosphere, telescope, DM to the Instrument camera
    atm*src*tel*dm*src_cam
    
    # save residuals corresponding to the NGS
    residual_SRC[i] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD_SRC         = tel.mean_removed_OPD.copy()
    if frame_delay ==1:        
        wfsSignal=wfs.signal
    
    # apply the commands on the DM
    dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,wfsSignal)
    
    # store the slopes after computing the commands => 2 frames delay
    if frame_delay ==2:        
        wfsSignal=wfs.signal
    
    print('Elapsed time: ' + str(time.time()-a) +' s')
    
    # update displays if required
    if display==True and i>0:        
        
        SRC_PSF = np.log10(np.abs(src_cam.frame))
        # update range for PSF images
        plot_obj.list_lim = [None,None,None,None,None,None,[NGS_PSF.max()-3, NGS_PSF.max()],[SRC_PSF.max()-4, SRC_PSF.max()]]        
        # update title
        plot_obj.list_title = ['Turbulence WFE:'+str(np.round(total[i]))+'[nm]',
                               'NGS@'+str(ngs.coordinates[0])+'" WFE:'+str(np.round(residual_NGS[i]))+'[nm]',
                               'SRC@'+str(src.coordinates[0])+'" WFE:'+str(np.round(residual_SRC[i]))+'[nm]',
                                None,
                                'WFS Detector',
                                'WFS Focal Plane Camera',
                                None,
                                None]

        cl_plot(list_fig   = [1e9*atm.OPD,1e9*OPD_NGS,1e9*OPD_SRC,[np.arange(i+1),residual_SRC[:i+1],residual_NGS[:i+1]],wfs.cam.frame,wfs.focal_plane_camera.frame,NGS_PSF, SRC_PSF],
                               plt_obj = plot_obj)
        plt.pause(0.001)
        if plot_obj.keep_going is False:
            break
    print('Loop'+str(i)+'/'+str(nLoop)+' NGS: '+str(residual_NGS[i])+' -- SRC:' +str(residual_SRC[i])+ '\n')

'''




