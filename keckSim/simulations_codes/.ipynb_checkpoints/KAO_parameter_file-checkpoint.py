
import numpy as np
from os import path
from astropy.io import fits
from OOPAO.tools.tools  import createFolder


def initializeParameterFile():
    # initialize the dictionaries
    param = dict()
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATMOSPHERE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['ZenithAngle'          ] = 0*np.pi/180                                                    # zenith angle in radians                      
    param['r0'                   ] = 0.16                                                           # value of r0 @500 nm in [m]
    param['L0'                   ] = 80 #50                                                             # value of r0 @500 nm in [m]
    param['fractionnalR0'        ] = [0.517, 0.119, 0.063, 0.061, 0.105, 0.081, 0.054]              # Cn2 profile
    param['windSpeed'            ] = np.array([6.8, 6.9, 7.1, 7.5, 10.0, 26.9, 18.5])-1                         # wind speed of the different layers in [m.s-1]
    param['windDirection'        ] = [0, np.pi/2, np.pi/4, 3*np.pi/2, 8*np.pi/3, np.pi/8, np.pi]    # wind direction of the different layers in [degrees]
    param['altitude'             ] = np.array([0.1, 0.5, 1, 2, 4, 8, 16])*1e3                       # altitude of the different layers in [degrees]
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TELESCOPE PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    param['diameter'             ] = 10.95  # diameter in [m]
    param['resolution'           ] = 480  #480 #336    # resolution of the telescope driven by the PWFS
    param['numberSegments'       ] = 36     # number of segments
    param['samplingTime'         ] = 1/1000  # loop sampling time in [s] == AO Loop frequency
    param['centralObstruction'   ] = 0.2356 # central obstruction in percentage of the diameter
    param['m1_reflectivity'      ] = 1      # reflectivity of the pupil
    param['D_outer'              ] = 9.82   # outer diameter im m
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% M1 aberrations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['M1_segments_pistons' ] = True    # piston of the segments
    param['M1_segments_tilts'   ] = False    # tilts of the segments
    param['M1_OPD_amplitude'    ] = 0*1e-9 # M1 OPD amplitude in [nm] RMS

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Filter TT & segments piston BASIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['regularization_TT'    ] = 0
    param['regularization_seg'   ] = 100
    param['filter_segments'      ] = False
    param['filter_TT'            ] = True

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NCPA aberrations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['NCPA'                ] = False        # to enable NCPA
    param['NCPA_nb_modes'       ] = 15          # number of modes to be used in the NCPA
    param['NCPA_amplitude'      ] = 75*1e-9    # NCPA amplitude in [nm] RMS
    param['NCPA_nmin'           ] = 5           # minimum radial order of the Zernike modes
    param['NCPA_nmax'           ] = 15          # maximum radial order of the Zernike modes
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Zernike aberrations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nb_Zpolynomials'     ] = 10          # define a Zernike basis

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Jitter %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['Jitter'              ] = False        # to enable NCPA
    param['Jitter_amp'          ] = 50          # Jitter amp in nm RMS
    param['Jitter_freq'         ] = 50          # Jitter frequency in Hz
    param['Jitter_nFrames'      ] = 2000        # Jitter nFrames

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NGS PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['magnitude_guide'      ] = 5 #8      # magnitude of the guide star
    param['opticalBand_guide'    ] = 'V'    # optical band of the guide star
    param['ngs_coordinate'       ] = [0,0]  # coordinate of the target in mas
    
    param['science_magnitude'    ] = 5      # magnitude of the guide star
    param['science_opticalBand'  ] = 'L'    # optical band of the guide star
    param['science_coordinate'   ] = [0,0]  # coordinate of the target in mas
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SH PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nSubaperture'         ] = 20 #28    # number of SHWFS subaperture along the telescope diameter
    param['nPixelPerSubap'       ] = 4     # sampling of the PWFS subapertures
    param['plateScale'           ] = 0.8   # pixel scale in arcsec
    param['shannon'              ] = False
    param['lightThreshold'       ] = 0.36   # light threshold to select the valid pixels
    param['is_geometric'         ] = False # post-processing of the PWFS signals 
    param['threshold_cog'        ] = 0.1   # COG threshold
    param['binning'              ] = 0     # binning of the detctor
   
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SH dectector PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['detector_wfs'         ] = None    #None means CCD
    param['photonNoise'          ] = True 	# to enable photon noise
    param['ron'                  ] = 0.5  	# value of the RON in e/pix/frame
    param['darkCurrent'          ] = 0    	# dark current in e/pix/frame
    param['QE'                   ] = 0.9 	# QE of the detector
    param['em_gain'	         ] = 1		# OCAM gain

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DM PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['nActuator'            ] = param['nSubaperture']+1 #56+1  # number of actuators 
    param['mechanicalCoupling'   ] = 0.1458                   # mechanical coupling between actuators
    param['dm_coordinates'       ] = None                     # using user-defined coordinates
    param['validActTreshold'     ] = 0.1                      # illumination treshold to define the valid actuator

    # mis-registrations                                                             
    param['shiftX'               ] = 0                        # shift X of the DM in pixel size units ( tel.D/tel.resolution ) 
    param['shiftY'               ] = 0                        # shift Y of the DM in pixel size units ( tel.D/tel.resolution )
    param['rotationAngle'        ] = 0                        # rotation angle of the DM in [degrees]
    param['anamorphosisAngle'    ] = 0                        # anamorphosis angle of the DM in [degrees]
    param['radialScaling'        ] = 0                        # radial scaling in percentage of diameter
    param['tangentialScaling'    ] = 0                        # tangential scaling in percentage of diameter
    
    param['TTnAct'               ] = 2 
    param['TT_coupling'          ] = param['mechanicalCoupling']
    param['stroke'               ] = 50*1e-9                  # stroke to calibrate the DM in [m]
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  MODAL BASIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['modal_basis'          ] ='KL'       # name of the basis
    param['nModes'               ] = 280 #540            # number of controlled modes  
    param['save_basis'           ] = False
    
    param['SVD_thr'              ] = 0
    param['modal_trunc'	         ] = 0
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% science detector PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['scienceDetectorRes'      ] = 1024 #512                         # resolution of the science detcetor
    param['science_photonNoise'     ] = True                        # to enable photon noise
    param['science_ron'             ] = 0.5                         # value of the RON in e/pix/frame
    param['science_darkCurrent'     ] = 0                           # dark current in e/pix/frame
    param['science_QE'              ] = 0.92                        # QE of the detector
    param['science_psf_sampling'    ] = 4                           # PSF sampling on the scientific detector in shannon
    param['science_binning'         ] = 1                           # PSF binning
    param['science_integrationTime' ] = param['samplingTime']*10    # integration time in s

    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOOP PROPERTIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    param['nLoop'               ] = 200     # number of iteration                             
    param['gainCL'              ] = 0.4     # integrator gain
    param['getProjector'        ] = True    # modal projector too get modal coefficients of the turbulence and residual phase
    param['latency'             ] = 1       # frame delay
    param['type_rec'            ] = 'modal'  # reconstructor type
    param['display_loop'        ] = False   # display plot when closing the loop
    param['print_display'       ] = False
    
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ZWFS param %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    param['ZWFS_wavelength'     ] = 1600                                  # wavelength of the ZWFS in [nm]
    param['ZWFS_dimple_pixels'  ] = 15                                    # dimple size in pixels
    param['ZWFS_diameter'       ] = np.array([2.4, 2.4])                  # np.array([2.717,2.717])
    param['ZWFS_depth'          ] = np.array([0.3 * np.pi, 0.68 * np.pi]) # depth of the dimple in [radians]
    
    param['ZWFS_nIterRec'       ] = 5                                     # number of iteration for the M1 reconstruction
    param['ZWFS_pupil_both'     ] = 'both'                                # pupil to use for the M1 loop
    param['ZWFS_pupil_single'   ] = 'right'                               # pupil to use for the M1 loop
    param['ZWFS_doUnwrap'       ] = 0                                     # Activate do Unwarp for M1 loop
    param['ZWFS_algo'           ] = 'JPL'                                 # algo for the TTM loop
        
    ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # name of the system
    param['name'     ] = f"KAO_{param['opticalBand_guide']}_band_SH_{param['nSubaperture']}x{param['nSubaperture']}" 
    param['pathInput'] = '/home/mcisse/keckAOSim/keckSim/data/'
    
    return param
