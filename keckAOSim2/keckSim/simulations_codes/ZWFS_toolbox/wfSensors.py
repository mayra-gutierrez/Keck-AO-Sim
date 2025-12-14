#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for wavefront sensors:
	* 4-sided Pyramid WFS
	* Zernike WFS
	* Shack-Hartmann WFS
	* Photonic Lantern WFS

.. warning::
You can add any new object, but you will need to have this 2 methods in your class:
	 * wfs.getImage() : record WFS signal (can be already processed)
	 * wfs.img2cmd(img) : reconstruct command (as vector) to send to DM from measurement

.. note::
	 More sensors should be added: FAST, focal plane WFS (EFC)

"""

from xmlrpc.client import boolean
from tqdm import tqdm, trange
#from tools import *
from numpy import matlib
from abc import ABCMeta, abstractmethod
from itertools import product

import os, sys
os.chdir("/Users/mayragutierrez/home/lab/maygut/keckAOSim2/keckSim")
from simulations_codes.ZWFS_toolbox.tools import *

class WavefrontSensor(metaclass=ABCMeta):
	"""
	Abstract base class for wavefront sensors; 
	makes sure that any WFS implements the essential methods
	and defines common functionality for any WFS.
	"""
	
	@abstractmethod
	def getImage(self): 
		pass
	
	@abstractmethod
	def img2cmd(self, img):
		pass

	def calibrate(self,dm, amp_calib=0.05, modes_number = None):
		"""
		Push-pull calibration for the given DM. It can be zonal or modal (only Zernike modes for now).

			Parameters:
				dm (DM_seal object): DM for used for calibration
				modal (bool): Set if the calibration is done on a modal basis - Zernike modes (=1) - or not (=0)
		"""
		if self.modal == 'zonal':
			# ---- ZONAL ---------
			self.validActuators = dm.valid_actuators_map
			self.intMat = np.zeros((self.output_size,int(np.sum(dm.valid_actuators_map))))
			compt = 0

			for (i, j) in tqdm(list(product(range(dm.nAct), range(dm.nAct)))):
				if dm.valid_actuators_map[i,j] == 1:
					# --------- PUSH --------
					dm.pokeAct(amp_calib,[i,j])
					time.sleep(0.1)
					s_push = self.getImage()
					# --------- PULL --------
					dm.pokeAct(-amp_calib,[i,j])
					time.sleep(0.1)
					s_pull = self.getImage()
					# -------- Push-pull ------
					s = (s_push-s_pull)/(2*amp_calib)
					if self.model == 'vector ZWFS': # keeping only one pupil for vector ZWFS
						s = s[:,0:int(self.nPx_img)]/np.sum(s[:,0:int(self.nPx_img)])
					self.intMat[:,compt] = s.ravel()
					compt = compt + 1
		elif self.modal == 'zernike':
			if modes_number is None:
				nModes = dm.Z2C.shape[1]
			else:
				nModes = modes_number
			self.intMat = np.zeros((self.output_size,nModes))
			for k in trange(nModes):
					# --------- PUSH --------
					dm.pokeZernike(amp_calib,k+2)
					time.sleep(0.1)
					s_push = self.getImage()
					# --------- PULL --------
					dm.pokeZernike(-amp_calib,k+2)
					time.sleep(0.1)
					s_pull = self.getImage()
					# -------- Push-pull ------
					s = (s_push-s_pull)/(2*amp_calib)
					if self.model == 'vector ZWFS': # keeping only one pupil for vector ZWFS
						s = s[:,0:int(self.nPx_img)]/np.sum(s[:,0:int(self.nPx_img)])
					self.intMat[:,k] = s.ravel()
					# print('Percentage done: ',100*k/nModes)
		else:
			print('NOT IMPLEMENTED YET')
		dm.setFlatSurf()
		if np.all(self.intMat == 0):
			raise RuntimeError("All elements of the interaction matrix are zero! Double check that your WFS is running.")
		self.compute_cmdMat()

	def linearity(self, dm, amp_calib=0.05, modes_number=None, lim=1.0, step=0.1, recalibrate=True, plot=True):
		"""
		Make linearity curves for the WFS by sweeping (-lim, lim) in increments of `step`, and optionally plot them.

		Parameters
		----------
		dm : dm_SEAL
			The DM being used to apply modes.
		modes_number : int
			The number of modes to check linearity for.
		lim : float, default
			The lower and upper limits on each mode, in DM units.
		step : float, default
			The increment used when sweeping through each mode, in DM units.
		plot : bool, default
			Whether or not to plot the result.

		Returns
		-------
		amp_range : np.array [nAmps]
			The amplitudes used.
		responses : np.array [nModes, nAmps, nWFS]
			The full responses of the WFS to each mode.
		"""
		if recalibrate:
			self.calibrate(dm, amp_calib=amp_calib, modes_number=modes_number, lim=lim) # keep the interaction matrix up to date
		else:
			modes_number = self.intMat.shape[1]
		if modes_number is None:
			nModes = dm.Z2C.shape[1]
		else:
			nModes = modes_number
		amp_range = np.arange(-lim, lim+2*np.finfo(float).eps, step)
		ref_image = self.getImage()
		ref_command = self.img2cmd(ref_image)
		responses = np.zeros((nModes, len(amp_range), len(ref_command)))
		# interpretation: responses[i, j, k] contains the response in mode k to an input in mode i of amplitude amp_range[j].
		# The range on i is set by the number of modes the user requests.
		# The range on j is set by the limit and step size the user requests.
		# The range on k is inherent to the WFS.
		for i in range(2, nModes+2):
			for (j, amp) in enumerate(tqdm(amp_range)):
				dm.pokeZernike(amp, i)
				responses[i - 2, j, :] = self.img2cmd(self.getImage())

		if plot:
			self.plot_linearity(amp_range, responses, dm_name=dm.model)

		return amp_range, responses

	def plot_linearity(self, amp_range, responses, dm_name=None):
		nModes = responses.shape[0]
		nrows = 3 
		ncols = int(np.ceil(nModes / nrows))
		lim = (np.min(amp_range), np.max(amp_range))
		_, axs = plt.subplots(nrows, ncols, figsize=(8, 8 * nrows / ncols), sharex=True)
		for k in range(responses.shape[2]):
			ax = axs[k % nrows, k // nrows]
			ax.set_title(f"Mode {k + 2}")
			ax.set_xlim(lim)
			# ax.set_ylim(lim)
			ax.plot(amp_range, amp_range, 'k--')

		for i in range(nModes):
			for k in range(responses.shape[2]):
				alpha = 1 if i == k else 0.1
				axs[k % nrows, k // nrows].plot(amp_range, responses[i,:,k], alpha=alpha)

		title = f"{self.model} linearity curves"
		if dm_name is not None:
			title += f" with {dm_name}"
		plt.suptitle(title)
		plt.show()

	def compute_cmdMat(self,thres=1/30,intMat=None):
		"""
		Compute pseudo inverse of the interaction matrix.

			Parameters:
				threshold (float - optional): conditionning for pseudo-inverse computation.
		"""
		if intMat is None:
			intMat = self.intMat
		self.cmdMat = np.linalg.pinv(intMat,thres)
		
	def load_pokeMatrix(self,pokeMatrix,thres_svd = None):
		"""
		Load a poke matrix computed through calibration process, and compute its pseudo-inverse.
		It is useful to project reconstructed phase onto DM actuators.(This poke matrix is therefore associated with a diven dm.)

			Parameters:
				pokeMatrix (np.array): poke phase in WFS space
				calib (bool): launching or not the synthetic calibration
		
		"""
		self.mode2phase = pokeMatrix
		if thres_svd is None:
			thres = 1/30
		else:
			thres = thres_svd
		self.phase2mode = np.linalg.pinv(self.mode2phase,thres)
		   
#%% ====================================================================
#   ============== PYRAMID WAVEFRONT SENSOR OBJECT =====================
#   ====================================================================

class pyramidWFS(WavefrontSensor):
	"""
	This is a class to use the 4-sided PWFS

	Attributes
	----------
	pupil_tel: array
		pupil intensities
	pupil_footprint: array
		pupil footprint (0 and 1)
	pupil_pad: array
		shannon-padded pupil
	pupil_pad_footprint: array
		shannon-padded pupil footprint
	cam: cameras Object
		camera used for this sensor
	nPx: int
		pupil resolution
	pad: int
		integer to pad array to shannon
	mask_phase: array
		pyramid mask in phase
	mask: array
		pyramid mask (exp(1j*mask_phase))
	nIterRec: int
		number of iterations for GS algorithm
	nonLinRec: bool
		choose if using non-linear reconstructor while reconstructing DM commands - 1 by default
	doUnwrap: bool
		choose if reconstructed phase is unwrapped - 0 by default
	"""

	def __init__(self,pupil_tel,position_pup,shannon,wavelength,cam = 0):  
		""" CONSTRUCTOR """
		self.model = 'PWFS'
		self.modal = 'zonal'
		# ------------- SEAL PWFS CAMERA PARAMETERS -------------
		if cam == 0:
			self.crop = 0
		else:
			self.crop = int(360/cam.binning)
		# ---------- Pupil and resolution -----------------
		pupil_tel[pupil_tel<0] = 0
		self.pupil = pupil_tel/np.sum(pupil_tel)
		self.wavelength = wavelength # useful only for DIPSLAYING phase in NANOMETERS
		self.pupil_footprint = np.copy(self.pupil)
		self.pupil_footprint[self.pupil_footprint>0] = 1
		self.nPx = self.pupil.shape[0] # Resolution in our pupil
		self.shannon = shannon # Shannon sampling : 1 = 2px per lambda/D - Recommended parameter: shannon = 2
		self.pad = int((2*self.shannon-1)*self.nPx/2) # padding in pupil plane to reach Shannon
		self.pupil_pad = np.pad(self.pupil,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		self.pupil_pad_footprint = np.copy(self.pupil_pad)
		self.pupil_pad_footprint[self.pupil_pad_footprint>0] = 1
		self.modu = 0 # no modulation by default
		# ----- Non linear Reconstructor by default ----
		self.nonLinRec = 1
		self.nIterRec = 5
		self.doUnwrap = 0
		self.startPhase = np.zeros((self.nPx,self.nPx))
		# --- Camera for the WFS --
		if cam == 0: # case for simulation only
			self.cam = []
			self.nPx_img = 2*self.shannon*self.nPx
		else:
			self.cam = cam
			self.nPx_img = int(self.cam.nPx)
		self.output_size = self.nPx_img ** 2
		# --- data for display ----
		self.img_simulated = [] # Simulated image from phase estimation - to compare with true data
		self.phase_rec = []
		self.phase_rec_unwrap = []
		self.opd_rec = []
		self.img = [] # image
		self.stopping_criteria = 0.01
		# ------- Define MASK SHAPE from position_pup -------
		self.position_pup = position_pup
		#self.offset_angle =np.ones((8,1))
		self.offset_angle = np.array([0.96736842, 1., 0.95684211,0.96947368, 0.99473684,1., 1.01157895,0.94])
		mask = self.build_mask(self.position_pup)
		self.mask_phase = mask
		self.mask = np.exp(1j*mask)
		# Reference intensities
		self.img0 = self.cropImg(self.getImageSimu(np.zeros((self.nPx,self.nPx))))

	################################ BUILD PYRAMIDAL MASK ####################### 

	def changeMask(self,mask_new):
		self.mask_phase = mask_new
		self.mask = np.exp(1j*self.mask_phase)
		# Reference intensities
		self.img0 = self.cropImg(self.getImageSimu(np.zeros((self.nPx,self.nPx))))

	def build_mask(self,position_pup):
		""" Take pupil position and build mask """
		# position_pup in percent of Diameter along X and Y axis
		# Angle for all faces
		l = np.linspace(-int(self.nPx*self.shannon),int(self.nPx*self.shannon)-1,int(2*self.nPx*self.shannon))
		[tip_x,tilt_y] = np.meshgrid(l,l)
		# left top pupil -----
		x_lt = position_pup[0]*self.offset_angle[0]
		y_lt = position_pup[1]*self.offset_angle[1]
		angle_lt = tip_x*np.pi*x_lt/self.shannon+tilt_y*np.pi*y_lt/self.shannon
		# left bottom pupil -----
		x_lb = position_pup[2]*self.offset_angle[2]
		y_lb = position_pup[3]*self.offset_angle[3]
		angle_lb = tip_x*np.pi*x_lb/self.shannon+tilt_y*np.pi*y_lb/self.shannon
		# right top pupil -----
		x_rt = position_pup[4]*self.offset_angle[4]
		y_rt = position_pup[5]*self.offset_angle[5]
		angle_rt = tip_x*np.pi*x_rt/self.shannon+tilt_y*np.pi*y_rt/self.shannon
		# right bottom pupil -----
		x_rb = position_pup[6]*self.offset_angle[6]
		y_rb = position_pup[7]*self.offset_angle[7]
		angle_rb = tip_x*np.pi*x_rb/self.shannon+tilt_y*np.pi*y_rb/self.shannon
		# Create mask
		mask = np.zeros((int(2*self.nPx*self.shannon),int(2*self.nPx*self.shannon)))
		mask[0:int(self.nPx*self.shannon),0:int(self.nPx*self.shannon)] = angle_rt[0:int(self.nPx*self.shannon),0:int(self.nPx*self.shannon)]
		mask[0:int(self.nPx*self.shannon),int(self.nPx*self.shannon):] = angle_lt[0:int(self.nPx*self.shannon),int(self.nPx*self.shannon):]
		mask[int(self.nPx*self.shannon):,0:int(self.nPx*self.shannon)] = angle_rb[int(self.nPx*self.shannon):,0:int(self.nPx*self.shannon)]
		mask[int(self.nPx*self.shannon):,int(self.nPx*self.shannon):] = angle_lb[int(self.nPx*self.shannon):,int(self.nPx*self.shannon):]
		return mask

	################################ CALIBRATION PROCESS #######################  
	

	def calibrateSimu(self,mode2phase=None,phaseRef=None):
		""" 
		Compute synthetic interaction matrix.

			Parameters:
				mode2phase (np.array): mode to calibrate. By default, it is the pokeMatrix that was loaded.
		"""
		if not(mode2phase is None):
			self.modal = 'zernike'
			self.mode2phase = mode2phase
			self.phase2mode = np.linalg.pinv(self.mode2phase,1/30)
		else:
			self.modal = 'zonal'
		if phaseRef is None:
			phaseRef = np.zeros((self.nPx,self.nPx))
		# -- Linear Calibration -----
		amp_calib = 0.00001
		self.intMat_simu = np.zeros((self.nPx_img*self.nPx_img,self.mode2phase.shape[1]))
		for k in trange(self.mode2phase.shape[1]):
				poke_calib = self.mode2phase[:,k].reshape(int(np.sqrt(self.mode2phase.shape[0])),int(np.sqrt(self.mode2phase.shape[0])))
				# --------- PUSH --------
				I_push = self.cropImg(self.getImageSimu(phaseRef+amp_calib*poke_calib))
				I_push = I_push/np.sum(I_push) # normalisation
				# --------- PULL --------
				I_pull = self.cropImg(self.getImageSimu(phaseRef-amp_calib*poke_calib))
				I_pull = I_pull/np.sum(I_pull) # normalisation
				# -------- Push-pull ------
				s = (I_push-I_pull)/(2*amp_calib)
				self.intMat_simu[:,k] = s.ravel()
		self.compute_cmdMat(self.intMat_simu)

	def load_intMat(self,intMat,dm,modal='zonal'):
		""" 
		Load true interaction matrix already computed before.
		Precise if it is a modal or a zonal matrix (zonal by default).

			Parameters:
				intMat (np.array): Interaction Matrix to be loaded
				dm (DM_seal object): DM associated with this interaction matrix
				modal (bool): if it is modal or not (=0 by default)
		"""
		self.modal=modal
		self.intMat = intMat
		self.compute_cmdMat(self.intMat)


	################################ NOISE and SENSITIVITY #######################
	 
	def noisePropag(self,sigma_ron,sigma_dark,Nph):
		""" Noise propagation model in action """
		# Uniform noise
		self.S_uniform = np.sqrt(np.diag(np.dot(np.transpose(self.intMat),self.intMat)))
		# Photon Noise
		I0 = np.transpose(matlib.repmat(self.img0.ravel(),self.intMat.shape[1],1))
		D = self.intMat/np.sqrt(I0)
		self.S_photon = np.sqrt(np.diag(np.dot(np.transpose(D),D)))        
		# ---- Compute noise propagation for all modes ----
		# RON
		sigma_phi_ron = sigma_ron/(self.S_uniform*Nph)
		# Dark
		sigma_phi_dark = sigma_dark/(self.S_uniform*Nph)
		# photon noise
		sigma_phi_photon = 1/(self.S_photon*np.sqrt(Nph))
		# Sum on all modes
		sigma_phi = np.sqrt(sigma_phi_ron**2+sigma_phi_dark**2+sigma_phi_photon**2)
		#sigma_phi = sum(sigma_phi)/np.sqrt(self.intMat.shape[1])
		return sigma_phi

	################################ PROPAGATION ####################### 
	 
	def propag(self,phi,psf_img=None):
		""" PROPAGATION of the EM field """
		phi_pad =  np.pad(phi,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		# To first focal plane
		amp = np.sqrt(self.pupil_pad) # amplitude = sqrt(intensities)
		Psi_FP = fft_centre(amp*np.exp(1j*phi_pad))
		# Multiply by Zernike Phase mask
		if psf_img is None:
			Psi_FP = self.mask*Psi_FP
		else:
			Psi_FP = self.mask*np.sqrt(psf_img)*np.exp(1j*np.angle(Psi_FP))
		# Back to pupil plane
		Psi_PP = ifft_centre(Psi_FP)
		return Psi_PP


	def backPropag(self,amp,phi,psf_img=None):
		""" BACKWARD PROPAGATION of the EM field for GS algorithm """ 
		# To first focal plane
		Psi_FP = fft_centre(amp*np.exp(1j*phi))
		# Multiply by conjugate of Zernike Phase mask
		if psf_img is None:
			Psi_FP = np.conj(self.mask)*Psi_FP
		else:
			Psi_FP = np.conj(self.mask)*np.sqrt(psf_img)*np.exp(1j*np.angle(Psi_FP))
		# Back to pupil plane
		Psi_PP = ifft_centre(Psi_FP)
		
		return Psi_PP

	def getImage(self):
		""" Record True image """
		img = self.cam.get()
		img[img<0]=0
		img = self.roi(img)
		img = img/np.sum(img)
		return img
	
	def roi(self,img):
		img = img[:,self.crop:-self.crop] # Square image for PWFS
		return img
	
	def getImageSimu(self,phi):
		""" Simulation of Pyramid WFS image - include a modulated case if needed """
		# ========= Non-modulated case =========
		if self.modu == 0:
			Psi_PP = self.propag(phi)
			# Intensities
			img = np.abs(Psi_PP)**2
			img = img/np.sum(img)
		# ========= Modulated case =========
		else:
			img = np.zeros((self.shannon*self.nPx*2,self.shannon*self.nPx*2))
			for k in range(0,self.TTmodu.shape[2]):
				Psi_PP = self.propag(phi+self.TTmodu[:,:,k])
				# Intensities
				img_modu = np.abs(Psi_PP)**2
				img = img+img_modu	
			img = img/np.sum(img)
		return img

	def getImageSimuMultiLambda(self,phi,lambda_list):
		""" Simulation of BROADBAND Pyramid WFS image - include a modulated case if needed """
		img = np.zeros((self.shannon*self.nPx*2,self.shannon*self.nPx*2))
		for l in lambda_list:
			phi_l = phi*self.wavelength/l # scale phase to right value
			img = img + self.getImageSimu(phi_l)
		img = img/np.sum(img)
		return img

	def setModu(self,modu):
		if modu != 0:
			self.modu = modu
			# ======== Create modulation ring ========
			w = np.zeros((self.shannon*self.nPx*2,self.shannon*self.nPx*2))
			Rmod_px = 2*self.shannon*self.modu
			for i in range(0,self.shannon*self.nPx*2):
				for j in range(0,self.shannon*self.nPx*2):
					if np.sqrt((i-(self.shannon*self.nPx*2-1)/2)**2+(j-(self.shannon*self.nPx*2-1)/2)**2)< Rmod_px+1 and np.sqrt((i-(self.shannon*self.nPx*2-1)/2)**2+(j-(self.shannon*self.nPx*2-1)/2)**2)>=Rmod_px:
						w[i,j]=1
			self.w = w/sum(sum(w))
			# ===== Create Cube with all modulation tip-tilt
			TTmodu = None
			l = np.linspace(-self.shannon*self.nPx,self.shannon*self.nPx,self.shannon*self.nPx*2)
			[xx,yy] = np.meshgrid(l,l)
			for i in range(0,self.shannon*self.nPx*2):
				for j in range(0,self.shannon*self.nPx*2):
					if self.w[i,j]!=0:
						ls = 2*np.pi/(self.shannon*self.nPx*2)*((i-(self.shannon*self.nPx*2-1)/2)*xx+(j-(self.shannon*self.nPx*2-1)/2)*yy)
						ls = ls[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)] 
						if TTmodu is None:
							TTmodu = ls
						else:
							TTmodu = np.dstack([TTmodu,ls])
			self.TTmodu = TTmodu
		# ==== Update Reference ========
		self.img0 = self.cropImg(self.getImageSimu(np.zeros((self.nPx,self.nPx))))

	def reconLinear(self,img):
		""" Linear reconstructor using synthetic interaction matrix """
		self.img = img
		dI = img.ravel()/np.sum(img)-self.img0.ravel()/np.sum(self.img0) # reduced intensities
		cmd = np.dot(self.cmdMat,dI)
		self.phase_rec = np.dot(self.mode2phase,cmd)
		self.phase_rec = np.reshape(self.phase_rec,(self.nPx,self.nPx))
		self.img_simulated = self.cropImg(self.getImageSimu(self.phase_rec))
		return cmd

	def reconNonLinear(self,img,nIterRec=None,psf_img=None,verbose=True):
		""" Non-linear reconstructor using GS algorithm
		img: PWFS recorded image (dark removed)
		nIter: number of iteration for the reconstructor
		"""
		if nIterRec is None:
			nIterRec = self.nIterRec
		img = np.copy(img)
		img[img<0] = 0 #killing negative values
		# Pad image if true image
		if img.shape[0] < 2*self.nPx*self.shannon:
				img = np.pad(img,((int(self.nPx*self.shannon-self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2)),(int(self.nPx*self.shannon-self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2))), 'constant') # padded pupil
		#  ========= GS algortihm to reconstruct phase ==========
		# --- 0 point for phase in detector plane ----
		Psi_0 = self.propag(self.startPhase,psf_img)
		phi_0 = np.angle(Psi_0)
		# --- 0 point for amplitude in detector plane ---
		frame = np.copy(img)
		amp_0 = np.sqrt(frame) # SQRT because img is the intensity
		# --- First BACK PROPAGATION ----
		Psi_p = self.backPropag(amp_0,phi_0,psf_img)
		# First phase estimate
		phi_k = np.angle(Psi_p)
		phi_k = phi_k*self.pupil_pad_footprint
		phi_k[np.isnan(phi_k)] = 0 # avoid NaN values
		phi_k = phi_k[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)] 
		err_k_previous = float('inf') # for stopping criteria
		for k in range(nIterRec):
			if verbose:
				print('GS algorithm iteration number: ',k)
			# ---- Direct propagation ----
			Psi_d_k = self.propag(phi_k,psf_img)
			phi_d_k = np.angle(Psi_d_k) # record phase in WFS camera plane
			# ---- BACK PROPAGATION ----
			Psi_p = self.backPropag(amp_0,phi_d_k,psf_img)
			phi_k = np.angle(Psi_p)
			phi_k = phi_k*self.pupil_pad_footprint
			phi_k[np.isnan(phi_k)] = 0 # avoid NaN values
			phi_k = phi_k[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]
			# STOPPING CRITERIA -----------
			#err_k = error_rms(image_k,img_red)
		# ----- Record last phase --------
		phi = phi_k
		self.img_simulated = self.cropImg(self.getImageSimu(phi_k))
		self.img = img
		self.phase_rec = phi
		self.opd_rec = self.phase_rec*self.wavelength/2*np.pi
		self.phase_rec_unwrap = phase_unwrap(phi)*self.pupil_footprint
			 

	def getPSF(self,phi = None):
		""" Get PSF from estimated phase """
		if phi is None:
				phi = self.phase_rec
		phi_pad =  np.pad(phi,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		# ---- PROPAGATION of the EM field ----
		# To first focal plane
		amp = np.sqrt(self.pupil_pad) # amplitude = sqrt(intensities)
		Psi_FP = fft_centre(amp*np.exp(1j*phi_pad))
		psf = np.abs(Psi_FP)**2
		return psf

	def cropImg(self,img):
		""" crop Image to have same size as true image """
		img = img[int(self.nPx*self.shannon-self.nPx_img/2):int(self.nPx*self.shannon+self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2):int(self.nPx*self.shannon+self.nPx_img/2)]
		return img

	################################ DM Command from images #######################
	def img2cmd(self,img,verbose=True):
		""" Full reconstruction of the signal: image to DM commands """
		if self.nonLinRec == 0:
			self.img = img
			cmd = self.reconLinear(img)
		else:
			self.reconNonLinear(img,self.nIterRec,verbose=verbose)
			if self.doUnwrap == 0:
				cmd = np.dot(self.phase2mode,self.phase_rec.ravel())
				self.opd_rec = self.phase_rec*self.wavelength/2*np.pi
			else:
				cmd = np.dot(self.phase2mode,self.phase_rec_unwrap.ravel())
				self.opd_rec = self.phase_rec_unwrap*self.wavelength/2*np.pi			
		return cmd


#%% ====================================================================
#   ============== ZERNIKE WAVEFRONT SENSOR OBJECT =====================
#   ====================================================================


class zernikeWFS(WavefrontSensor):
	"""
	This is a class to use the Zernike WFS
	Class went rhough major modification after OWL program 2023.
	- Using new way to simulate ZWFS (Soummer 2007)
	- Enabling to switch between vortex and scalar mask

	Attributes
	----------
	pupil_tel: array
		pupil intensities
	pupil_footprint: array
		pupil footprint (0 and 1)
	pupil_pad: array
		shannon-padded pupil
	pupil_pad_footprint: array
		shannon-padded pupil footprint
	cam: cameras Object
		camera used for this sensor
	nPx: int
		pupil resolution
	pad: int
		integer to pad array to shannon
	mask_phase: array
		pyramid mask in phase
	mask: array
		pyramid mask (exp(1j*mask_phase))
	nIterRec: int
		number of iterations for GS algorithm
	nonLinRec: bool
		choose if using non-linear reconstructor while reconstructing DM commands - 1 by default
	doUnwrap: bool
		choose if reconstructed phase is unwrapped - 0 by default
	"""

	def __init__(self,pupil_tel,position_pups,diameter,depth,dimple_pixels,wavelength,cam = 0):  
		""" CONSTRUCTOR """
		self.modal = 'zonal'
		self.factorSize = 1 # for GS algorithm

		if pupil_tel.ndim == 2:
			self.model = 'scalar ZWFS'
			# ------------- SEAL ZWFS CAMERA PARAMETERS -------------
			self.center_x = position_pups[0]
			self.center_y = position_pups[1]
			self.nPx_img = position_pups[2]
			self.output_size = self.nPx_img ** 2
			# ------------- DEFINE CAMERA OR SIMULATION MODE -------------
			if cam == 0: # case for simulation only
				self.cam = []
			else:
				self.cam = cam
				
				
			################# PUPIL PLANE ###################
			# ---------- Pupil and resolution -----------------
			self.nPx = pupil_tel.shape[1] # Resolution in our pupil
			self.wavelength = wavelength # useful only for DIPSLAYING phase in NANOMETERS
			self.pad = int((self.nPx_img-self.nPx)/2) # padding in pupil plane to match measurements
			self.pupil_extension = self.nPx_img/self.nPx
			# ---- for GS algorithm ----
			self.pad_large = int((self.factorSize*self.nPx_img-self.nPx)/2)
			# ---------- Single pupil -----------
			self.pupil = pupil_tel#/np.sum(pupil_tel) # Pupil as recorded when off dimple (INTENSITIES)
			self.pupil[self.pupil<0] = 0
			self.pupil_footprint = np.copy(self.pupil)
			self.pupil_footprint[self.pupil_footprint >0] = 1
			self.pupil_pad = np.pad(self.pupil,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
			self.pupil_pad_footprint = np.copy(self.pupil_pad).astype(bool)
			self.pupil_pad_footprint[self.pupil_pad_footprint>0] = 1
			################# FOCAL PLANE ###################
			self.dimple_pixels = dimple_pixels #pixels in dimple diameter
			self.nPx_mask = self.dimple_pixels # define array 2 times +1 bigger (to have odd number)
			# SCALAR MASK
			self.diameter = diameter # Diameter in lambda/D at lambda = wavelength
			self.depth = depth # Depth in phase shift (radians) at lambda = wavelength
			self.fourier_sampling = self.dimple_pixels/self.diameter
			self.fourier_extension = (self.nPx_mask-1)/self.fourier_sampling
			self.mask = self.makeMask(self.depth)

		elif pupil_tel.ndim == 3:
			self.model = 'vector ZWFS'
			# ------------- SEAL ZWFS CAMERA PARAMETERS -------------
			self.left_x = position_pups[0]
			self.left_y = position_pups[1]
			self.right_x = position_pups[2]
			self.right_y = position_pups[3]
			self.nPx_img_x = position_pups[4]
			self.nPx_img_y = position_pups[5]
			# ------------- DEFINE CAMERA OR SIMULATION MODE -------------
			if cam == 0: # case for simulation only
				self.cam = []
				self.nPx_img = self.nPx_img_y
			else:
				self.cam = cam
				self.nPx_img = self.nPx_img_y

			self.output_size = self.nPx_img ** 2

			################# PUPIL PLANE ###################
			# ---------- Pupil and resolution -----------------
			self.nPx = pupil_tel.shape[1] # Resolution in our pupil
			self.wavelength = wavelength # useful only for DIPSLAYING phase in NANOMETERS
			self.pad = int((self.nPx_img-self.nPx)/2) # padding in pupil plane to match measurements
			self.pupil_extension = self.nPx_img/self.nPx
			# ---- for GS algorithm ----
			self.pad_large = int((self.factorSize*self.nPx_img-self.nPx)/2)

			# ---------- LEFT pupil -----------
			self.pupil_left = pupil_tel[0,:,:]#/np.sum(pupil_tel[0,:,:]) # Pupil as recorded when off dimple (INTENSITIES)
			self.pupil_left[self.pupil_left<0] = 0
			self.pupil_footprint_left = np.copy(self.pupil_left)
			self.pupil_footprint_left [self.pupil_footprint_left >0] = 1
			self.pupil_pad_left = np.pad(self.pupil_left,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
			self.pupil_pad_footprint_left = np.copy(self.pupil_pad_left).astype(bool)
			self.pupil_pad_footprint_left[self.pupil_pad_footprint_left>0] = 1
			# ---------- RIGHT pupil -----------
			self.pupil_right = pupil_tel[1,:,:]#/np.sum(pupil_tel[1,:,:]) # Pupil as recorded when off dimple (INTENSITIES)
			self.pupil_right[self.pupil_right<0] = 0
			self.pupil_footprint_right = np.copy(self.pupil_right)
			self.pupil_footprint_right [self.pupil_footprint_right >0] = 1
			self.pupil_pad_right = np.pad(self.pupil_right,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
			self.pupil_pad_footprint_right = np.copy(self.pupil_pad_right).astype(bool)
			self.pupil_pad_footprint_right[self.pupil_pad_footprint_right>0] = 1

			################# FOCAL PLANE ###################
			self.dimple_pixels = dimple_pixels #pixels in dimple diameter
			self.nPx_mask = self.dimple_pixels # define array 2 times +1 bigger (to have odd number)
			# LEFT MASK
			self.diameter_left = diameter[0] # Diameter in lambda/D at lambda = wavelength
			self.depth_left = depth[0] # Depth in phase shift (radians) at lambda = wavelength
			self.fourier_sampling_left = self.dimple_pixels/self.diameter_left
			self.fourier_extension_left = (self.nPx_mask-1)/self.fourier_sampling_left
			self.mask_left = self.makeMask(self.depth_left)
			#self.switchMask('left') # left mask by default
			# RIGHT MASK
			self.diameter_right = diameter[1] # Diameter in lambda/D at lambda = wavelength
			self.depth_right = depth[1] # Depth in phase shift (radians) at lambda = wavelength
			self.fourier_sampling_right = self.dimple_pixels/self.diameter_right
			self.fourier_extension_right = (self.nPx_mask-1)/self.fourier_sampling_right
			self.mask_right = self.makeMask(self.depth_right)
			self.switchMask('right') # right mask by default
			
		################# RECONSTRUCTION ###################

		# --- Stopping criteria ----
		self.stopping_criteria = 0 # error difference to stop iterative loop in reconstruction
		# --- reconstruction parameters ----
		self.nonLinRec = 1
		self.algo = 'JPL'
		self.startPhase = np.zeros((self.nPx,self.nPx))
		self.doUnwrap = 0
		# --- data for reconstruction----
		self.img_simulated = [] # Simulated image from phase estimation - to compare with true data
		self.nIterRec = 10
		self.phase_rec = []
		self.phase_rec_unwrap = []
		self.opd_rec = []
		self.img = [] # image
		# Reference intensities
		self.img0 = self.getImageSimu(np.zeros((self.nPx,self.nPx)))

	def switchMask(self,pupil_choice = 'left'):
		if self.model == 'scalar ZWFS':
			print("You are using a SCALAR ZWFS, you can't switch mask")
		elif self.model == 'vector ZWFS':
			self.pupilRec = pupil_choice
			if pupil_choice == 'left':
				self.mask = self.mask_left
				self.diameter = self.diameter_left
				self.depth = self.depth_left
				self.pupil = self.pupil_left
				self.pupil_footprint = self.pupil_footprint_left
				self.pupil_pad = self.pupil_pad_left
				self.pupil_pad_footprint = self.pupil_pad_footprint_left
				self.fourier_sampling = self.fourier_sampling_left 
				self.fourier_extension = self.fourier_extension_left
			elif pupil_choice == 'right':
				self.mask = self.mask_right
				self.diameter = self.diameter_right
				self.depth = self.depth_right
				self.pupil = self.pupil_right
				self.pupil_footprint = self.pupil_footprint_right
				self.pupil_pad = self.pupil_pad_right
				self.pupil_pad_footprint = self.pupil_pad_footprint_right
				self.fourier_sampling = self.fourier_sampling_right 
				self.fourier_extension = self.fourier_extension_right 

	def makeMask(self,depth):
		# -------- BUILDING FOCAL PLANE MASK --------
		l = np.linspace(-(self.nPx_mask-1)/2,(self.nPx_mask-1)/2,self.nPx_mask)
		[xx,yy] = np.meshgrid(l,l)
		# Polar coordinates
		r = np.sqrt(xx**2+yy**2)
		mask = np.ones((self.nPx_mask,self.nPx_mask),dtype=complex)
		mask[r<self.dimple_pixels/2] = np.exp(1j*depth)
		return mask
	
	def changeFactorSize(self,factor):
		# ---- for GS algorithm ----
		self.factorSize = factor # for GS algorithm
		self.pad_large = int((self.factorSize*self.nPx_img-self.nPx)/2)

	################################ CALIBRATION PROCESS #######################  
	

	
	def calibrateSimu(self,mode2phase=None,phaseRef=None):
		""" 
		Compute synthetic interaction matrix.

			Parameters:
				mode2phase (np.array): mode to calibrate. By default, it is the pokeMatrix that was loaded.
		"""
		if not(mode2phase is None):
			self.modal = 1
			self.mode2phase = mode2phase
			self.phase2mode = np.linalg.pinv(self.mode2phase,1/30)
		else:
			self.modal = 0
		if phaseRef is None:
			phaseRef = np.zeros((self.nPx,self.nPx))
		# -- Linear Calibration -----
		amp_calib = 0.00001
		self.intMat = np.zeros((self.nPx_img*self.nPx_img,self.mode2phase.shape[1]))
		for k in trange(self.mode2phase.shape[1]):
				poke_calib = self.mode2phase[:,k].reshape(int(np.sqrt(self.mode2phase.shape[0])),int(np.sqrt(self.mode2phase.shape[0])))
				# --------- PUSH --------
				I_push = self.getImageSimu(phaseRef+amp_calib*poke_calib)
				I_push = I_push[:,0:int(self.nPx_img)]
				I_push = I_push/np.sum(I_push) # normalisation
				# --------- PULL --------
				I_pull = self.getImageSimu(phaseRef-amp_calib*poke_calib)
				I_pull = I_pull[:,0:int(self.nPx_img)]
				I_pull = I_pull/np.sum(I_pull) # normalisation
				# -------- Push-pull ------
				s = (I_push-I_pull)/(2*amp_calib)
				self.intMat[:,k] = s.ravel()
		self.compute_cmdMat()

	def load_intMat(self,intMat,dm,modal=0):
		""" 
		Load true interaction matrix already computed before.
		Precise if it is a modal or a zonal matrix (zonal by default).

			Parameters:
				intMat (np.array): Interaction Matrix to be loaded
				dm (DM_seal object): DM associated with this interaction matrix
				modal (bool): if it is modal or not (=0 by default)
		"""
		self.modal=modal
		self.intMat = intMat
		self.compute_cmdMat(self.intMat)

	################################ PROPAGATION ####################### 

	def propag(self,phi,amp=None,psf_img=None):
		""" PROPAGATION of the EM field """
		# ---- PROPAGATION of the EM field ----
		# To first focal plane
		if amp is None:
			amp = np.sqrt(self.pupil) # amplitude = sqrt(intensities)
			E_in = np.pad(amp*np.exp(1j*phi),((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		elif amp.shape[0]>self.nPx:
			E_in = amp*np.pad(np.exp(1j*phi),((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		else:
			E_in = np.pad(amp*np.exp(1j*phi),((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		E_fp = MFT(E_in,self.fourier_sampling,self.fourier_extension,self.nPx)
		# Multiply by (1 - Zernike Phase mask)
		E_fp = (1-self.mask)*E_fp
		# Back to pupil plane
		E_ref = iMFT(E_fp,self.nPx,self.nPx_img/self.nPx,self.fourier_sampling)
		E_out = E_in-E_ref
		return E_out

	def propagLarge(self,phi,amp=None):
		""" PROPAGATION of the EM field """
		# ---- PROPAGATION of the EM field ----
		# To first focal plane
		if amp is None:
			amp = np.sqrt(self.pupil) # amplitude = sqrt(intensities)
			E_in = np.pad(amp*np.exp(1j*phi),((self.pad_large,self.pad_large),(self.pad_large,self.pad_large)), 'constant') # padded pupil
		elif amp.shape[0]>self.nPx:
			E_in = amp*np.pad(np.exp(1j*phi),((self.pad_large,self.pad_large),(self.pad,self.pad_large)), 'constant') # padded pupil
		else:
			E_in = np.pad(amp*np.exp(1j*phi),((self.pad_large,self.pad_large),(self.pad_large,self.pad_large)), 'constant') # padded pupil
		E_fp = MFT(E_in,self.fourier_sampling,self.fourier_extension,self.nPx)
		# Multiply by (1 - Zernike Phase mask)
		E_fp = (1-self.mask)*E_fp
		# Back to pupil plane
		E_ref = iMFT(E_fp,self.nPx,self.factorSize*self.nPx_img/self.nPx,self.fourier_sampling)
		E_out = E_in-E_ref
		return E_out

	def backPropag(self,amp,phi,psf_img=None):
		""" BACKWARD PROPAGATION of the EM field for GS algorithm """
		# ---- PROPAGATION of the EM field ----
		# To first focal plane
		E_out = amp*np.exp(1j*phi)
		E_fp = MFT(E_out,self.fourier_sampling,self.fourier_extension,self.nPx)
		# Multiply by Zernike Phase mask
		E_fp = (1-np.conj(self.mask))*E_fp # Conjugate of the mask
		# Back to pupil plane
		E_ref = iMFT(E_fp,self.nPx,self.nPx_img/self.nPx,self.fourier_sampling)
		E_in = E_out-E_ref
		return E_in

	def backPropagLarge(self,amp,phi):
		""" BACKWARD PROPAGATION of the EM field for GS algorithm """
		# ---- PROPAGATION of the EM field ----
		# To first focal plane
		E_out = amp*np.exp(1j*phi)
		E_fp = MFT(E_out,self.fourier_sampling,self.fourier_extension,self.nPx)
		# Multiply by Zernike Phase mask
		E_fp = (1-np.conj(self.mask))*E_fp # Conjugate of the mask
		# Back to pupil plane
		E_ref = iMFT(E_fp,self.nPx,self.factorSize*self.nPx_img/self.nPx,self.fourier_sampling)
		E_in = E_out-E_ref
		return E_in

	def getImage(self):
		""" Record True image """
		img = self.cam.get()
		#img = img/np.sum(img) # normalization
		img[img<0]=0 #killing negative values
		if self.model == 'scalar ZWFS':
			img = self.roi_scalar(img)
		if self.model == 'vector ZWFS':
			img = self.roi_vector(img)
		return img
	
	def roi_vector(self,img):
		#img = img[int(self.center_y-self.nPx_img_y/2):int(self.center_y+self.nPx_img_y/2),int(self.center_x-self.nPx_img_x/2):int(self.center_x+self.nPx_img_x/2)]
		img_L = img[self.left_y-int(self.nPx_img/2):self.left_y+int(self.nPx_img/2),self.left_x-int(self.nPx_img/2):self.left_x+int(self.nPx_img/2)]
		img_R = img[self.right_y-int(self.nPx_img/2):self.right_y+int(self.nPx_img/2),self.right_x-int(self.nPx_img/2):self.right_x+int(self.nPx_img/2)]
		stacked_img = np.zeros((self.nPx_img_y,self.nPx_img_x))
		stacked_img[:,0:self.nPx_img_y] = img_L#/np.sum(img_L) # normalization
		stacked_img[:,self.nPx_img_y:] = img_R#/np.sum(img_R) # normalization
		return stacked_img
	
	def roi_scalar(self,img):
		#img = img[int(self.center_y-self.nPx_img_y/2):int(self.center_y+self.nPx_img_y/2),int(self.center_x-self.nPx_img_x/2):int(self.center_x+self.nPx_img_x/2)]
		img = img[self.center_y-int(self.nPx_img/2):self.center_y+int(self.nPx_img/2),self.center_x-int(self.nPx_img/2):self.center_x+int(self.nPx_img/2)]
		return img

	def getImageSimu(self,phi):
		""" Simulation of Zernike WFS image """
		if self.model == 'scalar ZWFS':
			Psi_PP = self.propag(phi)
			img = np.abs(Psi_PP)**2
		if self.model == 'vector ZWFS':
			pupilRec = self.pupilRec
			# Intensities - LEFT
			self.switchMask('left')
			Psi_PP = self.propag(phi)
			img_left = np.abs(Psi_PP)**2
			# Intensities - RIGHT
			self.switchMask('right')
			Psi_PP = self.propag(phi)
			img_right = np.abs(Psi_PP)**2
			# Full image
			img = np.concatenate((img_left,img_right),axis = 1)
			# Back to pupilRec
			self.switchMask(pupilRec)
		return img

	def reconLinear(self,img_input):
		""" Linear reconstructor using synthetic interaction matrix """
		# keeping on pupil only
		img = img_input[:,0:int(self.nPx_img)]
		img0 = self.img0[:,0:int(self.nPx_img)]
		# -----
		self.img = img
		dI = img.ravel()/np.sum(img)-img0.ravel()/np.sum(img0) # reduced intensities
		cmd = np.dot(self.cmdMat,dI)
		self.phase_rec = np.dot(self.mode2phase,cmd)
		self.phase_rec = np.reshape(self.phase_rec,(self.nPx,self.nPx))
		return cmd
		#self.img_simulated = self.cropImg(self.getImageSimu(self.phase_rec))

	def reconLinearModel(self,img,nIterRec=None,psf_img=None):
		""" Linear reconstructor
		img: ZWFS recorded image (dark removed)
		nIter: number of iteration for the reconstructor
		"""
		img_r = img[:,0:int(self.nPx_img)]
		# --- First PROPAGATIONS  ----
		E_in = np.sqrt(self.pupil_pad)*np.exp(1j*np.pad(self.startPhase,((self.pad,self.pad),(self.pad,self.pad)), 'constant'))
		E_out = self.propag(self.startPhase)
		E_ref = E_in - E_out

		# --- Reference beam  ----
		I_ref = np.abs(E_ref)**2
		phi_ref = np.angle(E_ref)

		# --- Arcsin  ----
		phi_k = -(img_r-self.pupil_pad-I_ref)/(-2*np.sign(self.depth)*np.sqrt(self.pupil_pad*I_ref)*np.sin(phi_ref))-np.cos(phi_ref)/np.sin(phi_ref)
		phi_k = phi_k[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]
		phi_k = phi_k*self.pupil_footprint
		phi_k[np.isnan(phi_k)] = 0
		self.phase_rec_right = phi_k
		self.phase_rec = phi_k
		
	def reconNonLinear(self,img):
		# Normlasiation image
		img[:,0:int(self.nPx_img)] = img[:,0:int(self.nPx_img)]#/np.sum(img[:,0:int(self.nPx_img)])
		img[:,int(self.nPx_img):] = img[:,int(self.nPx_img):]#/np.sum(img[:,int(self.nPx_img):])
		self.img = img


		# --------- SCALAR -------------
		if self.model == 'scalar ZWFS':
			if self.algo == 'JPL':
				self.reconJPL(img)
			elif self.algo == 'GS':
				self.reconGS(img)

		# --------- VECTOR -------------
		if self.model == 'vector ZWFS':
			# Left pupil -------------
			if self.pupilRec == 'left':
				self.switchMask('left')
				img_r = img[:,0:int(self.nPx_img)]
				if self.algo == 'JPL':
					self.reconJPL(img_r)
				elif self.algo == 'GS':
					self.reconGS(img_r)
			# Right pupil -------------
			elif self.pupilRec == 'right':
				self.switchMask('right')
				img_r = img[:,int(self.nPx_img):]
				if self.algo == 'JPL':
					self.reconJPL(img_r)
				elif self.algo == 'GS':
					self.reconGS(img_r)
			# Both pupils -------------
			elif self.pupilRec == 'both':
				if self.algo == 'JPL':
					self.reconPhaseShiftedIterative(img)
				elif self.algo == 'GS':
					# Left pupil
					self.switchMask('left')
					img_r = img[:,0:int(self.nPx_img)]
					self.reconGS(img_r)
					self.phase_rec_left = np.copy(self.phase_rec)
					# right pupil
					self.switchMask('right')
					img_r = img[:,int(self.nPx_img):]
					self.reconGS(img_r)
					self.phase_rec_right = np.copy(self.phase_rec)
					# combine
					self.reconPhaseShifted()
				self.pupilRec = 'both'

	def reconGS(self,img_r):
		""" Non-linear reconstructor using GS algorithm
		img: ZWFS recorded image (dark removed) with two pupils inside
		nIter: number of iteration for the reconstructor
		"""
		#  ========= GS algortihm to reconstruct phase ==========
		# --- 0 point for phase in detector plane ----
		Psi_0 = self.propagLarge(self.startPhase)
		phi_0 = np.angle(Psi_0)
		amp_0 = np.abs(Psi_0)
		norm = np.sum(np.abs(amp_0[int((self.factorSize-1)*self.nPx_img/2):int((self.factorSize+1)*self.nPx_img/2),int((self.factorSize-1)*self.nPx_img/2):int((self.factorSize+1)*self.nPx_img/2)])**2)
		# --- 0 point for amplitude in detector plane ---
		frame = np.copy(img_r)
		frame = frame/np.sum(frame)*norm
		#amp_0 = np.sqrt(frame)
		amp_0[int((self.factorSize-1)*self.nPx_img/2):int((self.factorSize+1)*self.nPx_img/2),int((self.factorSize-1)*self.nPx_img/2):int((self.factorSize+1)*self.nPx_img/2)]=np.sqrt(frame)# SQRT because img is the intensity
		#fig(amp_0)
		# --- First BACK PROPAGATION ----
		Psi_p = self.backPropagLarge(amp_0,phi_0)
		# keep amplitude
		#amp_k0 = np.abs(Psi_p)#[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)])
		# First phase estimate
		phi_k = np.angle(Psi_p)
		phi_k = phi_k[int(self.factorSize*self.nPx_img/2+1-self.nPx/2-1):int(self.factorSize*self.nPx_img/2+self.nPx/2),int(self.factorSize*self.nPx_img/2+1-self.nPx/2-1):int(self.factorSize*self.nPx_img/2+self.nPx/2)] 
		phi_k = phi_k*self.pupil_footprint
		phi_k[np.isnan(phi_k)] = 0 # avoid NaN values
		err_k_previous = float('inf') # for stopping criteria
		for k in range(0,self.nIterRec):
				# ---- Direct propagation ----
				Psi_d_k = self.propagLarge(phi_k)#,amp_k0)
				amp_d_k = np.abs(Psi_d_k)
				norm = np.sum(np.abs(amp_d_k[int((self.factorSize-1)*self.nPx_img/2):int((self.factorSize+1)*self.nPx_img/2),int((self.factorSize-1)*self.nPx_img/2):int((self.factorSize+1)*self.nPx_img/2)])**2)
				frame = frame/np.sum(frame)*norm
				amp_d_k[int((self.factorSize-1)*self.nPx_img/2):int((self.factorSize+1)*self.nPx_img/2),int((self.factorSize-1)*self.nPx_img/2):int((self.factorSize+1)*self.nPx_img/2)]=np.sqrt(frame)# SQRT because img is the intensity
				phi_d_k = np.angle(Psi_d_k) # record phase in Zernike WFS camera plane
				# ---- BACK PROPAGATION ----
				Psi_p = self.backPropagLarge(amp_d_k,phi_d_k)
				phi_k = np.angle(Psi_p)
				phi_k = phi_k[int(self.factorSize*self.nPx_img/2+1-self.nPx/2-1):int(self.factorSize*self.nPx_img/2+self.nPx/2),int(self.factorSize*self.nPx_img/2+1-self.nPx/2-1):int(self.factorSize*self.nPx_img/2+self.nPx/2)]
				phi_k = phi_k*self.pupil_footprint
				phi_k[np.isnan(phi_k)] = 0 # avoid NaN values
				# STOPPING CRITERIA -----------
				'''
				# Stop if error decrease less than n_p percent
				image_k = self.getImageSimu(phi_k) # Record image of estimated phase through Zernike 
				image_k = image_k[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]
				img_red = img_r[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]
				err_k = error_rms(image_k,img_red)
				if False:#(err_k_previous - err_k)/err_k_previous < self.stopping_criteria: # break if STOPPING CONDITIONS ARE MET
						print('local minimum found')
						print(k) 
						break
				else:
						err_k_previous = err_k
						#phi = phi_k
				'''
				# ---------------------------
				
		# Record last phase
		self.img_simulated = self.getImageSimu(phi_k)
		self.phase_rec = phi_k
		self.opd_rec = self.phase_rec*self.wavelength/(2*np.pi)
		self.phase_rec_unwrap = phase_unwrap(phi_k)*self.pupil_footprint

	def reconGS_old(self,img_r):
		""" Non-linear reconstructor using GS algorithm
		img: ZWFS recorded image (dark removed) with two pupils inside
		nIter: number of iteration for the reconstructor
		"""
		#  ========= GS algortihm to reconstruct phase ==========
		# --- 0 point for phase in detector plane ----
		Psi_0 = self.propag(self.startPhase)
		phi_0 = np.angle(Psi_0)
		# --- 0 point for amplitude in detector plane ---
		frame = np.copy(img_r)
		amp_0 = np.sqrt(frame) # SQRT because img is the intensity
		# --- First BACK PROPAGATION ----
		Psi_p = self.backPropag(amp_0,phi_0)
		# keep amplitude
		#amp_k0 = np.abs(Psi_p)#[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)])
		# First phase estimate
		phi_k = np.angle(Psi_p)
		phi_k = phi_k*self.pupil_pad_footprint
		phi_k[np.isnan(phi_k)] = 0 # avoid NaN values
		phi_k = phi_k[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)] 
		err_k_previous = float('inf') # for stopping criteria
		for k in range(0,self.nIterRec):
				# ---- Direct propagation ----
				Psi_d_k = self.propag(phi_k)#,amp_k0)
				phi_d_k = np.angle(Psi_d_k) # record phase in Zernike WFS camera plane
				# ---- BACK PROPAGATION ----
				Psi_p = self.backPropag(amp_0,phi_d_k)
				phi_k = np.angle(Psi_p)
				phi_k = phi_k*self.pupil_pad_footprint
				phi_k[np.isnan(phi_k)] = 0 # avoid NaN values
				phi_k = phi_k[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]
				# STOPPING CRITERIA -----------
				'''
				# Stop if error decrease less than n_p percent
				image_k = self.getImageSimu(phi_k) # Record image of estimated phase through Zernike 
				image_k = image_k[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]
				img_red = img_r[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]
				err_k = error_rms(image_k,img_red)
				if False:#(err_k_previous - err_k)/err_k_previous < self.stopping_criteria: # break if STOPPING CONDITIONS ARE MET
						print('local minimum found')
						print(k) 
						break
				else:
						err_k_previous = err_k
						#phi = phi_k
				'''
				# ---------------------------
				
		# Record last phase
		self.img_simulated = self.getImageSimu(phi_k)
		self.phase_rec = phi_k
		self.opd_rec = self.phase_rec*self.wavelength/(2*np.pi)
		self.phase_rec_unwrap = phase_unwrap(phi_k)*self.pupil_footprint

	def reconJPL(self,img_r):
		""" Non-linear reconstructor using Iterative algorithm
		img: ZWFS recorded image (dark removed)
		nIter: number of iteration for the reconstructor
		"""
		#  ========= JPL algortihm to reconstruct phase ==========
		# Use a formula based on a interferometric model of the Zernike WFS

		# --- First PROPAGATIONS  ----
		E_in = np.sqrt(self.pupil_pad)*np.exp(1j*np.pad(self.startPhase,((self.pad,self.pad),(self.pad,self.pad)), 'constant'))
		E_out = self.propag(self.startPhase)
		E_ref = E_in - E_out
		norm = np.sum(np.abs(E_out)**2)

		img_r = img_r/np.sum(img_r)*norm
		# --- Reference beam  ----
		I_ref = np.abs(E_ref)**2
		phi_ref = np.angle(E_ref)

		# --- Arcsin  ----
		r = (img_r-self.pupil_pad-I_ref)/(-2*np.sign(self.depth)*np.sqrt(self.pupil_pad*I_ref))
		r = np.clip(r,-1,1)
		t = np.arcsin(r)
		t[np.isnan(t)] = 0
		phi_k = np.pi/2+phi_ref-t

		phi_k = phi_k*self.pupil_pad_footprint
		# Crop array to nPx
		phi_k = phi_k[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]      
		
		# --- LOOP on Reconstructor ----
		for k in range(0,self.nIterRec):
						# --- First PROPAGATIONS  ----
						E_in = np.sqrt(self.pupil_pad)*np.exp(1j*np.pad(phi_k,((self.pad,self.pad),(self.pad,self.pad)), 'constant'))
						E_out = self.propag(phi_k)
						E_ref = E_in - E_out

						# --- Reference beam  ----
						I_ref = np.abs(E_ref)**2
						phi_ref = np.angle(E_ref)

						# --- Arcsin  ----
						r = (img_r-self.pupil_pad-I_ref)/(-2*np.sign(self.depth)*np.sqrt(self.pupil_pad*I_ref))
						r = np.clip(r,-1,1)
						t = np.arcsin(r)
						t[np.isnan(t)] = 0
						phi_k = np.pi/2+phi_ref-t

						phi_k = phi_k*self.pupil_pad_footprint
						# Crop array to nPx
						phi_k = phi_k[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]
						# remove piston
						phi_k = phi_k-np.sum(phi_k*self.pupil_footprint)*self.pupil_footprint/np.sum(self.pupil_footprint)      
						

		# ----- Record last phase --------
		self.img_simulated = np.abs(E_out)**2
		# remove piston
		phi_k = phi_k-np.sum(phi_k)*self.pupil_footprint/np.sum(self.pupil_footprint)
		self.phase_rec = phi_k
		self.opd_rec = self.phase_rec*self.wavelength/(2*np.pi)
		self.phase_rec_unwrap = phase_unwrap(phi_k)*self.pupil_footprint 
		
	def reconPhaseShifted(self):
		"""
		Using data from ZWFS_1 et ZWFS_2 to increase dynamic - Have to be used after using reconstructors for ZWFS_1 and ZWFS_2
		"""
		# ----- ZWFS left -------
		self.switchMask('left')
		# --- First PROPAGATIONS  ----
		E_in = np.sqrt(self.pupil_pad)*np.exp(1j*np.pad(self.phase_rec_left,((self.pad,self.pad),(self.pad,self.pad)), 'constant'))
		E_out = self.propag(self.phase_rec_left)
		E_ref = E_in - E_out
		norm = np.sum(np.abs(E_out)**2)

		#----- Define sinus-------
		I_ref = np.abs(E_ref)**2
		img_r = self.img[:,0:int(self.nPx_img)]/np.sum(self.img[:,0:int(self.nPx_img)])*norm
		ZWFS_sin_left = (img_r-self.pupil_pad-I_ref)/(-2*np.sqrt(self.pupil_pad*I_ref))
		ZWFS_sin_left = np.clip(ZWFS_sin_left,-1,1)
		A_left = -np.cos(np.angle(E_ref)+np.pi/2)
		B_left = np.sin(np.angle(E_ref)+np.pi/2)

		# ----- ZWFS right -------
		self.switchMask('right')
		# --- First PROPAGATIONS  ----
		E_in = np.sqrt(self.pupil_pad)*np.exp(1j*np.pad(self.phase_rec_right,((self.pad,self.pad),(self.pad,self.pad)), 'constant'))
		E_out = self.propag(self.phase_rec_right)
		E_ref = E_in - E_out
		norm = np.sum(np.abs(E_out)**2)

		#----- Define sinus-------
		I_ref = np.abs(E_ref)**2
		img_r = self.img[:,int(self.nPx_img):]/np.sum(self.img[:,int(self.nPx_img):])*norm
		ZWFS_sin_right = (img_r-self.pupil_pad-I_ref)/(-2*np.sqrt(self.pupil_pad*I_ref))
		ZWFS_sin_right = np.clip(ZWFS_sin_right,-1,1)
		A_right = -np.cos(np.angle(E_ref)+np.pi/2)
		B_right = np.sin(np.angle(E_ref)+np.pi/2)

		# ---- LOOP for all element of PHI ----
		phi = np.zeros((self.nPx_img,self.nPx_img))
		self.singular_values = np.zeros((self.nPx_img,self.nPx_img))
		
		for i in range(0,self.nPx_img):
				for j in range(0,self.nPx_img):
						if self.pupil_pad_footprint[i,j] == 1:
							# --- matrix to be inverted
							M = np.array([[A_left[i,j],B_left[i,j]],[A_right[i,j],B_right[i,j]]])
							D = np.array([ZWFS_sin_left[i,j],ZWFS_sin_right[i,j]])
							M_dag = np.linalg.pinv(M)
							# ======== FOR TESTING =========
							U,S,V = np.linalg.svd(M)
							self.singular_values[i,j] = S[0]+S[1]
							# =======================
							# --- sin(phi) / cos(phi) vector
							R = np.dot(M_dag,D)
							phi[i,j] = np.angle(R[1]+1j*R[0])#np.arctan2(R[1],R[0])
		
		phi = phi[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]      



		phi = phi*self.pupil_footprint
		#self.singular_values = self.pupil_footprint*self.singular_values[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]      
		# ----- Record last phase --------
		self.img_simulated = self.getImageSimu(phi)
		self.phase_rec = phi
		self.opd_rec = self.phase_rec*self.wavelength/(2*np.pi)
		self.phase_rec_unwrap = phase_unwrap(phi)*self.pupil_footprint
		
		
	def reconPhaseShiftedIterative(self,img):
		"""
		Using data from ZWFS_1 et ZWFS_2 to increase dynamic - Have to be used after using reconstructors for ZWFS_1 and ZWFS_2
		"""
		
		for k in range(0,self.nIterRec):
			if k == 0:
				phi = self.startPhase
				
			# ----- ZWFS left -------
			self.switchMask('left')
			# --- First PROPAGATIONS  ----
			E_in = np.sqrt(self.pupil_pad)*np.exp(1j*np.pad(phi,((self.pad,self.pad),(self.pad,self.pad)), 'constant'))
			E_out = self.propag(phi)
			E_ref = E_in - E_out
			norm = np.sum(np.abs(E_out)**2)

			#----- Define sinus-------
			I_ref = np.abs(E_ref)**2
			img_r = img[:,0:int(self.nPx_img)]/np.sum(img[:,0:int(self.nPx_img)])*norm
			ZWFS_sin_left = (img_r-self.pupil_pad-I_ref)/(-2*np.sqrt(self.pupil_pad*I_ref))
			ZWFS_sin_left = np.clip(ZWFS_sin_left,-1,1)
			A_left = -np.cos(np.angle(E_ref)+np.pi/2)
			B_left = np.sin(np.angle(E_ref)+np.pi/2)

			# ----- ZWFS right -------
			self.switchMask('right')
			# --- First PROPAGATIONS  ----
			E_in = np.sqrt(self.pupil_pad)*np.exp(1j*np.pad(phi,((self.pad,self.pad),(self.pad,self.pad)), 'constant'))
			E_out = self.propag(phi)
			E_ref = E_in - E_out
			norm = np.sum(np.abs(E_out)**2)

			#----- Define sinus-------
			I_ref = np.abs(E_ref)**2
			img_r = img[:,int(self.nPx_img):]/np.sum(img[:,int(self.nPx_img):])*norm
			ZWFS_sin_right = (img_r-self.pupil_pad-I_ref)/(-2*np.sqrt(self.pupil_pad*I_ref))
			ZWFS_sin_right = np.clip(ZWFS_sin_right,-1,1)
			A_right = -np.cos(np.angle(E_ref)+np.pi/2)
			B_right = np.sin(np.angle(E_ref)+np.pi/2)

			# ---- LOOP for all element of PHI ----
			phi = np.zeros((self.nPx_img,self.nPx_img))
				
			# faster method
			
			D = np.transpose(np.array([ZWFS_sin_left[self.pupil_pad_footprint],ZWFS_sin_right[self.pupil_pad_footprint]]))
			D = D[:, :, np.newaxis]
			M = np.transpose(np.array([[A_left[self.pupil_pad_footprint],B_left[self.pupil_pad_footprint]],[A_right[self.pupil_pad_footprint],B_right[self.pupil_pad_footprint]]]),axes=[2,0,1])
			M_inv = np.linalg.pinv(M)
			R = np.squeeze(np.matmul(M_inv,D))
			phi_vec = np.angle(R[:,1]+1j*R[:,0])
			phi[self.pupil_pad_footprint] = phi_vec
			phi = phi[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]      


		phi = phi*self.pupil_footprint
		# remove piston
		phi = phi-np.sum(phi)*self.pupil_footprint/np.sum(self.pupil_footprint)
		# ----- Record last phase --------
		self.img_simulated = self.getImageSimu(phi)
		self.phase_rec = phi
		self.opd_rec = self.phase_rec*self.wavelength/(2*np.pi)
		self.phase_rec_unwrap = phase_unwrap(phi)*self.pupil_footprint
		

	def getPSF(self,phi = None,amp = None,shannon=2):
		pad = int((2*shannon-1)*self.nPx/2)
		# Get PSF from estmated phase
		if phi is None:
				phi = self.phase_rec
		if amp is None:
				amp = np.sqrt(self.pupil)
		phi_pad =  np.pad(phi,((pad,pad),(pad,pad)), 'constant') # padded pupil
		amp_pad =  np.pad(amp,((pad,pad),(pad,pad)), 'constant') # padded pupil
		# ---- PROPAGATION of the EM field ----
		Psi_FP = fftshift(fft2(fftshift(amp_pad*np.exp(1j*phi_pad))))
		psf = np.abs(Psi_FP)**2
		return psf
		

	#################################################################################
	################################ PSI approach (S.Haffert) #######################
	#################################################################################


	def getImagePSI(self,diversity_cmd_cube,dm):
		"""
		Get true images of Phase-Shifting-Interferometry technique
		(S. Haffert)
		"""
		nProbes = diversity_cmd_cube.shape[0]
		diversity_img_cube = np.zeros((nProbes,int(self.nPx_img),int(self.nPx_img)))
		for d in range(0,nProbes): # Loop on probes
			dm.setSurf(dm.flat_surf+diversity_cmd_cube[d,:,:])
			time.sleep(0.1)
			img = self.getImage()
			diversity_img_cube[d,:,:] = img[:,0:int(self.nPx_img)]
		dm.setFlatSurf()
		
		return diversity_img_cube
		

	def getImageSimuPSI(self,phi,diversity_phase_cube):
		"""
		Get simulated images of Phase-Shifting-Interferometry technique
		(S. Haffert)
		"""
		nProbes = diversity_phase_cube.shape[0]
		diversity_img_cube = np.zeros((nProbes,int(self.nPx_img),int(self.nPx_img)))
		for d in range(0,nProbes): # loop on probes
			img = self.getImageSimu(phi+diversity_phase_cube[d,:,:])
			diversity_img_cube[d,:,:] = img[:,0:int(self.nPx_img)]
			
		return diversity_img_cube
	
	def reconPSI_haffert(self,diversity_img_cube,diversity_phase_cube):
		"""
		Using several frames to perform phase diversity - Sebastiaan Haffert technique
		"""
		nProbes = diversity_img_cube.shape[0]
		for k in range(0,self.nIterRec): # Loop on iterations for propagation 
		
			if k == 0:
				phi = self.startPhase

			D = np.zeros((nProbes,np.sum(self.pupil_pad_footprint)))
			M = np.zeros((np.sum(self.pupil_pad_footprint),nProbes,2))
			for d in range(0,nProbes): # loop on probes

				################# ZWFS left ################
				self.switchMask('left')
				# --- First PROPAGATIONS  ----
				E_in = np.sqrt(self.pupil_pad)*np.exp(1j*np.pad(phi+diversity_phase_cube[d,:,:],((self.pad,self.pad),(self.pad,self.pad)), 'constant'))
				E_out = self.propag(phi+diversity_phase_cube[d,:,:])
				E_ref = E_in - E_out
				norm = np.sum(np.abs(E_out)**2)

				#----- Define sinus-------
				I_ref = np.abs(E_ref)**2
				img_r = diversity_img_cube[d,:,0:int(self.nPx_img)]/np.sum(diversity_img_cube[d,:,0:int(self.nPx_img)])*norm
				sin_formula = (img_r-self.pupil_pad-I_ref)/(-2*np.sqrt(self.pupil_pad*I_ref))
				sin_formula = np.clip(sin_formula,-1,1)
				D[d,:] = sin_formula[self.pupil_pad_footprint]
				cos_angle = -np.cos(np.angle(E_ref)-np.pad(diversity_phase_cube[d,:,:],((self.pad,self.pad),(self.pad,self.pad)), 'constant')+np.pi/2)
				sin_angle = np.sin(np.angle(E_ref)-np.pad(diversity_phase_cube[d,:,:],((self.pad,self.pad),(self.pad,self.pad)), 'constant')+np.pi/2)
				M[:,d,0] = cos_angle[self.pupil_pad_footprint]
				M[:,d,1] = sin_angle[self.pupil_pad_footprint]

			# ---- LOOP for all element of PHI ----
			phi = np.zeros((self.nPx_img,self.nPx_img))


			D = np.transpose(D)
			D = D[:, :, np.newaxis]
			M_inv = np.linalg.pinv(M)
			R = np.squeeze(np.matmul(M_inv,D))
			phi_vec = np.angle(R[:,1]+1j*R[:,0])
			phi[self.pupil_pad_footprint] = phi_vec
			phi = phi[int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2),int(self.nPx_img/2+1-self.nPx/2-1):int(self.nPx_img/2+self.nPx/2)]      


		phi = phi*self.pupil_footprint
		# remove piston
		phi = phi-np.sum(phi)*self.pupil_footprint/np.sum(self.pupil_footprint)
		# ----- Record last phase --------
		self.img_simulated = self.getImageSimu(phi)
		self.phase_rec = phi
		self.opd_rec = self.phase_rec*self.wavelength/(2*np.pi)
		self.phase_rec_unwrap = phase_unwrap(phi)*self.pupil_footprint


	################################ DM Command from images #######################
	def img2cmd(self,img):
		""" Full reconstruction of the signal: image to DM commands """
		if self.nonLinRec == 0:
			self.img = img
			dI = img.ravel()/np.sum(img)-self.img0.ravel()/np.sum(self.img0) # reduced intensities
			cmd = np.dot(self.cmdMat,dI)
		else:
			self.reconNonLinear(img)
			if self.doUnwrap == 0:
				cmd = np.dot(self.phase2mode,self.phase_rec.ravel())
				self.opd_rec = self.phase_rec*self.wavelength/2*np.pi
			else:
				cmd = np.dot(self.phase2mode,self.phase_rec_unwrap.ravel())
		return cmd

###### OLD VERSION OF ZWFS

class zernikeWFS_old(WavefrontSensor):
	"""
	This is a class to use the Zernike WFS

	Attributes
	----------
	pupil_tel: array
		pupil intensities
	pupil_footprint: array
		pupil footprint (0 and 1)
	pupil_pad: array
		shannon-padded pupil
	pupil_pad_footprint: array
		shannon-padded pupil footprint
	cam: cameras Object
		camera used for this sensor
	nPx: int
		pupil resolution
	pad: int
		integer to pad array to shannon
	mask_phase: array
		pyramid mask in phase
	mask: array
		pyramid mask (exp(1j*mask_phase))
	nIterRec: int
		number of iterations for GS algorithm
	nonLinRec: bool
		choose if using non-linear reconstructor while reconstructing DM commands - 1 by default
	doUnwrap: bool
		choose if reconstructed phase is unwrapped - 0 by default
	"""

	def __init__(self,pupil_tel,position_pups,diameter,depth,shannon,wavelength,cam = 0):  
		""" CONSTRUCTOR """
		self.model = 'ZWFS'
		# ------------- SEAL ZWFS CAMERA PARAMETERS -------------
		self.left_x = position_pups[0]
		self.left_y = position_pups[1]
		self.right_x = position_pups[2]
		self.right_y = position_pups[3]
		#self.center_x = position_img[0]
		#self.center_y = position_img[1]
		self.nPx_img_x = position_pups[4]
		self.nPx_img_y = position_pups[5]
		# ---------- Pupil and resolution -----------------
		self.nPx = pupil_tel.shape[1] # Resolution in our pupil
		self.wavelength = wavelength # useful only for DIPSLAYING phase in NANOMETERS
		self.shannon = shannon # Shannon sampling : 1 = 2px per lambda/D - Recommended parameter: shannon = 2
		self.pad = int((2*self.shannon-1)*self.nPx/2) # padding in pupil plane to reach Shannon
		# ---------- LEFT pupil -----------
		self.pupil_left = pupil_tel[0,:,:] # Pupil as recorded when off dimple (INTENSITIES)
		self.pupil_left[self.pupil_left<0] = 0
		self.pupil_footprint_left = np.copy(self.pupil_left)
		self.pupil_footprint_left [self.pupil_footprint_left >0] = 1
		self.pupil_pad_left = np.pad(self.pupil_left,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		self.pupil_pad_footprint_left = np.copy(self.pupil_pad_left)
		self.pupil_pad_footprint_left[self.pupil_pad_footprint_left>0] = 1
		self.diameter_left = diameter[0] # Diameter in lambda/D at lambda = wavelength
		self.depth_left = depth[0] # Depth in phase shift (radians) at lambda = wavelength
		self.mask_left,self.mask_ref_left = self.makeMask(self.diameter_left,self.depth_left)
		self.switchMask('left') # left mask by default
		# ---------- RIGHT pupil -----------
		self.pupil_right = pupil_tel[1,:,:] # Pupil as recorded when off dimple (INTENSITIES)
		self.pupil_right[self.pupil_right<0] = 0
		self.pupil_footprint_right = np.copy(self.pupil_right)
		self.pupil_footprint_right [self.pupil_footprint_right >0] = 1
		self.pupil_pad_right = np.pad(self.pupil_right,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		self.pupil_pad_footprint_right = np.copy(self.pupil_pad_right)
		self.pupil_pad_footprint_right[self.pupil_pad_footprint_right>0] = 1
		self.diameter_right = diameter[1] # Diameter in lambda/D at lambda = wavelength
		self.depth_right = depth[1] # Depth in phase shift (radians) at lambda = wavelength
		self.mask_right,self.mask_ref_right = self.makeMask(self.diameter_right,self.depth_right)

		# --- Camera for the WFS --
		if cam == 0: # case for simulation only
			self.cam = []
			self.nPx_img = self.nPx_img_y#2*self.shannon*self.nPx
		elif cam == 1:
			self.cam = cam
			self.nPx_img = self.nPx_img_y
		else:
			self.cam = cam
			self.nPx_img = self.nPx_img_y

		self.output_size = self.nPx_img ** 2

		# --- Stiopping criteria ----
		self.stopping_criteria = 0 # error difference to stop iterative loop in reconstruction
		# --- reconstruction parameters ----
		self.nonLinRec = 1
		self.algo = 'JPL'
		self.pupilRec = 'left' # cen left - right - both
		self.startPhase = np.zeros((self.nPx,self.nPx))
		self.doUnwrap = 0
		# --- data for reconstruction----
		self.img_simulated = [] # Simulated image from phase estimation - to compare with true data
		self.nIterRec = 50
		self.phase_rec = []
		self.phase_rec_unwrap = []
		self.opd_rec = []
		self.img = [] # image
		self.stopping_criteria = 0.01
		# Reference intensities
		self.img0 = self.cropImg(self.getImageSimu(np.zeros((self.nPx,self.nPx))))

	def switchMask(self,pupil_choice = 'left'):
			self.pupilRec = pupil_choice
			if pupil_choice == 'left':
				self.mask = self.mask_left
				self.mask_ref = self.mask_ref_left
				self.diameter = self.diameter_left
				self.depth = self.depth_left
				self.pupil = self.pupil_left
				self.pupil_footprint = self.pupil_footprint_left
				self.pupil_pad = self.pupil_pad_left
				self.pupil_pad_footprint = self.pupil_pad_footprint_left
			elif pupil_choice == 'right':
				self.mask = self.mask_right
				self.mask_ref = self.mask_ref_right
				self.diameter = self.diameter_right
				self.depth = self.depth_right
				self.pupil = self.pupil_right
				self.pupil_footprint = self.pupil_footprint_right
				self.pupil_pad = self.pupil_pad_right
				self.pupil_pad_footprint = self.pupil_pad_footprint_right

	def makeMask(self,diameter,depth):
		# -------- BUILDING FOCAL PLANE MASK --------
		l = np.linspace(-self.nPx*self.shannon-1,self.nPx*self.shannon,2*self.nPx*self.shannon)
		[xx,yy] = np.meshgrid(l,l)
		# Polar coordinates
		r = np.sqrt(xx**2+yy**2)
		mask = np.ones((2*self.nPx*self.shannon,2*self.nPx*self.shannon),dtype = complex)
		mask[r<int((diameter/2)*2*self.shannon)] = np.exp(1j*depth)
		# -------- BUILDING AMPLITUDE MASK for reference wave: used for JPL reconstructor --------
		mask_ref = np.zeros((2*self.nPx*self.shannon,2*self.nPx*self.shannon))
		mask_ref[r<int((diameter/2)*2*self.shannon)] = 1
		return mask,mask_ref
	
	################################ CALIBRATION PROCESS #######################  
	
	def load_pokeMatrix(self,pokeMatrix,validActuators,thres,calib=1):
		""" Load poke matrix computed through calibration process """
		self.mode2phase = pokeMatrix
		self.phase2mode = np.linalg.pinv(self.mode2phase,1/30)
		self.validActuators = validActuators
		# --- End-to-End calibration matrix ------
		if calib == 1:
			self.calibrate(thres)
		 
	def mapValidCmd(self,cmd_vector):
		""" Map commands vector to map"""
		# map command vector to surface command on WFS valid actuators
		c_map = np.zeros(self.validActuators.shape).astype('float32')
		k = 0
		for i in range(0,c_map.shape[0]):
				for j in range(0,c_map.shape[0]):
					if self.validActuators[i][j] == 1:
							c_map[i][j] = cmd_vector[k]
							k = k + 1
		return c_map
	
	def calibrate(self,thres,mode2phase=None):
		""" Synthetic interaction matrix """
		if not(mode2phase is None):
			self.mode2phase = mode2phase
			self.phase2mode = np.linalg.pinv(self.mode2phase,1/30)
		# -- Linear Calibration -----
		amp_calib = 0.00001
		self.intMat = np.zeros((self.nPx_img*self.nPx_img,self.mode2phase.shape[1]))
		for k in range(self.mode2phase.shape[1]):
				poke_calib = self.mode2phase[:,k].reshape(int(np.sqrt(self.mode2phase.shape[0])),int(np.sqrt(self.mode2phase.shape[0])))
				# --------- PUSH --------
				I_push = self.cropImg(self.getImageSimu(amp_calib*poke_calib))
				I_push = I_push/np.sum(I_push) # normalisation
				# --------- PULL --------
				I_pull = self.cropImg(self.getImageSimu(-amp_calib*poke_calib))
				I_pull = I_pull/np.sum(I_pull) # normalisation
				# -------- Push-pull ------
				s = (I_push-I_pull)/(2*amp_calib)
				self.intMat[:,k] = s.ravel()
		self.cmdMat = np.linalg.pinv(self.intMat,thres)

	################################ PROPAGATION ####################### 

	def propag(self,phi,psf_img=None):
		""" PROPAGATION of the EM field """
		phi_pad =  np.pad(phi,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		# ---- PROPAGATION of the EM field ----
		# To first focal plane
		amp = np.sqrt(self.pupil_pad) # amplitude = sqrt(intensities)
		Psi_FP = fftshift(fft2(fftshift(amp*np.exp(1j*phi_pad))))
		# Multiply by Zernike Phase mask
		if psf_img is None:
			Psi_FP = self.mask*Psi_FP
		else:
			Psi_FP = self.mask*np.sqrt(psf_img)*np.exp(1j*np.angle(Psi_FP))
		# Back to pupil plane
		Psi_PP = fftshift(ifft2(fftshift((Psi_FP))))
		return Psi_PP


	def propagRef(self,phi,psf_img=None):
		""" PROPAGATION of the reference EM field """
		phi_pad =  np.pad(phi,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		# ---- PROPAGATION of the EM field ----
		# To first focal plane
		amp = np.sqrt(self.pupil_pad) # amplitude = sqrt(intensities)
		Psi_FP = fftshift(fft2(fftshift(amp*np.exp(1j*phi_pad))))
		# Multiply by Zernike Phase mask
		if psf_img is None:
			Psi_FP = self.mask_ref*Psi_FP
		else:
			Psi_FP = self.mask_ref*np.sqrt(psf_img)*np.exp(1j*np.angle(Psi_FP))
		# Back to pupil plane
		Psi_PP = fftshift(ifft2(fftshift((Psi_FP))))
		return Psi_PP


	def backPropag(self,amp,phi,psf_img=None):
		""" BACKWARD PROPAGATION of the EM field for GS algorithm """
		# ---- BACKWARD PROPAGATION of the EM field for GS algorithm ---- 
		# To first focal plane
		Psi_FP = fftshift(fft2(fftshift(amp*np.exp(1j*phi))))
		# Multiply by Zernike Phase mask
		if psf_img is None:
			Psi_FP = np.conj(self.mask)*Psi_FP
		else:
			Psi_FP = np.conj(self.mask)*np.sqrt(psf_img)*np.exp(1j*np.angle(Psi_FP))
		# Back to pupil plane
		Psi_PP = fftshift(ifft2(fftshift((Psi_FP))))
		return Psi_PP

	def getImage(self):
		""" Record True image """
		img = self.cam.get()
		img[img<0]=0 #killing negative values
		img = self.roi(img)
		return img
	
	def roi(self,img):
		#img = img[int(self.center_y-self.nPx_img_y/2):int(self.center_y+self.nPx_img_y/2),int(self.center_x-self.nPx_img_x/2):int(self.center_x+self.nPx_img_x/2)]
		img_L = img[self.left_y-int(self.nPx_img/2):self.left_y+int(self.nPx_img/2),self.left_x-int(self.nPx_img/2):self.left_x+int(self.nPx_img/2)]
		img_R = img[self.right_y-int(self.nPx_img/2):self.right_y+int(self.nPx_img/2),self.right_x-int(self.nPx_img/2):self.right_x+int(self.nPx_img/2)]
		stacked_img = np.zeros((self.nPx_img_y,self.nPx_img_x))
		stacked_img[:,0:self.nPx_img_y] = img_L
		stacked_img[:,self.nPx_img_y:] = img_R
		return stacked_img
	
	def getImageSimu(self,phi):
		""" Simulation of Zernike WFS image """
		pupilRec = self.pupilRec
		# Intensities - LEFT
		self.switchMask('left')
		Psi_PP = self.propag(phi)
		img_left = np.abs(Psi_PP)**2
		# Intensities - RIGHT
		self.switchMask('right')
		Psi_PP = self.propag(phi)
		img_right = np.abs(Psi_PP)**2
		# Full image
		img = np.concatenate((img_left,img_right),axis = 1)
		# Back to pupilRec
		self.switchMask(pupilRec)
		return img

	def reconLinear(self,img):
		""" Linear reconstructor using synthetic interaction matrix """
		self.img = img
		dI = img.ravel()/np.sum(img)-self.img0.ravel()/np.sum(self.img0) # reduced intensities
		cmd = np.dot(self.cmdMat,dI)
		self.phase_rec = np.dot(self.mode2phase,cmd)
		#self.img_simulated = self.cropImg(self.getImageSimu(self.phase_rec))


	def reconNonLinear(self,img,nIterRec=None,psf_img=None):
		# Number of iterations for reconstructor
		self.img = img
		if nIterRec is None:
			nIterRec = self.nIterRec
		# Left pupil -------------
		if self.pupilRec == 'left':
			self.switchMask('left')
			img_r = img[:,0:int(self.nPx_img)]#[self.left_y-int(self.nPx/2):self.left_y+int(self.nPx/2),self.left_x-int(self.nPx/2):self.left_x+int(self.nPx/2)]#
			if self.algo == 'JPL':
				self.reconJPL(img_r,nIterRec,psf_img)
			elif self.algo == 'GS':
				self.reconGS(img_r,nIterRec,psf_img)
		# Right pupil -------------
		elif self.pupilRec == 'right':
			self.switchMask('right')
			img_r = img[:,int(self.nPx_img):]#[self.right_y-int(self.nPx/2):self.right_y+int(self.nPx/2),self.right_x-int(self.nPx/2):self.right_x+int(self.nPx/2)]#
			if self.algo == 'JPL':
				self.reconJPL(img_r,nIterRec,psf_img)
			elif self.algo == 'GS':
				self.reconGS(img_r,nIterRec,psf_img)
		# Both pupils -------------
		elif self.pupilRec == 'both':
			# Left pupil
			self.switchMask('left')
			img_r = img[:,0:int(self.nPx_img)]#[self.left_y-int(self.nPx/2):self.left_y+int(self.nPx/2),self.left_x-int(self.nPx/2):self.left_x+int(self.nPx/2)]#
			if self.algo == 'JPL':
				self.reconJPL(img_r,nIterRec,psf_img)
			elif self.algo == 'GS':
				self.reconGS(img_r,nIterRec,psf_img)
			self.phase_rec_left = np.copy(self.phase_rec)
			# right pupil
			self.switchMask('right')
			img_r = img[:,int(self.nPx_img):]#[self.right_y-int(self.nPx/2):self.right_y+int(self.nPx/2),self.right_x-int(self.nPx/2):self.right_x+int(self.nPx/2)]#
			if self.algo == 'JPL':
				self.reconJPL(img_r,nIterRec,psf_img)
			elif self.algo == 'GS':
				self.reconGS(img_r,nIterRec,psf_img)
			self.phase_rec_right = np.copy(self.phase_rec)
			# combine
			self.reconPhaseShifted()
			self.pupilRec = 'both'

	def reconGS(self,img_r,nIterRec=None,psf_img=None):
		""" Non-linear reconstructor using GS algorithm
		img: ZWFS recorded image (dark removed) with two pupils inside
		nIter: number of iteration for the reconstructor
		"""
		# Number of iterations for reconstructor
		if nIterRec is None:
			nIterRec = self.nIterRec
		# Pad image if true image
		if img_r.shape[0] < 2*self.nPx*self.shannon:
			img_r = np.pad(img_r,((int(self.nPx*self.shannon-self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2)),(int(self.nPx*self.shannon-self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2))), 'constant') # padded pupil
		#  ========= GS algortihm to reconstruct phase ==========
		# --- 0 point for phase in detector plane ----
		Psi_0 = self.propag(self.startPhase)
		phi_0 = np.angle(Psi_0)
		# --- 0 point for amplitude in detector plane ---
		frame = np.copy(img_r)
		amp_0 = np.sqrt(frame) # SQRT because img is the intensity
		# --- First BACK PROPAGATION ----
		Psi_p = self.backPropag(amp_0,phi_0)
		# First phase estimate
		phi_k = np.angle(Psi_p)
		phi_k = phi_k*self.pupil_pad_footprint
		phi_k[np.isnan(phi_k)] = 0 # avoid NaN values
		phi_k = phi_k[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)] 
		err_k_previous = float('inf') # for stopping criteria
		for k in range(0,nIterRec):
				# ---- Direct propagation ----
				Psi_d_k = self.propag(phi_k)
				phi_d_k = np.angle(Psi_d_k) # record phase in Zernike WFS camera plane
				# ---- BACK PROPAGATION ----
				Psi_p = self.backPropag(amp_0,phi_d_k)
				phi_k = np.angle(Psi_p)
				phi_k = phi_k*self.pupil_pad_footprint
				phi_k[np.isnan(phi_k)] = 0 # avoid NaN values
				phi_k = phi_k[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]
				# STOPPING CRITERIA -----------
				# Stop if error decrease less than n_p percent
				image_k = self.getImageSimu(phi_k) # Record image of estimated phase through Zernike 
				image_k = image_k[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]
				img_red = img_r[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]
				err_k = error_rms(image_k,img_red)
				if False:#(err_k_previous - err_k)/err_k_previous < self.stopping_criteria: # break if STOPPING CONDITIONS ARE MET
						print('local minimum found')
						print(k) 
						break
				else:
						err_k_previous = err_k
						phi = phi_k
				# ---------------------------
				
		# Record last phase
		self.img_simulated = self.cropImg(self.getImageSimu(phi_k))
		self.phase_rec = phi
		self.opd_rec = self.phase_rec*self.wavelength/(2*np.pi)
		self.phase_rec_unwrap = phase_unwrap(phi)*self.pupil_footprint

	def reconJPL(self,img_r,nIterRec=None,psf_img=None):
		""" Non-linear reconstructor using Iterative algorithm
		img: ZWFS recorded image (dark removed)
		nIter: number of iteration for the reconstructor
		"""
		# Number of iterations for reconstructor
		if nIterRec is None:
			nIterRec = self.nIterRec
		# Pad image if true image
		if img_r.shape[0] < 2*self.nPx*self.shannon:
			img_r = np.pad(img_r,((int(self.nPx*self.shannon-self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2)),(int(self.nPx*self.shannon-self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2))), 'constant') # padded pupil
		#  ========= JPL algortihm to reconstruct phase ==========
		# Use a formula based on a interferometric model of the Zernike WFS - adapted to any phase-shift
		# --- First PROPAGATION through reference ----
		Psi_0 = self.propagRef(self.startPhase)
		I_b = np.abs(Psi_0)**2 # reference intensities
		r = (img_r - self.pupil_pad-4*np.sin(self.depth/2)**2*I_b)/(4*np.sin(self.depth/2)*np.sqrt(self.pupil_pad*I_b))
		r[r>1] = 1 # handle ARCSIN limits in python
		r[r<-1] = -1 # handle ARCSIN limits in python
		t = np.real(np.arcsin(r))
		t[np.isnan(t)] = 0
		phi_k = self.depth/2+t+np.angle(Psi_0)
		phi_k = phi_k*self.pupil_pad_footprint
		phi_k = phi_k[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]      
		# --- LOOP PROPAGATION ----
		err_k_previous = float('inf') # for stopping criteria    
		for k in range(0,nIterRec):
				Psi_k = self.propagRef(phi_k)
				I_b = np.abs(Psi_k)**2
				r = (img_r - self.pupil_pad-4*np.sin(self.depth/2)**2*I_b)/(4*np.sin(self.depth/2)*np.sqrt(self.pupil_pad*I_b))
				r[r>1] = 1
				r[r<-1] = -1
				t = np.real(np.arcsin(r))
				t[np.isnan(t)] = 0
				phi_k = self.depth/2+t+np.angle(Psi_k)
				phi_k = phi_k*self.pupil_pad_footprint
				phi_k =  phi_k - self.pupil_pad_footprint*np.sum(np.sum(phi_k))/np.sum(np.sum(self.pupil_pad_footprint))
				phi_k = phi_k[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]         
				# STOPPING CRITERIA -----------
				# Stop if error decrease less than n_p percent
				image_k = self.getImageSimu(phi_k) # Record image of estimated phase through Zernike 
				image_k = image_k[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]
				img_red = img_r[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]
				err_k = error_rms(image_k,img_red)
				if False:#(err_k_previous - err_k)/err_k_previous < self.stopping_criteria or err_k >= err_k_previous: # break if STOPPING CONDITIONS ARE MET
						print('local minimum found')
						print(k)
						break
				else:
						err_k_previous = err_k
						phi = phi_k
		# ----- Record last phase --------
		self.img_simulated = self.cropImg(self.getImageSimu(phi_k))
		self.phase_rec = phi
		self.opd_rec = self.phase_rec*self.wavelength/(2*np.pi)
		self.phase_rec_unwrap = phase_unwrap(phi)*self.pupil_footprint
			 
		
	def reconPhaseShifted(self):
		"""
		Using data from ZWFS_1 et ZWFS_2 to increase dynamic - Have to be used after using reconstructors for ZWFS_1 and ZWFS_2
		"""
		# ----- ZWFS left -------
		self.switchMask('left')
		beta_1 = np.angle(self.propagRef(self.phase_rec_left))
		beta_1 = beta_1[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]  
		ZWFS_sin_1 = np.sin(self.phase_rec_left-beta_1-self.depth/2)
		A_1 = np.cos(-beta_1-self.depth/2)
		B_1 = np.sin(-beta_1-self.depth/2)
		# ----- ZWFS right -------
		self.switchMask('right')
		beta_2 = np.angle(self.propagRef(self.phase_rec_right))
		beta_2 = beta_2[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]  
		ZWFS_sin_2 = np.sin(self.phase_rec_right-beta_2-self.depth/2)
		A_2 = np.cos(-beta_2-self.depth/2)
		B_2 = np.sin(-beta_2-self.depth/2)
		# ---- LOOP for all element of PHI ----
		phi = np.zeros((self.nPx,self.nPx))
		self.singular_values = np.zeros((self.nPx,self.nPx))
		for i in range(0,self.nPx):
				for j in range(0,self.nPx):
						# --- matrix to be inverted
						M = np.array([[A_1[i,j],B_1[i,j]],[A_2[i,j],B_2[i,j]]])
						D = np.array([ZWFS_sin_1[i,j],ZWFS_sin_2[i,j]])
						M_dag = np.linalg.pinv(M)
						# ======== FOR TESTING =========
						U,S,V = np.linalg.svd(M)
						self.singular_values[i,j] = S[0]+S[1]
						# =======================
						# --- sin(phi) / cos(phi) vector
						R = np.dot(M_dag,D)
						phi[i,j] = np.angle(R[1]+1j*R[0])#np.arctan2(R[1],R[0])
		
		phi = phi*self.pupil_footprint
		self.singular_values = self.singular_values*self.pupil_footprint
		# ----- Record last phase --------
		self.img_simulated = self.cropImg(self.getImageSimu(phi))
		self.phase_rec = phi
		self.opd_rec = self.phase_rec*self.wavelength/(2*np.pi)
		self.phase_rec_unwrap = phase_unwrap(phi)*self.pupil_footprint

	def reconDiversity(self,img_diversity_p,img_diversity_m,diversity_map):
		# Using data from a diversity image to increase dynamic range
		# Diversity phase: heaviside in pupil seems to be the best
		# img_diversity = image recorded with phase diversity 
		# diversity_map = -1 and 1 map to clearify if this zone of the pupil was 'pushed' or 'pulled'
		beta = np.angle(self.propagRef(self.phase_rec))
		beta = beta[int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2),int(self.shannon*self.nPx+1-self.nPx/2-1):int(self.shannon*self.nPx+self.nPx/2)]  
		#img_diff = img_diversity_m - img_diversity_p
		img_diff = self.img- img_diversity_p
		phase_extended = np.copy(self.phase_rec)
		phase_thres = np.zeros((self.nPx,self.nPx))
		phase_thres[np.where((img_diff>0) & (diversity_map == 1))] = 1#-(phase_extended-beta-self.depth/2)
		phase_thres[np.where((img_diff<0) & (diversity_map == -1))] = 1#-(phase_extended-beta-self.depth/2)
		# --- Modify phase ----
		#phase_extended = (phase_extended-beta-self.depth/2)*(1-phase_thres)+(np.pi-(phase_extended-beta-self.depth/2))*phase_thres
		phase_extended = (phase_extended)*(1-phase_thres)+(np.pi-phase_extended)*phase_thres
		
		
		return phase_extended

	def getPSF(self,phi = None):
		# Get PSF from estmated phase
		if phi is None:
				phi = self.phase_rec
		phi_pad =  np.pad(phi,((self.pad,self.pad),(self.pad,self.pad)), 'constant') # padded pupil
		# ---- PROPAGATION of the EM field ----
		# To first focal plane
		amp = np.sqrt(self.pupil_pad) # amplitude = sqrt(intensities)
		Psi_FP = fftshift(fft2(fftshift(amp*np.exp(1j*phi_pad))))
		psf = np.abs(Psi_FP)**2
		return psf
		
	def cropImg(self,img):
		""" crop Image to have same size as true image """
		nPx_shannon = img.shape[1]
		img_left = img[:,0:int(nPx_shannon/2)]
		# crop
		img_left = img_left[int(self.nPx*self.shannon-self.nPx_img/2):int(self.nPx*self.shannon+self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2):int(self.nPx*self.shannon+self.nPx_img/2)]
		img_right = img[:,int(nPx_shannon/2):]
		# crop
		img_right = img_right[int(self.nPx*self.shannon-self.nPx_img/2):int(self.nPx*self.shannon+self.nPx_img/2),int(self.nPx*self.shannon-self.nPx_img/2):int(self.nPx*self.shannon+self.nPx_img/2)]
		# Full image
		img = np.concatenate((img_left,img_right),axis = 1)
		return img

	################################ DM Command from images #######################
	def img2cmd(self,img):
		""" Full reconstruction of the signal: image to DM commands """
		if self.nonLinRec == 0:
			self.img = img
			dI = img.ravel()/np.sum(img)-self.img0.ravel()/np.sum(self.img0) # reduced intensities
			cmd = np.dot(self.cmdMat,dI)
			self.reconLinear(img) # Redondant, just to plot images in close loop.
		else:
			self.reconNonLinear(img)
			if self.doUnwrap == 0:
				cmd = np.dot(self.phase2mode,self.phase_rec.ravel())
				self.opd_rec = self.phase_rec*self.wavelength/2*np.pi
			else:
				cmd = np.dot(self.phase2mode,self.phase_rec_unwrap.ravel())
		return cmd

class SHWFS(WavefrontSensor):
	"""
	This is a class to use the Shack-Hartmann PWFS

	Attributes
	----------
	"""

	def __init__(self):
		print('DO NOT FORGET TO LAUNCH wfs.py on windows machine - cf SEAL procedures')
		self.model = 'SHWFS'
		self.modal = 'zonal'
		# Connexion to Windows computer
		port="5556"
		context = zmq.Context()
		self.socket = context.socket(zmq.REQ)
		self.socket.connect("tcp://128.114.22.20:%s" % port)
		# Pupil size
		self.socket.send_string("pupSize")
		data=self.socket.recv()
		pupSize = np.frombuffer(data, dtype=np.int32)
		self.pupSize = int(pupSize[0])
		# Image size
		self.socket.send_string("imageSize")
		data=self.socket.recv()
		self.imSize = np.frombuffer(data, dtype=np.int32)
		# Number of frames
		self.nFrames = 5
		# Size for pupil
		self.pupSizeHS = 23#np.array([23,23])
		# Get valid apertures mask
		nstack=100 
		wf_ini=self.getWavefront(1)
		wf_ini_arr=np.zeros((nstack,wf_ini.shape[0],wf_ini.shape[1])) #average a sequence of frames to determine the aperture mask, since some frames have fluctuating nan/non-nan values around the edges
		for i in range(nstack):
			wf_ini_arr[i]=self.getWavefront(1)
		wf_ini=np.nanmedian(wf_ini_arr,axis=0)
		self.wfMask=np.zeros(wf_ini.shape)
		self.wfMask[wf_ini!=0]=1
		self.wfMask[11,11] = 1 # central pixel to 0 by definition
		self.wfMask = self.wfMask.astype(bool)
		self.Nsap = int(np.sum(self.wfMask))
		self.output_size = 2*self.Nsap
		# Define Tip-Tilt mode to be removed
		zer2phase_uncropped = zernikeBasis(3,self.pupSize)
		self.zer2phase = np.zeros((self.pupSize**2,3))
		for k in range(0,3):
			zer_cropped = zer2phase_uncropped[:,k].reshape((self.pupSize,self.pupSize))*self.wfMask
			self.zer2phase[:,k] = zer_cropped.ravel()
		self.phase2zer = np.linalg.pinv(self.zer2phase)
		# ---------------- Set if using Slopes or wavefront ----------------
		self.useSlopes = 1 # to use slopes instead of directly reconstructed WF. Works better by using slopes (=1) !
		img = np.load('/home/lab/libSEAL/calibration_files/SHWFS/refSlopes_SHWFS.npy') #computed through close loop with PWFS
		sx = img[0,self.wfMask]
		sy = img[1,self.wfMask]
		self.slopesRef = np.concatenate((sx, sy),0)
	################################ Get Data from SHWFS #######################


	def getImage(self,nFrames = None):
		if nFrames is None:
			nFrames = self.nFrames
		if self.useSlopes == 1:
			img = self.getSlopes(nFrames)
			sx = img[0,self.wfMask]
			sy = img[1,self.wfMask]
			signal = np.concatenate((sx, sy),0)
			# reference
			sx_ref = self.slopesRef[0,self.wfMask]
			sy_ref = self.slopesRef[1,self.wfMask]
			signalRef = np.concatenate((sx_ref, sy_ref),0)
			signal = signal.ravel()-signalRef.ravel()
		else:
			signal  = self.getWavefront(nFrames)
			signal = signal.ravel()
		return signal


	def getImage_raw(self,nFrames=None):
		if nFrames is None:
			nFrames = self.nFrames
		imi=np.zeros(self.imSize)
		for i in range(nFrames):
			self.socket.send_string("image")
			data=self.socket.recv()
			imi = imi + np.frombuffer(data, dtype=np.uint8).reshape(self.imSize)
			imi[np.isnan(imi)] = 0
		imi = imi/nFrames
		return imi

	def getWavefront(self,nFrames=None):
		if nFrames is None:
			nFrames = self.nFrames
		imw=np.zeros((self.pupSize,self.pupSize))
		for i in range(nFrames):
			self.socket.send_string("wavefront")
			data=self.socket.recv()
			imw = imw + np.frombuffer(data, dtype=np.float32).reshape(self.pupSize, self.pupSize)
			imw[np.isnan(imw)] = 0
		imw = imw/nFrames 
		return imw

	def getSlopes(self,nFrames=None):
		if nFrames is None:
				nFrames = self.nFrames
		ims=np.zeros((2,self.pupSize, self.pupSize))
		for i in range(nFrames):
			self.socket.send_string("slopes")
			data=self.socket.recv()
			slopes = np.frombuffer(data, dtype=np.float32).reshape(self.pupSize, 2*self.pupSize)
			sx = slopes[:,:self.pupSize]
			sy = slopes[:,self.pupSize:]
			ims = ims+np.array([sx, sy])
		ims = ims/nFrames
		return ims

	def setRefSlopes(self):
		s = self.getSlopes()
		self.slopesRef = s
		print('Reference slopes')

	def getSpots(self,nFrames=None):
		if nFrames is None:
			nFrames = self.nFrames
		imsp=np.zeros((2,self.pupSizeHS, self.pupSizeHS))
		for i in range(nFrames):
			self.socket.send_string("spots")
			data=self.socket.recv()
			spots = np.frombuffer(data, dtype=np.float32).reshape(2*self.pupSizeHS, self.pupSizeHS)
			spotx = spots[:self.pupSizeHS, :]
			spoty = spots[self.pupSizeHS:, :]
			imsp = imsp + np.array([spotx, spoty])
		imsp = imsp/nFrames
		return imsp


	################################ Calibration #######################
	'''
	def calibrate(self,dm,amp_calib=None):
		self.validActuators = dm.valid_actuators_map
		if self.useSlopes == 1:
			self.intMat = np.zeros((2*self.Nsap,int(np.sum(dm.valid_actuators_map))))
		else:
			self.intMat = np.zeros((self.Nsap,int(np.sum(dm.valid_actuators_map))))
		if amp_calib is None:
			amp_calib = 0.1 # for pseudo-inverse
		compt = 0
		for i in range(dm.nAct):
			for j in range(dm.nAct):
				if dm.valid_actuators_map[i,j] == 1:
					# --------- PUSH --------
					dm.pokeAct(amp_calib,[i,j])
					time.sleep(0.1)
					s_push = self.getImage()
					# --------- PULL --------
					dm.pokeAct(-amp_calib,[i,j])
					time.sleep(0.1)
					s_pull = self.getImage()
					# -------- Push-pull ------
					if self.useSlopes == 1:
						sx = (s_push[0,self.wfMask]-s_pull[0,self.wfMask])/(2*amp_calib)
						sy = (s_push[1,self.wfMask]-s_pull[1,self.wfMask])/(2*amp_calib)
						s = np.concatenate((sx, sy),0)
					else:
						s = (s_push[self.wfMask]-s_pull[self.wfMask])/(2*amp_calib)
					self.intMat[:,compt] = s.ravel()
					compt = compt + 1
					print(compt/int(np.sum(dm.valid_actuators_map)))
		self.intMat[np.isnan(self.intMat)] = 0 # removing NaN
		dm.setFlatSurf()
		self.compute_cmdMat()
		

	def compute_cmdMat(self,thres=None):
		if thres is None:
			thres = 1/30 # for pseudo-inverse
		self.cmdMat = np.linalg.pinv(self.intMat,thres)

	
	def mapValidCmd(self,cmd_vector):
		""" Map commands vector to map"""
		# map command vector to surface command on WFS valid actuators
		c_map = np.zeros(self.validActuators.shape).astype('float32')
		k = 0
		for i in range(0,c_map.shape[0]):
				for j in range(0,c_map.shape[0]):
					if self.validActuators[i][j] == 1:
							c_map[i][j] = cmd_vector[k]
							k = k + 1
		return c_map
	'''
	def rm_ptt_wf(self,img):
		ttp = np.dot(self.phase2zer,img.ravel())
		img = img-ttp[0]*self.zer2phase[:,0].reshape((self.pupSize,self.pupSize))
		img = img-ttp[1]*self.zer2phase[:,1].reshape((self.pupSize,self.pupSize))
		img = img-ttp[2]*self.zer2phase[:,2].reshape((self.pupSize,self.pupSize))
		img = img*self.wfMask
		return img

	def img2cmd(self,img):
		""" Full reconstruction of the signal: image to DM commands """
		if self.useSlopes == 1:
			#sx = img[0,self.wfMask]
			#sy = img[1,self.wfMask]
			#signal = np.concatenate((sx, sy),0)-self.slopesRef
		#else:
		#	img = self.rm_ptt_wf(img)
		#	signal = img[self.wfMask]
			cmd = np.dot(self.cmdMat,img)
		return cmd

class PhotonicLantern(WavefrontSensor):
	"""
	A photonic lantern wavefront sensor. Note: till I decide what to do with this, 
	it'll have an external dependency on /home/lab/asengupta/photonics via "reader".
	"""
	def __init__(self, cam, reader):
		self.model = "Photonic Lantern"
		self.modal = 'zernike'
		self.exclude_ports = np.zeros(reader.nports, dtype=bool)
		reader.set_centroids(cam.get(1000) + cam.dark)
		self.reader = reader
		self.cam = cam
		self.output_size = reader.nports
		self.flat_img = self.getImage()

	def calibrate(self, dm, lim=0.2, **kwargs):
		print("Checking for saturated ports across several aberrations")
		max_frame = np.zeros(self.reader.imgshape)
		z = np.arange(2, 12)
		for _ in trange(20):
			dm.pokeZernike(np.random.uniform(-lim, lim, len(z)), z)
			max_frame = np.maximum(max_frame, self.cam.get() + self.cam.dark)
		dm.setFlatSurf()
		self.update_excluded(max_frame)
		self.output_size = self.reader.nports - np.sum(self.exclude_ports)
		super().calibrate(dm, **kwargs)

	def update_flat(self, dm, nFrames=None):
		dm.setFlatSurf()
		self.flat_img = self.getImage(exclude=False, nFrames=nFrames)

	def update_excluded(self, img):
		self.exclude_ports = self.reader.saturating_ports(img + self.cam.dark)
		print(f"Recalibrating to exclude ports {np.where(self.exclude_ports)[0]}")

	def getImage(self, exclude=True, nFrames=None):
		img = self.cam.get(self.cam.nFrames if nFrames is None else nFrames)
		img = img / np.sum(img)
		if exclude:
			intensities = self.reader.get_intensities(img, exclude=np.where(self.exclude_ports)[0])
		else:
			intensities = self.reader.get_intensities(img)
		if np.median(intensities) <= 0:
			print("Most measurements are negative: check that your dark frame is up to date!")
		return intensities
	
	def img2cmd(self, img):
		return self.cmdMat @ (img - self.flat_img[~self.exclude_ports])

	def plot_percent_IM_change(self, dm, amp=0.02):
		self.calibrate(dm, amp_calib=amp, modes_number=self.intMat.shape[1])
		im_one = np.copy(self.intMat)
		self.calibrate(dm, amp_calib=amp, modes_number=im_one.shape[1])
		im_two = self.intMat
		fig(100 * (im_two - im_one) / im_one, title="Instant percent change in the interaction matrix")
