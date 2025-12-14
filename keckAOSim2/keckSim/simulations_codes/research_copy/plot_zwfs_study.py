#%%
import numpy as np
import sys
import math
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from io import BytesIO
import imageio.v2 as imageio
import os

os.chdir("/home/mcisse/keckAOSim/keckSim/")
sys.path.insert(0, os.getcwd())
from simulations_codes.research_copy.KAO_parameter_file_research_copy import initializeParameterFile
from simulations_codes.research_copy.initialize_AO_research_copy import initialize_AO_hardware
from simulations_codes.vandamstrehl import *


param = initializeParameterFile()
KAO = initialize_AO_hardware(param)
r0 = param['r0']
mag = param['magnitude_guide']
freq_zwfs = 4
alpha = param['alpha']
samp = param['samplingTime']
NCPA_vect = [False,True]
M1_vect = [False,True]

start_idx = 50
ZWFS_param = {'activate': True, 'max_CLiter':5,'gain':1,'n_average':5,'avg_time':5,'freq':4,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

path = '/home/mcisse/keckAOSim/keckSim/data/'
study = 'AO_data_ZWFS_exposure_study'
sys_ = 'KAO_R_band_SH_20x20_NGS_magnitude'
'''
expo_zwfs = [1,2,4]
M1_list = {}
PSF_list = {}
SR_list = {}
        
for k in expo_zwfs:
    name_gen = f'{path}{sys_}_{mag}_M1_{M1_vect[1]}_NCPA_{NCPA_vect[0]}_r0_{int(r0*100)}cm_alpha_{alpha}_ZWFS_fps{freq_zwfs}_AO_data_ZWFS_expo_{k}s.npy'
    
    expo_time = (1/freq_zwfs)   
    frames_per_img = int(expo_time / samp) 
    avg_time = k   
    Nimgs = int(freq_zwfs * avg_time) # ZWFS exposures to average for one measurement
    nrep_max = ZWFS_param['n_average']
    ACS_iter = frames_per_img * Nimgs * nrep_max
    print(f"In total {ACS_iter} AO iterations are needed for ONE ACS command")
    
    AO_data_gains = np.load(name_gen, allow_pickle=True).item()
    SR = AO_data_gains['SR']
    res = AO_data_gains['residual']
    PSF = AO_data_gains['PSF_LE']
    M1_amp = AO_data_gains['M1_OPD']
    
    size_ = SR.shape[0]-ACS_iter*ZWFS_param['max_CLiter']
    sr_ini = SR[start_idx:size_+start_idx]
    sr_end = SR[SR.shape[0]-size_:]
    M1_list[k] = M1_amp
    PSF_list[k] = PSF
    SR_list[k] = [np.mean(sr_ini),np.mean(sr_end)]
    
    #plots
    label = f'Exposure {k}s'
    plt.figure(), plt.plot(M1_amp,'bo',label=label),plt.xlabel('ACS commands'),plt.ylabel('M1 nm RMS'), plt.legend(), plt.title(f'Star mag {mag} r0 = {r0*100:.0f}cm: M1 OPD vs Iterations'),plt.show(block=False)


markers = ['o', 's', 'd']

fig_m1  = plt.figure(); ax_m1  = plt.gca()
ax_m1.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: M1 OPD vs Iterations'); ax_m1.set_xlabel('ACS commands'); ax_m1.set_ylabel('M1 nm RMS'); ax_m1.grid(True)

fig_sr  = plt.figure(); ax_sr  = plt.gca()
ax_sr.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: SR vs Iterations'); ax_sr.set_xlabel('ZWFS Closed-loop number'); ax_sr.set_ylabel('SR'); ax_sr.grid(True)

aa = name_gen.split('expo')[0]
        
for i,k in enumerate(SR_list.keys()):
    label = f'Exposure {k}s'
    line = ax_m1.plot(M1_list[k],marker=markers[i % len(markers)],label=label)[0]
    color = line.get_color()
    ax_sr.plot(i+1,SR_list[k][0],'*',color = color,markersize=10)
    ax_sr.plot(i+1,SR_list[k][1],'o',color = color,markersize=10,label=label)
    
ax_sr.legend()
ax_m1.legend()    
plt.show(block=False)
fig_sr.savefig(f'{aa}exposure_study_SR.png', dpi=300, bbox_inches='tight')
fig_m1.savefig(f'{aa}exposure_study_M1.png', dpi=300, bbox_inches='tight')    
'''    

#%% Long study
'''
expo_zwfs = np.arange(1,32,10)    
name_gen = f'{path}{sys_}_{mag}_M1_{M1_vect[1]}_NCPA_{NCPA_vect[0]}_r0_{int(r0*100)}cm_alpha_{alpha}_ZWFS_fps{freq_zwfs}_{study}.npy'

long_data = np.load(name_gen, allow_pickle=True).item()
keys = sorted(long_data.keys())
ncols = math.ceil(np.sqrt(len(keys)))
nrows = math.ceil(len(keys) / ncols)
    
SR_mat  = np.array([long_data[k]['SR']       for k in keys], dtype=object)
PSF_mat = np.stack([long_data[k]['PSF_LE']   for k in keys], axis=0)
M1_mat  = np.array([long_data[k]['M1_OPD']   for k in keys],dtype=object)    

fig_m1  = plt.figure(); ax_m1  = plt.gca()
ax_m1.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: M1 OPD vs Iterations'); ax_m1.set_xlabel('ACS commands'); ax_m1.set_ylabel('M1 nm RMS'); ax_m1.grid(True)

fig_sr_bis  = plt.figure(); ax_sr_bis  = plt.gca()
ax_sr_bis.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: SR vs Iterations'); ax_sr_bis.set_xlabel('ZWFS Closed-loop number'); ax_sr_bis.set_ylabel('SR'); ax_sr_bis.grid(True)

markers = ['o', 's', 'd','*','.']  
  
for i , k in enumerate(keys):
    
    expo_time = (1/freq_zwfs)   
    frames_per_img = int(expo_time / samp) 
    avg_time = expo_zwfs[i]   
    Nimgs = int(freq_zwfs * avg_time) # ZWFS exposures to average for one measurement
    nrep_max = ZWFS_param['n_average']
    ACS_iter = frames_per_img * Nimgs * nrep_max
    print(f"In total {ACS_iter} AO iterations are needed for ONE ACS command") 
    
    sr_i = SR_mat[i]
    size_ = sr_i.shape[0]-ACS_iter*ZWFS_param['max_CLiter']
    sr_ini = np.mean(sr_i[start_idx:size_+start_idx])
    sr_end = np.mean(sr_i[sr_i.shape[0]-size_:])

    label = f'Exposure {avg_time}s'
    
    line = ax_m1.plot(M1_mat[i],marker=markers[i % len(markers)],label=label)[0]
    color = line.get_color()
    ax_sr_bis.plot(i+1,sr_ini,'*',color = color,markersize=10)
    ax_sr_bis.plot(i+1,sr_end,'o',color = color,markersize=10,label=label)
    
ax_m1.legend()    
ax_sr_bis.legend()    
plt.show(block=False)    
    
aa = name_gen.split('study')[0] 
fig_name = f'{aa}{expo_zwfs[0]}s_{expo_zwfs[-1]}s'   
fig_sr_bis.savefig(f'{fig_name}_SR.png', dpi=300, bbox_inches='tight')
fig_m1.savefig(f'{fig_name}_M1.png', dpi=300, bbox_inches='tight')       
'''    

# plot all data
'''
expo_zwfs_all = np.concatenate(([1,2,4],  expo_zwfs[1:]))
 
fig_m1  = plt.figure(); ax_m1  = plt.gca()
ax_m1.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: M1 OPD vs Iterations'); ax_m1.set_xlabel('ACS commands'); ax_m1.set_ylabel('M1 nm RMS'); ax_m1.grid(True)

fig_sr_bis  = plt.figure(); ax_sr_bis  = plt.gca()
ax_sr_bis.set_title(f'Star mag {mag} r0 = {r0*100:.0f}cm: SR vs Iterations'); ax_sr_bis.set_xlabel('ZWFS Closed-loop number'); ax_sr_bis.set_ylabel('SR'); ax_sr_bis.grid(True)    

markers = ['o', 's', 'd','*','.','v','+']     
SR_bis = SR_mat[1:]
M1_bis = M1_mat[1:]
for i in range(len(expo_zwfs_all)):
    exp = expo_zwfs_all[i]
    label = f'Exposure {exp}s'
    
    if i<3:
        line = ax_m1.plot(M1_list[exp],marker=markers[i % len(markers)],label=label)[0]
        color = line.get_color()
        ax_sr_bis.plot(i+1,SR_list[exp][0],'*',color = color,markersize=10)
        ax_sr_bis.plot(i+1,SR_list[exp][1],'o',color = color,markersize=10,label=label)
    else:
        expo_time = (1/freq_zwfs)   
        frames_per_img = int(expo_time / samp)   
        Nimgs = int(freq_zwfs * exp) 
        nrep_max = ZWFS_param['n_average']
        ACS_iter = frames_per_img * Nimgs * nrep_max
        print(f"In total {ACS_iter} AO iterations are needed for ONE ACS command") 
    
        sr_i = SR_bis[i-3]
        size_ = sr_i.shape[0]-ACS_iter*ZWFS_param['max_CLiter']
        sr_ini = np.mean(sr_i[start_idx:size_+start_idx])
        sr_end = np.mean(sr_i[sr_i.shape[0]-size_:])
    
        line = ax_m1.plot(M1_bis[i-3],marker=markers[i % len(markers)],label=label)[0]
        color = line.get_color()
        ax_sr_bis.plot(i+1,sr_ini,'*',color = color,markersize=10)
        ax_sr_bis.plot(i+1,sr_end,'o',color = color,markersize=10,label=label)
        
ax_m1.legend()    
ax_sr_bis.legend()    
plt.show(block=False)     
aa = name_gen.split('study')[0] 
fig_name = f'{aa}{expo_zwfs[0]}s_{expo_zwfs[-1]}s'   
fig_sr_bis.savefig(f'{fig_name}_SR_All.png', dpi=300, bbox_inches='tight')
fig_m1.savefig(f'{fig_name}_M1_All.png', dpi=300, bbox_inches='tight')     
'''    
    
#%% New data
# PSf diff
psf_ini = []
psf_diff = []

ratio_time = int(KAO.science_detector.integrationTime/KAO.tel.samplingTime)

for k in range(ratio_time+1):
    KAO.tel.resetOPD()
    KAO.ngs * KAO.tel * KAO.ttm * KAO.dm * KAO.wfs 
    
    KAO.science * KAO.tel *KAO.science_detector # if cog offset KAO.opd_ncpa 
    psf = KAO.science_detector.frame[:]
    
    if k>0 and k%ratio_time==0:
        psf_diff.append(psf) 

ZWFS_param = {'activate': True, 'max_CLiter':5,'gain':1,'n_average':5,'avg_time':4,'freq':4,'subGZ':4,'maxZ':1,'maxGlobalZ':4}

name_gen_false = f'{path}New_{param["name"]}_NGS_magnitude_8_M1_True_NCPA_False_r0_16cm_alpha_20_ZWFS_fps{ZWFS_param["freq"]}_exposure_{ZWFS_param["avg_time"]}s_offset_False.npy'
name_gen = f'{path}New_KAO_R_band_SH_20x20_NGS_magnitude_8_M1_True_NCPA_False_r0_16cm_alpha_20_ZWFS_fps4_exposure_4s_offset_True.npy'
data_all = np.load(name_gen, allow_pickle=True).item()

PSF = data_all['PSF_LE']
psf_batch = data_all['PSF_batches']
SR = data_all['SR']
residual = data_all['residual']
m1_amp = data_all['M1_OPD']
total_M1 = data_all['total_M1_applied']
M1_OPD = data_all['Input_M1']
M1_final = data_all['M1_final_shape']
offset_seg = data_all['Offset_seg']
offset = data_all['Offset']
pup = M1_OPD.copy()
pup[np.where(pup!=0)]=1
input_segments = M1_OPD+offset_seg
input_segments[np.where(M1_OPD==0)]=0
amp_seg = float(np.std(input_segments[np.where(pup!= 0)])) * 1e9
amp_off_seg = float(np.std(offset_seg[np.where(pup!= 0)])) * 1e9
fig_name = name_gen.split('.')[0]
plt.figure()
plt.imshow(PSF**0.2)
plt.colorbar()
plt.show(block=False)
#plt.savefig(f'{fig_name}_LE_PSF.png', dpi=300, bbox_inches='tight') 

plt.figure()
plt.imshow(total_M1*1e9)
plt.title('total M1')
plt.colorbar()
plt.show(block=False)
plt.savefig(f'{fig_name}_output_phase_wfs.png', dpi=300, bbox_inches='tight') 

vmin = M1_OPD.min()*1e9
vmax = M1_OPD.max()*1e9
plt.figure()
plt.imshow(M1_OPD*1e9,vmin=vmin ,vmax=vmax)
plt.title('M1 initial shape [nm]')
plt.colorbar()
plt.show(block=False)
plt.savefig(f'{fig_name}_M1_initial_shape.png', dpi=300, bbox_inches='tight') 

plt.figure()
plt.imshow(M1_final*1e9,vmin=vmin ,vmax=vmax)
plt.title('M1 final shape [nm]')
plt.colorbar()
plt.show(block=False)
plt.savefig(f'{fig_name}_M1_final_shape.png', dpi=300, bbox_inches='tight') 

plt.figure()
plt.imshow(offset*1e9)
plt.title('NCPA [nm]')
plt.colorbar()
plt.show(block=False)
plt.savefig(f'{fig_name}_offset.png', dpi=300, bbox_inches='tight') 

plt.figure()
plt.imshow(offset_seg*pup*1e9)
plt.title('NCPA projected on segment [nm]')
plt.colorbar()
plt.show(block=False)
plt.savefig(f'{fig_name}_offset_seg.png', dpi=300, bbox_inches='tight') 

plt.figure()
plt.imshow(input_segments*1e9)
plt.title('M1+ncpa projected on segments [nm]')
plt.colorbar()
plt.show(block=False)
plt.savefig(f'{fig_name}_M1_plus_offset.png', dpi=300, bbox_inches='tight') 

plt.figure()
plt.plot(m1_amp,'ro')
plt.title('M1 amplitude')
plt.xlabel('ZWFS Iterations #')
plt.ylabel('WFE [nm] RMS')
plt.show(block=False)
plt.savefig(f'{fig_name}_M1_amplitude.png', dpi=300, bbox_inches='tight') 

plt.figure()
plt.plot(SR,'r')
plt.title('SR')
plt.show(block=False)
plt.savefig(f'{fig_name}_SR.png', dpi=300, bbox_inches='tight') 

plt.figure()
plt.plot(residual,'ro')
plt.title('residual [nm] RMS')
plt.xlabel('Iterations #')
plt.ylabel('WFE [nm] RMS')
plt.show(block=False)
plt.savefig(f'{fig_name}_residual.png', dpi=300, bbox_inches='tight') 

images = []
deb = 512//2-50
fin = 512//2+50

psf_acs = [psf_batch[0,1,:,:],psf_batch[1,1,:,:], psf_batch[2,1,:,:],psf_batch[3,1,:,:],psf_batch[4,1,:,:], psf_batch[5,1,:,:],psf_batch[6,1,:,:]]

psf_beg = psf_batch[0,:,:,:]
sr_ini = strehl(psf_beg[0,:,:],psf_diff[0],photometry_radius=50,pos=[256,256])
sr_final = strehl(psf_acs[-1],psf_diff[0],photometry_radius=50,pos=[256,256])

fig = plt.figure(); ax  = plt.gca()

'''
for k in range(psf_beg.shape[0]):
    aa = psf_beg[k,:,:]
    ax.imshow(aa[deb:fin,deb:fin]**0.2)
    ax.set_title(f'PSF SR = {round(sr_ini*100)}% @H band', fontsize=14)  # ? Add title here
     
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    images.append(imageio.imread(buf))
    buf.close()
    ax.clear()
plt.close(fig)
imageio.mimsave(f'{path}Before_ZWFS_M1_True_offset_True_psf_zoom_animation.gif', images, duration=0.3)

images = []
fig = plt.figure(); ax  = plt.gca()
for k in range(len(psf_acs)):
    aa = psf_acs[k]
    ax.imshow(aa[deb:fin,deb:fin]**0.2)
    ax.set_title(f'PSF SR = {round(sr_final*100)}% @H band', fontsize=14)  # ? Add title here
     
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    images.append(imageio.imread(buf))
    buf.close()
    ax.clear()
plt.close(fig)
imageio.mimsave(f'{path}During_ZWFS_CL_M1_True_offset_True_psf_zoom_animation.gif', images, duration=0.3)

for k in range(7):
    psf_acsk = psf_batch[k,:,:,:]
    fig, axes = plt.subplots(1, 2, figsize=(10,10))
    axes = axes.ravel()
    fig.suptitle(f'ACS iteration{k}', fontsize=14)
    ax = axes[0]
    im = ax.imshow(psf_acsk[0,:,:]**0.2, cmap='viridis')
    ax.axis('off')
    ax = axes[1]
    im = ax.imshow(psf_acsk[-1,:,:]**0.2, cmap='viridis')
    ax.axis('off')
    plt.show(block=False)

for k in range(10):
    psf = psf_beg[k,:,:]
    fig = plt.figure(); axes  = plt.gca()
    fig.suptitle(f'Initial psf iteration{k}', fontsize=14)
    axes.imshow(psf**0.2, cmap='viridis')
    axes.axis('off')
    plt.show(block=False)
'''












