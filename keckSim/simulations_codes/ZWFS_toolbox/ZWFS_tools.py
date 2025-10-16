import numpy as np
import sys 
import time
import datetime
from tqdm import tqdm, trange
from hcipy import *
from scipy.ndimage.interpolation import rotate

#from PIL import Image
import scipy.ndimage
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

import os
os.chdir("/home/mcisse/keckAOSim/keckSim/")
from simulations_codes.ZWFS_toolbox.tools import *



def auto_exp(CAM):
    """
    Define automatically an exposure so the detector is not saturating.
    - Criteria: Mean of 40 pixels with most count below 6000
    """
    # Get image
    trigger_saturation = True
    low_light = 0
    
    while trigger_saturation:
        nFrames = 10
        grab_info = CAM.grab_n(nFrames)[0]
        im = grab_info.data
        img = np.sum(im,axis=0)/nFrames
        value = np.sort(img.ravel())
        
        if np.mean(value[-40:])>10000:
            # get t_int -------
            t_int = CAM.get_tint()
            if low_light == 0:
                t_int_new = t_int/2
            else:
                t_int_new = 0.75*t_int
                low_light = 0
            # set new t_int ----
            CAM.set_fps(1/t_int_new)
            CAM.set_tint(t_int_new)
                
        else:
            if np.mean(value[-40:])<6000:
                # get t_int -------
                t_int = CAM.get_tint()
                t_int_new = 2*t_int
                if t_int_new > 0.5:
                    t_int_new = 0.5
                    trigger_saturation = False
                # set new t_int ----
                CAM.set_fps(1/t_int_new)
                CAM.set_tint(t_int_new)
                # flag
                low_light = 1
            else:
                trigger_saturation = False
                print('Exposure time adjusted')
        
        
def mionningMatrix(dm,n_act_minion):
    """
    Defining Edge Actuators Minionning !
    """
    # -------- Valid Actuator grid -----
    nAct = int(dm.nAct)
    nAct_minion = int(n_act_minion)
    grid=np.mgrid[0:nAct,0:nAct]
    rgrid=np.sqrt((grid[0]-nAct/2+0.5)**2+(grid[1]-nAct/2+0.5)**2)
    free_actuators_map = np.zeros((nAct,nAct)).astype(np.float32)
    free_actuators_map[np.where(rgrid<(17.5-nAct_minion))]=1
    free_actuators_map[20,7] = 0

    minionned_actuators_map = dm.valid_actuators_map-free_actuators_map
    N_minions = np.sum(minionned_actuators_map)
    N_free = np.sum(np.sum(dm.valid_actuators_map)-N_minions)


    # ---- free actuator with referencing number map ------
    free_actuators_number = np.zeros_like(dm.valid_actuators_map)
    k = 0
    for i in range(0,nAct):
	    for j in range(0,nAct):
		    if dm.valid_actuators_map[i][j] == 1 and minionned_actuators_map[i,j] == 0:
			    free_actuators_number[i][j] = k
			    k = k + 1
					    
					    
    minioning_matrix = np.zeros((int(np.sum(dm.valid_actuators_map)),int(N_free))) # Mionions Matrix
    minioning_matrix_outer_ring = np.zeros((int(np.sum(dm.valid_actuators_map)),int(np.sum(dm.valid_actuators_map)))) # Mionions Matrix outer ring

    compt = 0
    for i in range(0,nAct):
        for j in range(0,nAct):
            if dm.valid_actuators_map[i,j] == 1:
                if minionned_actuators_map[i,j] == 1: # Minionned actuator
                    # around given actuator
                    for k in np.arange(-1,2):
                        for l in np.arange(-1,2):
                            if 0<= i+k <= (nAct-1) and 0<= j+l <= (nAct-1): 
                                if minionned_actuators_map[i+k,j+l] == 0 and dm.valid_actuators_map[i+k,j+l] == 1 and (np.abs(k)+np.abs(l)) != 0:# actuator not minnioned
                                    minioning_matrix[compt,int(free_actuators_number[i+k,j+l])] = 1
                    if np.sum(minioning_matrix[compt,:]) != 0:
                        minioning_matrix[compt,:] = minioning_matrix[compt,:]/np.sum(minioning_matrix[compt,:]) # normalize
                    else:
                        # around given actuator
                        for k in np.arange(-1,2):
                            for l in np.arange(-1,2):
                                if 0<= np.abs(i+k) <= (nAct-1) and 0<= np.abs(j+l) <= (nAct-1): # any actuators
                                    minioning_matrix_outer_ring[compt,int(dm.valid_actuators_number[i+k,j+l]-1)] = 1
                        minioning_matrix_outer_ring[compt,:] = minioning_matrix_outer_ring[compt,:]/np.sum(minioning_matrix_outer_ring[compt,:]) # normalize
                                    
                else: # free actuator case
                    minioning_matrix[compt,int(free_actuators_number[i,j])] = 1
                compt = compt + 1

    for k in range(0,int(np.sum(dm.valid_actuators_map))):
        if np.max(minioning_matrix_outer_ring[k,:]) == 0:
            minioning_matrix_outer_ring[k,k] = 1
            
    free2all = np.dot(minioning_matrix_outer_ring,minioning_matrix)
    
    return free2all,free_actuators_map
    
def moveTTM(FAM,requested_pos,max_iters=10):

    requested_pos_x, requested_pos_y = round(requested_pos[0],0),round(requested_pos[1],0)
    current_pos = FAM.get_pos()
    if round(current_pos[0],0) != requested_pos_x or round(current_pos[1],0) != requested_pos_y:
        # Move TTM
        FAM.set_pos(requested_pos, block=True)
        
    print(" Requested X,Y positions:",str(requested_pos_x),",",str(requested_pos_y))
    m = 0
    while round(FAM.get_pos()[0],0) != requested_pos_x or round(FAM.get_pos()[1],0) != requested_pos_y:
        # Move TTM
        FAM.set_pos(requested_pos, block=True)
        #print(" X,Y positions:",str(round(FAM.get_pos()[0],4)),",",str(round(FAM.get_pos()[1],4)))
        m+=1
        if m > max_iters:
            #print("  Requested X,Y positions:",str(requested_pos_x),",",str(requested_pos_y))
            #print("  X,Y positions:",str(round(FAM.get_pos()[0],4)),",",str(round(FAM.get_pos()[1],4)))
            break
    print(" X,Y positions:",str(round(FAM.get_pos()[0],4)),",",str(round(FAM.get_pos()[1],4)))

def takeImageOFFmask(cam,FAM,offset=600):
    
    # Record current on-mask position:
    onmask = FAM.get_pos()
    onmask_x = onmask[0]
    onmask_y = onmask[1]

    # Move off-mask:
    requested_pos = np.array([onmask_x+offset,onmask_y+offset])
    moveTTM(FAM,requested_pos)
    time.sleep(cam.get_exp()*2)

    # Take image:
    im = cam.get()
    time.sleep(cam.get_exp()*2)

    # Move back on-mask:
    moveTTM(FAM,onmask)

    return im

def XYalignment(onmask_x,onmask_y,metric,drange=0.02,Nsteps=5,stop_grid=True):
    x_range = np.linspace(onmask_x-drange,onmask_x+drange,Nsteps)
    y_range = np.linspace(onmask_y-drange,onmask_y+drange,Nsteps)

    sums = np.array([])
    xs = np.array([])
    ys = np.array([])

    plt.figure(figsize=(15,10))
    i = -1
    subplot_i = 0
    for yi, y in enumerate(y_range):
        for xi,x in enumerate(x_range):
            
            moveTTM(FAM,np.array([x,y]))

            im = ZWFS.getImage()
            sum_outside = np.nansum(im*metric)

            sums = np.append(sums,sum_outside)
            xs = np.append(xs,x)
            ys = np.append(ys,y)
            
            subplot_i += 1
            plt.subplot(len(x_range),len(y_range),subplot_i)
            #if subplot_i <= len(x_range):
            #    plt.title('('+str(int(round(FAM.get_pos()[0])))+','+str(int(round(FAM.get_pos()[1])))+')',fontsize=10)#\nOuter: '+str(round(sum_outside,1)))
            #elif xi == 0:
            #    plt.title('('+str(int(round(FAM.get_pos()[0])))+','+str(int(round(FAM.get_pos()[1])))+')',fontsize=10)#\nOuter: '+str(round(sum_outside,1)))
            plt.imshow(im)#,vmin=0,vmax=100)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.5)

            if stop_grid == True:
                i+=1
                if xi > 0 and sums[i] < sums[i-1]:
                    subplot_i += len(x_range)-xi-1
                    print("Moving in wrong direction, next row.")
                    break 

    max_sum = np.max(sums)
    onmask_x, onmask_y = xs[np.where(sums==max_sum)][0], ys[np.where(sums==max_sum)][0]

    return onmask_x,onmask_y

def XYalignmentSearch(cam,onmask_x,onmask_y,drange=0.02,Nsteps=5,nFrames=1):
    
    nFrames_orig = cam.nFrames
    cam.nFrames=nFrames

    x_range = np.linspace(onmask_x-drange,onmask_x+drange,Nsteps)
    y_range = np.linspace(onmask_y-drange,onmask_y+drange,Nsteps)

    xs = np.array([])
    ys = np.array([])

    #plt.figure(figsize=(15,10))
    #plt.subplots_adjust(right=1.0,left=0.0,bottom=0.0,top=1.0,hspace=0.0,wspace=0.0)
    i = -1
    #subplot_i = 0
    imgs = np.zeros((len(x_range)*len(y_range),ZWFS.nPx_img,2*ZWFS.nPx_img))
    for yi, y in enumerate(y_range):
        for xi,x in enumerate(x_range):
            i+=1
            print(i+1,"/",Nsteps*Nsteps)
            moveTTM(FAM,np.array([x,y]))
            
            time.sleep(cam.get_exp()*1.1)
            im = ZWFS.getImage()
            imgs[i] = im
            time.sleep(cam.get_exp()*1.1)

            xs = np.append(xs,x)
            ys = np.append(ys,y)
            """
            subplot_i += 1
            plt.subplot(len(x_range),len(y_range),subplot_i)
            #if subplot_i <= len(x_range):
            #    plt.title('('+str(int(round(FAM.get_pos()[0])))+','+str(int(round(FAM.get_pos()[1])))+')',fontsize=10)#\nOuter: '+str(round(sum_outside,1)))
            #elif xi == 0:
            #    plt.title('('+str(int(round(FAM.get_pos()[0])))+','+str(int(round(FAM.get_pos()[1])))+')',fontsize=10)#\nOuter: '+str(round(sum_outside,1)))
            plt.imshow(imdiff)#,vmin=0,vmax=100)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            """
    plt.figure(figsize=(15,10))
    plt.subplots_adjust(right=1.0,left=0.05,bottom=0.0,top=0.95,hspace=0.0,wspace=0.0)
    for i,im in enumerate(imgs):
        plt.subplot(len(x_range),len(y_range),i+1)
        imdiff = im[:,:ZWFS.nPx_img]-im[:,ZWFS.nPx_img:]
        if i+1 <= len(x_range):
            plt.title(str(int(xs[i])),fontsize=8)
        if i % Nsteps == 0:
            plt.ylabel(str(int(ys[i])),fontsize=10)#\nOuter: '+str(round(sum_outside,1)))
        plt.imshow(imdiff)
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(right=1.0,left=0.05,bottom=0.0,top=0.95,hspace=0.0,wspace=0.0)
    plt.show(block=False)
    
    plt.figure(figsize=(15,10))
    plt.subplots_adjust(right=1.0,left=0.0,bottom=0.0,top=1.0,hspace=0.0,wspace=0.0)
    for i,im in enumerate(imgs):
        plt.subplot(len(x_range),len(y_range),i+1)
        #imdiff = im[:,:ZWFS.nPx_img]-im[:,ZWFS.nPx_img:]
        plt.imshow(im)
        plt.xticks([])
        plt.yticks([])
    plt.subplots_adjust(right=1.0,left=0.0,bottom=0.0,top=1.0,hspace=0.0,wspace=0.0)
    plt.show(block=False)

    #max_sum = np.max(sums)
    #onmask_x, onmask_y = xs[np.where(sums==max_sum)][0], ys[np.where(sums==max_sum)][0]
    cam.nFrames = nFrames_orig
    
    return imgs,xs,ys#onmask_x,onmask_y



def make_keck_aperture(normalized=True, with_secondary=True,with_spiders=True, with_segment_gaps=True, gap_padding=1., segment_transmissions=1, return_header=False, return_segments=False):
    pupil_diameter = 10.95 #m actual circumscribed diameter
    actual_segment_flat_diameter = np.sqrt(3)/2 * 1.8 #m actual segment flat-to-flat diameter
    # iris_ao_segment = np.sqrt(3)/2 * .7 mm (~.606 mm)
    actual_segment_gap = 0.05 #m actual gap size between segments
    # (3.5 - (3 D + 4 S)/6 = iris_ao segment gap (~7.4e-17)
    spider_width = 1*5e-2#Value from Sam; Jules value: 0.02450 #m actual strut size
    if normalized: 
        actual_segment_flat_diameter/=pupil_diameter
        actual_segment_gap/=pupil_diameter
        spider_width/=pupil_diameter
        pupil_diameter/=pupil_diameter
    if with_segment_gaps == True:
        segment_gap = actual_segment_gap * gap_padding #padding out the segmentation gaps so they are visible and not sub-pixel
    else:
        segment_gap = 0.
        actual_segment_gap = 0.
    segment_transmissions = 1.

    segment_flat_diameter = actual_segment_flat_diameter - (segment_gap - actual_segment_gap)
    segment_circum_diameter = 2 / np.sqrt(3) * segment_flat_diameter #segment circumscribed diameter

    num_rings = 3 #number of full rings of hexagons around central segment

    segment_positions = make_hexagonal_grid(actual_segment_flat_diameter + actual_segment_gap, num_rings)
    segment_positions = segment_positions.subset(lambda grid: ~(circular_aperture(segment_circum_diameter)(grid) > 0))

    segment = hexagonal_aperture(segment_circum_diameter, np.pi / 2)

    segmented_aperture = make_segmented_aperture(segment, segment_positions, segment_transmissions, return_segments=True)

    segmentation, segments = segmented_aperture
    
    if with_spiders == True:
        spider1 = make_spider_infinite([0, 0], 0, spider_width)
        spider2 = make_spider_infinite([0, 0], 60, spider_width)
        spider3 = make_spider_infinite([0, 0], 120, spider_width)
        spider4 = make_spider_infinite([0, 0], 180, spider_width)
        spider5 = make_spider_infinite([0, 0], 240, spider_width)
        spider6 = make_spider_infinite([0, 0], 300, spider_width)

        def segment_with_spider(segment):
            return lambda grid: segment(grid) * spider1(grid) * spider2(grid) * spider3(grid) * spider4(grid)
        
        segments = [segment_with_spider(s) for s in segments]
    
    contour = make_segmented_aperture(segment, segment_positions)
    conversion=(10.95/(2*12.05/1000))

    central_obscuration_diameter=0.25

    def func(grid):
        #co=circular_aperture(central_obscuration_diameter)
        if with_spiders == True:
            ap = contour(grid) * spider1(grid) * spider2(grid) * spider3(grid)* spider4(grid) * spider3(grid)* spider5(grid) * spider6(grid)  #* co(grid)
        else:
            ap=contour(grid)
        co=circular_aperture(central_obscuration_diameter)(grid)
        if with_secondary == True:
            ap[co==1]=0
        else:
            ap[co==1]=1
        res = (ap )
        #res*=co
        return Field(res, grid)
    if return_segments: 
        return func, segments, segment_positions
    else:
        return func

def match_rotation(telescope_pupil,ang):
    newtelpup = np.copy(telescope_pupil.shaped)
    newtelpup[np.where(telescope_pupil.shaped!=1)]=0
    test=rotate(np.rot90(newtelpup),angle=ang,reshape=False)
    #test=rotate(newtelpup,angle=ang,reshape=False)
    newtelpup_rot=np.copy(test)
    newtelpup_rot[np.where(test<=0.5)]=0
    newtelpup_rot[np.where(test>0.5)]=1

    return newtelpup_rot

"""
def match_onsky(img,keck_aperture,left_1,left_2,right_1,right_2,radius):

    grid_size = radius*2
    pupil_grid = make_pupil_grid(grid_size, 1)
    telescope_pupil = evaluate_supersampled(keck_aperture, pupil_grid, 8)

    return telescope_pupil
"""
def match_onsky(keck_aperture,radius):

    grid_size = radius*2
    pupil_grid = make_pupil_grid(grid_size, 1)
    telescope_pupil = evaluate_supersampled(keck_aperture, pupil_grid, 8)

    return telescope_pupil

def match_segment_positions(segment_positions,r,ang):

    ang = -1*ang

    new_segment_position = []
    for i,sp in enumerate(segment_positions):
        if i == 0:
            continue

        xpos_prerotate = sp[0]*r*2
        ypos_prerotate = sp[1]*r*2

        # rotate(np.rot90(newtelpup),angle=-3)

        # 90 deg clockwise rotation
        xpos_temp = ypos_prerotate + r
        ypos_temp = -1*xpos_prerotate + r

        # -3 deg clockwise rotation
        xpos = (xpos_temp - r)*np.cos(np.deg2rad(ang)) - (ypos_temp - r)*np.sin(np.deg2rad(ang)) + r
        ypos = (xpos_temp - r)*np.sin(np.deg2rad(ang)) + (ypos_temp - r)*np.cos(np.deg2rad(ang)) + r

        new_segment_position.append(np.array([xpos,ypos]))

    return new_segment_position

def map_Keck_segments():
    matching_inds = {'1':1,
        '2':2,
        '3':3,
        '4':4,
        '5':5,
        '6':0,
        '7':8,
        '8':9,
        '9':10,
        '10':11,
        '11':12,
        '12':13,
        '13':14,
        '14':15,
        '15':16,
        '16':17,
        '17':6,
        '18':7,
        '19':21,
        '20':22,
        '21':23,
        '22':24,
        '23':25,
        '24':26,
        '25':27,
        '26':28,
        '27':29,
        '28':30,
        '29':31,
        '30':32,
        '31':33,
        '32':34,
        '33':35,
        '34':18,
        '35':19,
        '36':20}
    return matching_inds

def map_segment_IDs(Keck_seg_ids):
    dict_segment_IDs = map_Keck_segments()
    if type(Keck_seg_ids) == str or type(Keck_seg_ids) == int:
        return np.array([int(dict_segment_IDs[str(Keck_seg_ids)])])
    else:
        mapped_ids = np.array([],dtype=int)
        for segid in Keck_seg_ids:
            mapped_ids = np.append(mapped_ids,int(dict_segment_IDs[str(segid)]))
        return mapped_ids

def findZWFS_onsky(img,guess=[200, 246, 439, 246, 101, -3.1],dpix=50):

    # --------------- Track -----------------------------
    nPx_img = img.shape[0]
    figu, ax = plt.subplots()
    figu.set_size_inches(10, 6)
    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.15,top=0.9)
    #ax.axis([800, 1300, 450, 700])
    #ax.set_aspect("equal")
    plt.title('Find pupil positions')
    plt.imshow(img)
    #======================= SLIDERS ==============================
    axcolor = 'skyblue'
    start_left_1 = guess[0] #989#911#203
    start_left_2 = guess[1] #570#262

    start_right_1 = guess[2] #1157#1083#443
    start_right_2 = guess[3] #574#262

    start_radius = guess[4]#69#108
    start_angle = guess[5]
    print(guess)

    #======================= KECK PUPIL ==============================
    # ---- Telescope and segment apertures
    keck_aperture, keck_aperture_segments, segment_positions = make_keck_aperture(return_segments=True,normalized=True,with_secondary=True,with_spiders=True,with_segment_gaps=True)
    #telescope_pupil = match_onsky(img,keck_aperture,start_left_1,start_left_2,start_right_1,start_right_2,start_radius)
    telescope_pupil = match_onsky(keck_aperture,start_radius)
    keck_pupil = match_rotation(telescope_pupil,start_angle)

    yy,xx=np.where(keck_pupil==0)
    xx1 = xx+start_left_1-start_radius
    yy1 = yy+start_left_2-start_radius

    xx2 = xx+start_right_1-start_radius
    yy2 = yy+start_right_2-start_radius

    keck_grid_left = plt.scatter(xx1,yy1,marker='s',edgecolor='black',facecolors='none',s=1)
    keck_grid_right = plt.scatter(xx2,yy2,marker='s',edgecolor='black',facecolors='none',s=1)

    plt.show(block=False)
    print("Should be showing scatter points")

    #======================= SLIDERS ==============================
    # --- LEFT PUPIL -----
    sl_left_1 = plt.axes([0.05, 0.53, 0.15, 0.03], facecolor=axcolor)
    slider_left_1 = Slider(sl_left_1 , 'X', start_left_1-dpix,start_left_1+dpix, start_left_1)#100,600, start_left_1)#
    sl_left_2 = plt.axes([0.05, 0.48, 0.15, 0.03], facecolor=axcolor)
    slider_left_2 = Slider(sl_left_2, 'Y', start_left_2-dpix,start_left_2+dpix, start_left_2)#500,600, start_left_2)
    # --- RIGHT PUPIL -----
    sl_right_1 = plt.axes([0.75, 0.53, 0.15, 0.03], facecolor=axcolor)
    slider_right_1 = Slider(sl_right_1, 'X',start_right_1-dpix,start_right_1+dpix ,start_right_1)#100,600 ,start_right_1)#
    sl_right_2 = plt.axes([0.75, 0.48, 0.15, 0.03], facecolor=axcolor)
    slider_right_2 = Slider(sl_right_2 , 'Y', start_right_2-dpix, start_right_2+dpix, start_right_2)#500, 600, start_right_2)
    # ---- Pupils Diameter -----
    sl3 = plt.axes([0.35, 0.12, 0.3, 0.03], facecolor=axcolor)
    slider_r = Slider(sl3, 'Radius', start_radius-dpix,start_radius+dpix,start_radius)#65,75,start_radius)
    # ---- Pupils Rotation Angle -----
    sl4 = plt.axes([0.35, 0.05, 0.3, 0.03], facecolor=axcolor)
    slider_ang = Slider(sl4, 'Angle', -90 , 90, start_angle)

    #======================= PLOT ==============================
    def update(val):
        # ----- UPDATE X and Y position -----
        r_left_1 = slider_left_1.val
        r_left_2 = slider_left_2.val
        r_right_1 = slider_right_1.val
        r_right_2 = slider_right_2.val
        r_rad = slider_r.val
        r_ang = slider_ang.val

        #telescope_pupil = match_onsky(img,keck_aperture,r_left_1,r_left_2,r_right_1,r_right_2,r_rad)
        telescope_pupil = match_onsky(keck_aperture,r_rad)
        keck_pupil = match_rotation(telescope_pupil,r_ang)
        yy,xx=np.where(keck_pupil==0)
        xx1 = xx+r_left_1-r_rad
        yy1 = yy+r_left_2-r_rad
        xx2 = xx+r_right_1-r_rad
        yy2 = yy+r_right_2-r_rad

        keck_grid_left.set_offsets(np.transpose(np.array([xx1,yy1])))
        keck_grid_right.set_offsets(np.transpose(np.array([xx2,yy2])))

        figu.canvas.draw_idle()
        
    slider_left_1.on_changed(update)
    slider_left_2.on_changed(update)
    slider_right_1.on_changed(update)
    slider_right_2.on_changed(update)
    slider_r.on_changed(update)
    slider_ang.on_changed(update)

    plt.show(block=True)

    # ---- Final values ---- 
    r_left_1 = int(slider_left_1.val)
    r_left_2 = int(slider_left_2.val)
    r_right_1 = int(slider_right_1.val)
    r_right_2 = int(slider_right_2.val)
    r_rad = int(slider_r.val)
    r_ang = int(slider_ang.val)

    final_vals = [r_left_1,r_left_2,r_right_1,r_right_2,r_rad,r_ang]
    print("FINAL:",[r_left_1,r_left_2,r_right_1,r_right_2,r_rad,r_ang])

    # ----- Telescope Pupil -----
    #telescope_pupil = match_onsky(img,keck_aperture,r_left_1,r_left_2,r_right_1,r_right_2,r_rad)
    #segments_pupil = match_onsky(img,keck_aperture_segments,r_left_1,r_left_2,r_right_1,r_right_2,r_rad)
    telescope_pupil = match_onsky(keck_aperture,r_rad)
    segments_pupil = match_onsky(keck_aperture_segments,r_rad)
    # ----- Filled Telescope Pupil -----
    keck_aperture_filled = make_keck_aperture(return_segments=False,normalized=True,with_segment_gaps=False,with_secondary=True,with_spiders=True)
    telescope_pupil_filled = match_onsky(keck_aperture_filled,r_rad)

    # ---- Match on-sky pupil rotation
    keck_pupil = match_rotation(telescope_pupil,r_ang)
    keck_pupil_filled = match_rotation(telescope_pupil_filled,r_ang)

    # ---- Match on-sky pupil rotation for each segment
    segments_pupil_0 = match_rotation(segments_pupil[0],r_ang)*keck_pupil
    #all_segs = np.copy(segments_pupil_0)

    keck_segments = [segments_pupil_0]
    for i in range(1,len(segments_pupil)):
        segments_pupil_i = match_rotation(segments_pupil[i],r_ang)*keck_pupil
        keck_segments.append(segments_pupil_i)
        #all_segs += segments_pupil_i

    keck_segment_positions = match_segment_positions(segment_positions,r_rad,r_ang)

    # ---- Final displays
    nPx = 2*r_rad
    p = int((r_right_1-r_left_1 - nPx)/2)
    center_x = int(r_left_1+nPx/2 + p)
    center_y =int((r_right_2 +r_left_2)/2)
    nPx_img_y = nPx+2*p
    if nPx_img_y != 238: #NASTY WAY TO FIX BUG IN MFT
        nPx_img_y = 238
    nPx_img_x = nPx_img_y*2

    pupil_img_left = keck_pupil*img[r_left_2-int(nPx/2):r_left_2+int(nPx/2),r_left_1-int(nPx/2):r_left_1+int(nPx/2)]
    fig(pupil_img_left)
    pupil_img_right = keck_pupil*img[r_right_2-int(nPx/2):r_right_2+int(nPx/2),r_right_1-int(nPx/2):r_right_1+int(nPx/2)]
    fig(pupil_img_right)
    if r_left_1 == r_right_1 and r_left_2 == r_right_2:
        img_wfs = img[r_left_2-int(nPx/2):r_left_2+int(nPx/2),r_left_1-int(nPx/2):r_left_1+int(nPx/2)] 
    else:
        img_wfs = img[center_y-int(nPx_img_y/2):center_y+int(nPx_img_y/2),center_x-int(nPx_img_x/2):center_x+int(nPx_img_x/2)]
    fig(img_wfs)

    pupil_ZWFS = np.stack((pupil_img_left,pupil_img_right))
    position_pups = np.array([r_left_1,r_left_2,r_right_1,r_right_2,nPx_img_x,nPx_img_y])

    # ----- Filled Telescope Pupil -----
    #pupil_img_left_filled = keck_pupil_filled*img[r_left_2-int(nPx/2):r_left_2+int(nPx/2),r_left_1-int(nPx/2):r_left_1+int(nPx/2)]
    pupil_img_left_filled = img[r_left_2-int(nPx/2):r_left_2+int(nPx/2),r_left_1-int(nPx/2):r_left_1+int(nPx/2)]
    #pupil_img_right_filled = keck_pupil_filled*img[r_right_2-int(nPx/2):r_right_2+int(nPx/2),r_right_1-int(nPx/2):r_right_1+int(nPx/2)]
    pupil_img_right_filled = img[r_right_2-int(nPx/2):r_right_2+int(nPx/2),r_right_1-int(nPx/2):r_right_1+int(nPx/2)]
    pupil_ZWFS_filled = np.stack((pupil_img_left_filled,pupil_img_right_filled))

    return pupil_ZWFS,position_pups,keck_pupil,keck_segments,keck_segment_positions,pupil_ZWFS_filled,keck_pupil_filled,final_vals

"""
def segment_pup_crop_prep(seg_pup):

    #seg_pup = segs_pup[matching_inds[segID]]
    segys,segxs = np.where(seg_pup==1)
    miny,maxy,minx,maxx=min(segys),max(segys)+1,min(segxs),max(segxs)+1
    Npup = max([maxy-miny,maxx-minx])
    segy,segx = np.mean([miny,maxy]), np.mean([minx,maxx])
    
    y1, y2 = int(segy-Npup/2), int(segy+Npup/2)
    x1, x2 = int(segx-Npup/2), int(segx+Npup/2)
    pad_y1, pad_y2, pad_x1, pad_x2 = 0, 0, 0, 0
    if y1 < 0:
        pad_y1 = int(y1)
        y1 = 0
    if y2 > np.shape(seg_pup)[0]:
        pad_y2 = int(y2 - np.shape(seg_pup)[0])
        y2 = np.shape(seg_pup)[0]
    if x1 < 0:
        pad_x1 = abs(x1)
        x1 = 0
    if x2 > np.shape(seg_pup)[1]:
        pad_x2 = int(x2 - np.shape(seg_pup)[1])
        x2 = np.shape(seg_pup)[1]

    return Npup,segy,segx,y1,y2,x1,x2,pad_y1,pad_y2,pad_x1,pad_x2

def segment_pup_crop(seg_pup,seg_cuts):#segID,segs_pup,matching_inds):

    Npup,segy,segx,y1,y2,x1,x2,pad_y1,pad_y2,pad_x1,pad_x2 = seg_cuts
    seg_pup_cropped = seg_pup[y1:y2,x1:x2]
    seg_pup_cropped_padded = np.pad(seg_pup_cropped,pad_width=((pad_y1,pad_y2),(pad_x1,pad_x2)))

    return seg_pup_cropped_padded, segy, segx, Npup
"""
def segment_pup_crop(seg_pup):

    segys,segxs = np.where(seg_pup==1)
    miny,maxy,minx,maxx=min(segys),max(segys)+1,min(segxs),max(segxs)+1
    Npup = max([maxy-miny,maxx-minx])
    segy,segx = np.mean([miny,maxy]), np.mean([minx,maxx])
    
    y1, y2 = int(segy-Npup/2), int(segy+Npup/2)
    x1, x2 = int(segx-Npup/2), int(segx+Npup/2)
    seg_pup_cropped = seg_pup[y1:y2,x1:x2]

    return seg_pup_cropped, segy, segx, Npup

def generate_map_array(keck_pupil,segs_pup,matching_inds,seg_Zcoeffs=np.ones(36),pad=5,return_segxys=False):
    keck_pupil = np.pad(keck_pupil,pad_width=pad)
    tel_pup_Zmodes = np.zeros((np.shape(keck_pupil)[0],np.shape(keck_pupil)[1]))

    seg_xs = np.array([])
    seg_ys = np.array([])
    for k in matching_inds.keys():
        seg_pup = segs_pup[matching_inds[k]]
        seg_pup = np.pad(seg_pup,pad_width=pad)
        seg_pup_cropped, segy, segx, Npup = segment_pup_crop(seg_pup)
        tel_pup_Zmodes[int(segy-Npup/2):int(segy+Npup/2),int(segx-Npup/2):int(segx+Npup/2)] += seg_Zcoeffs[int(k)-1]*seg_pup_cropped
        if return_segxys == True:
            seg_pup_cropped, segy, segx, Npup = segment_pup_crop(seg_pup)
            seg_xs = np.append(seg_xs,segx)
            seg_ys = np.append(seg_ys,segy)

    if return_segxys == True:
        return tel_pup_Zmodes, seg_xs, seg_ys
    else:
        return tel_pup_Zmodes

def display_segment_map(keck_pupil,segs_pup,matching_inds,seg_Zcoeffs=np.ones(36),pad=5,inverty=False,save=False,vminmax=None,title=None,cbar_label=None,fontsize=12):

    keck_pupil = np.pad(keck_pupil,pad_width=pad)
    tel_pup_Zmodes = np.zeros((np.shape(keck_pupil)[0],np.shape(keck_pupil)[1]))
    plt.figure()
    if title is not None:
        plt.title(title,fontsize=fontsize)
    for k in matching_inds.keys():
        seg_pup = segs_pup[matching_inds[k]]
        seg_pup = np.pad(seg_pup,pad_width=pad)
        seg_pup_cropped, segy, segx, Npup = segment_pup_crop(seg_pup)
        tel_pup_Zmodes[int(segy-Npup/2):int(segy+Npup/2),int(segx-Npup/2):int(segx+Npup/2)] += seg_Zcoeffs[int(k)-1]*seg_pup_cropped
        plt.text(segx,segy,k,va='center',ha='center',fontsize=fontsize)
    if vminmax is not None:
        plt.imshow(tel_pup_Zmodes,vmin=vminmax[0],vmax=vminmax[1])
    else:
        plt.imshow(tel_pup_Zmodes)
    if inverty == True:
        plt.gca().invert_yaxis()
    if cbar_label is not None:
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=fontsize)
        cb.set_label(label=cbar_label,size=fontsize)
    else:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save!=False:
        plt.savefig(save)
    plt.show(block=False)

    return tel_pup_Zmodes

def zernikeBasis_nonCirc(maxZ,tel_pupil):
    zernike2phase = zernikeBasis(maxZ,np.shape(tel_pupil)[0])
    zernike2phase_keck = np.copy(zernike2phase)
    for z in range(np.shape(zernike2phase_keck)[1]):
        zernike2phase_keck[:,z] *= tel_pupil.ravel()
    phase2zernike = np.linalg.pinv(zernike2phase_keck)

    return zernike2phase_keck,phase2zernike

def global_Zmodes(phase_nm,zernike2phase,phase2zernike,tel_pupil):
    # ----- Decompose global Zernike modes -----
    zernike_decomposition = np.dot(phase2zernike,phase_nm.flatten())
    phase_Zmodes = np.zeros((len(zernike_decomposition),np.shape(phase_nm)[0],np.shape(phase_nm)[0]))
    for zi,zcoeff in enumerate(zernike_decomposition):
        z_indiv = np.zeros(np.shape(zernike_decomposition))
        z_indiv[zi] = zcoeff
        phase_z =  np.dot(zernike2phase,z_indiv)
        phase_z = phase_z.reshape((np.shape(phase_nm)[0],np.shape(phase_nm)[0]))
        phase_Zmodes[zi] = phase_z*tel_pupil

    return phase_Zmodes, zernike_decomposition

def remove_global_Zmodes(phase_nm,phase_Zmodes):
    # ----- Subtract global tip/tilt modes -----
    phase_nm_TT_removed = np.copy(phase_nm)
    for mi in range(len(phase_Zmodes)):
        phase_nm_TT_removed -= phase_Zmodes[mi]

    return phase_nm_TT_removed

def segment_Zmodes(phase_map,maxZ,maxSeg,matching_inds,segs_pup,pad=10,display=True):

    phase_map = np.pad(phase_map,pad_width=pad)

    tel_pup_Zmodes = np.zeros((maxZ,np.shape(phase_map)[0],np.shape(phase_map)[1]))
    seg_Zmode_coeffs = np.zeros((maxSeg,maxZ))
    seg_xs, seg_ys = np.array([]), np.array([])

    for k in matching_inds.keys():

        #print("Seg:",k)
        si = int(k)
        if si > maxSeg:
            break

        seg_pup = segs_pup[matching_inds[k]]
        seg_pup = np.pad(seg_pup,pad_width=pad)
        seg_pup_cropped, segy, segx, Npup = segment_pup_crop(seg_pup)
        phase_seg = phase_map[int(segy-Npup/2):int(segy+Npup/2),int(segx-Npup/2):int(segx+Npup/2)]*seg_pup_cropped

        seg_xs = np.append(seg_xs,segx)
        seg_ys = np.append(seg_ys,segy)

        #z2p = zernikeBasis(maxZ,Npup)
        #p2z = np.linalg.pinv(z2p)
        z2p, p2z = zernikeBasis_nonCirc(maxZ,seg_pup_cropped)
        seg_cropped_Zmodes, zernike_decomposition = global_Zmodes(phase_seg,z2p,p2z,seg_pup_cropped)

        #zernike_decomposition = np.dot(p2z,phase_seg.flatten())

        seg_Zmode_coeffs[si-1] = zernike_decomposition

        #seg_cropped_Zmodes = np.zeros((len(zernike_decomposition),Npup,Npup))
        for zi,zcoeff in enumerate(zernike_decomposition):
        #    z_indiv = np.zeros(np.shape(zernike_decomposition))
        #    z_indiv[zi] = zcoeff
        #    phase_seg_z = np.dot(z2p,z_indiv)
        #    phase_seg_z = phase_seg_z.reshape((Npup,Npup))*seg_pup_cropped
        #    seg_cropped_Zmodes[zi] = phase_seg_z
            tel_pup_Zmodes[zi,int(segy-Npup/2):int(segy+Npup/2),int(segx-Npup/2):int(segx+Npup/2)] += seg_cropped_Zmodes[zi]

    if display == True:
        plt.figure(figsize=(12,8))

        plt.subplot(221)
        plt.title('Reconstructed Phase')
        plt.imshow(phase_map)#tel_pup_Zmodes[0])
        plt.colorbar(label='OPD [nm]')

        plt.subplot(222)
        plt.title('Piston')
        plt.imshow(tel_pup_Zmodes[0])
        plt.colorbar(label='OPD [nm]')

        #plt.subplot(223)
        #plt.title('Tilt')
        #plt.imshow(tel_pup_Zmodes[2])
        #plt.colorbar(label='OPD [nm]')

        plt.subplot(212)
        #for k in matching_inds.keys():
        #    plt.plot(int(k),seg_Zmode_coeffs[matching_inds[k],0],'s',color='steelblue')#,label='Piston')
        plt.plot(np.arange(1,maxSeg+1),seg_Zmode_coeffs[:,0],'s',color='steelblue')
        #plt.plot(seg_Zmode_coeffs[:,1],'>',label='Tip')
        #plt.plot(seg_Zmode_coeffs[:,2],'^',label='Tilt')
        plt.xlabel('Keck Segment ID')
        plt.ylabel('Zernike Coefficient [nm]')
        #plt.legend()

        plt.tight_layout()
        plt.show(block=False)

    return tel_pup_Zmodes, seg_Zmode_coeffs, seg_xs, seg_ys

def display_segment_Zmodes(img,phase_map,tel_pup_Zmodes,seg_Zmode_coeffs,previous_coeffs,matching_inds,seg_xs,seg_ys,save=False,poked=None):

    plt.figure(figsize=(12,8))

    plt.subplot(231)
    plt.title('Image')
    plt.imshow(img,cmap='Greys_r')
    plt.colorbar(label='Counts',location='bottom')

    plt.subplot(232)
    plt.title('Reconstructed Phase')
    plt.imshow(phase_map)#tel_pup_Zmodes[0])
    plt.axis('off')
    plt.colorbar(label='OPD [nm]')

    plt.subplot(233)
    plt.title('Piston')
    plt.imshow(tel_pup_Zmodes[0])
    for i,k in enumerate(matching_inds.keys()):
        plt.text(seg_xs[i],seg_ys[i],k,va='center',ha='center')
    plt.axis('off')
    plt.colorbar(label='OPD [nm]')

    colormap = plt.cm.Blues
    colors_pre = [colormap(i) for i in np.linspace(0.1, 1.0, np.shape(previous_coeffs)[0])]
    plt.subplot(212)
    if poked is not None:
        for p in poked:
            plt.axhline(p,linestyle=':',color='black')
    for i in range(np.shape(previous_coeffs)[0]):
        plt.plot(np.arange(1,len(previous_coeffs[i])+1),previous_coeffs[i],'.',color=colors_pre[i])#,label='Piston')
    #    for k in matching_inds.keys():
    #        plt.plot(int(k),previous_coeffs[i,matching_inds[k]],'.',color='steelblue')#,label='Piston')
    #for k in matching_inds.keys():
    #    plt.plot(int(k),seg_Zmode_coeffs[matching_inds[k]],'s',color='red')#,label='Piston')
    plt.plot(np.arange(1,len(seg_Zmode_coeffs)+1),seg_Zmode_coeffs,'s',color='red')#,label='Piston')

    #plt.plot(seg_Zmode_coeffs[:,1],'>',label='Tip')
    #plt.plot(seg_Zmode_coeffs[:,2],'^',label='Tilt')
    plt.xlabel('Keck Segment ID')
    plt.ylabel('Piston [OPD nm]')
    #plt.legend()

    plt.tight_layout()
    if save != False:
        plt.savefig(save)
    plt.show(block=False)

"""
def ReconPhase_Segments(img,matching_inds,keck_segments,keck_pupil,ZWFS,maxZ=1,maxSeg=36):
    ############### RECONSTRUCT PHASE ###############
    print("Reconstructing Phase...")
    ZWFS.reconNonLinear(img)
    print("Done Reconstructing Phase.")
    phase_nm = ZWFS.opd_rec

    ############### Zernike Mode Basis ###############
    zernike2phase = zernikeBasis(3,ZWFS.nPx)
    zernike2phase_keck = np.copy(zernike2phase)
    for z in range(np.shape(zernike2phase_keck)[1]):
        zernike2phase_keck[:,z] *= keck_pupil.ravel()
    phase2zernike = np.linalg.pinv(zernike2phase_keck)

    # ----- Subtract global tip/tilt modes -----
    zernike_decomposition = np.dot(phase2zernike,phase_nm.flatten())
    phase_Zmodes = np.zeros((len(zernike_decomposition),ZWFS.nPx,ZWFS.nPx))
    for zi,zcoeff in enumerate(zernike_decomposition):
        z_indiv = np.zeros(np.shape(zernike_decomposition))
        z_indiv[zi] = zcoeff
        phase_z =  np.dot(zernike2phase_keck,z_indiv)
        phase_z = phase_z.reshape((ZWFS.nPx,ZWFS.nPx))
        phase_Zmodes[zi] = phase_z*keck_pupil

    phase_nm_TT_removed = phase_nm - phase_Zmodes[0] - phase_Zmodes[1] - phase_Zmodes[2]

    ############### SEGMENT ZERNIKE MODE DECOMPOSITION ###############
    tel_pup_Zmodes, seg_Zmode_coeffs,seg_xs,seg_ys = segment_Zmodes(phase_nm_TT_removed,maxZ,maxSeg,matching_inds,keck_segments,pad=10,display=False)

    return phase_nm, phase_nm_TT_removed, tel_pup_Zmodes, seg_Zmode_coeffs, seg_xs, seg_ys
"""
def ReconPhase_Segments(img_mean,zernike2phase_keck,phase2zernike,keck_pupil,matching_inds,keck_segments,ZWFS,maxZ=1,maxSeg=36,subGZ=3):
    ZWFS.reconNonLinear(img_mean)
    phase_nm = ZWFS.opd_rec
    phase_Zmodes, zernike_decomposition = global_Zmodes(phase_nm,zernike2phase_keck,phase2zernike,keck_pupil)
    #phase_PTT_removed = phase_nm - phase_Zmodes[0] - phase_Zmodes[1] - phase_Zmodes[2] - phase_Zmodes[3]
    phase_PTT_removed = phase_nm - np.sum(phase_Zmodes[0:subGZ],axis=0)
    tel_pup_Zmodes, seg_Zmode_coeffs, seg_xs, seg_ys = segment_Zmodes(phase_PTT_removed,maxZ,maxSeg,matching_inds,keck_segments,pad=10,display=False)
    return phase_nm, phase_PTT_removed, seg_Zmode_coeffs

def create_ACS_files(seg_corrections,path):

    f = open(path,'w')
    counter = 0
    for piston in seg_corrections:
        for i in range(3):
            counter+=1
            f.write(str(counter).rjust(3)+' '+str(int(round(piston))).rjust(3)+'\n')
    f.close()
    print('Wrote ACS file to: '+path)

def log_info(savedir,timestamp,comment_text):
    if not os.path.exists(savedir+'LOG_comments.txt'):
        log_file = open(savedir+'LOG_comments.txt','w')
        log_file.write('Timestamp,Comment\n')
        log_file.close()
        print("Created Log File: ",savedir+'LOG_comments.txt')
    #if timestamp == '':
    #    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = open(savedir+'LOG_comments.txt','a')
    log_file.write(timestamp+','+comment_text+'\n')
    log_file.close()
    print("Logged: ",comment_text)

############################################################
############### NEW FUNCTIONS ADDED 20240821 ###############
############################################################

def setupCamera(cam,tint,fps):
    if fps != cam.get_fps():
        cam.set_fps(int(fps))
    if tint != cam.get_exp():
        cam.set_exp(tint)

def TakeNewDark(cam,fw,ZWFS_calib_path,remove_blocking_filter=True,cred2_filter='h'):

    # ----- Take new dark -----
    # Put blocking filter 
    print("Placing Blocking Filter...")
    fw.set_pos('block', block=True)
    while fw.get_named_pos() != 'block':
        continue
    time.sleep(0.5)
    print("Taking Dark...")
    cam.getDark()
    time.sleep(2*cam.get_exp())
    print("Done Taking Dark.")
    # Save Dark:
    dark_filepath = ZWFS_calib_path+'Dark_onsky_'+str(round(cam.get_exp(),3))+'s.npy'
    np.save(dark_filepath,cam.dark)
    print("Dark file saved:",dark_filepath)
    if remove_blocking_filter == True:
        # Remove blocking filter 
        print("Removing Blocking Filter...")
        fw.set_pos(cred2_filter, block=True)#'1550_25nm', block=True)
        while fw.get_named_pos() != cred2_filter:
            continue
        print("Placed ",fw.get_named_pos()," Filter.")
    
    return dark_filepath

def saveOFFmask(img_offMask,ZWFS_calib_path,savedir):
    
    np.save(ZWFS_calib_path+'OffMask_onsky.npy',img_offMask)
    print("Saved:", ZWFS_calib_path+'OffMask_onsky.npy')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(savedir+'OffMask_onsky_'+timestamp+'.npy',img_offMask)
    print("Saved:", savedir+'OffMask_onsky_'+timestamp+'.npy')
    
    return timestamp

def saveRecenterPupils(pupil_tel,position_pups,keck_pupil,keck_segments,keck_segment_positions,pupil_ZWFS_filled,keck_pupil_filled,final_vals,timestamp,ZWFS_calib_path,savedir):

    np.save(ZWFS_calib_path+'pupil_tel_onsky.npy',pupil_tel)
    np.save(savedir+'pupil_tel_onsky_'+timestamp+'.npy',pupil_tel)
    np.save(ZWFS_calib_path+'position_pups_onsky.npy',position_pups)
    np.save(savedir+'position_pups_onsky_'+timestamp+'.npy',position_pups)
    np.save(ZWFS_calib_path+'keck_pupil_onsky.npy',keck_pupil)
    np.save(savedir+'keck_pupil_onsky_'+timestamp+'.npy',keck_pupil)
    np.save(ZWFS_calib_path+'keck_segments_onsky.npy',keck_segments)
    np.save(savedir+'keck_segments_onsky_'+timestamp+'.npy',keck_segments)
    np.save(ZWFS_calib_path+'keck_segment_positions_onsky.npy',keck_segment_positions)
    np.save(savedir+'keck_segment_positions_onsky_'+timestamp+'.npy',keck_segment_positions)
    np.save(ZWFS_calib_path+'pupil_ZWFS_filled_onsky.npy',pupil_ZWFS_filled)
    np.save(savedir+'pupil_ZWFS_filled_onsky_'+timestamp+'.npy',pupil_ZWFS_filled)
    np.save(ZWFS_calib_path+'keck_pupil_filled_onsky.npy',keck_pupil_filled)
    np.save(savedir+'keck_pupil_filled_onsky_'+timestamp+'.npy',keck_pupil_filled)
    np.save(ZWFS_calib_path+'final_vals_onsky.npy',final_vals)
    np.save(savedir+'final_vals_onsky_'+timestamp+'.npy',final_vals)

    print("Saved Recenter Pupils, timestamp:", timestamp)

def loadRecenterPupils(ZWFS_calib_path):
    # ----- Load pupil positions ----
    pupil_tel = np.load(ZWFS_calib_path+'pupil_tel_onsky.npy')
    position_pups = np.load(ZWFS_calib_path+'position_pups_onsky.npy')
    keck_pupil = np.load(ZWFS_calib_path+'keck_pupil_onsky.npy')
    keck_segments = np.load(ZWFS_calib_path+'keck_segments_onsky.npy')
    keck_segment_positions = np.load(ZWFS_calib_path+'keck_segment_positions_onsky.npy')
    pupil_ZWFS_filled = np.load(ZWFS_calib_path+'pupil_ZWFS_filled_onsky.npy')
    keck_pupil_filled = np.load(ZWFS_calib_path+'keck_pupil_filled_onsky.npy')
    final_vals = np.load(ZWFS_calib_path+'final_vals_onsky.npy')

    return pupil_tel,position_pups,keck_pupil,keck_segments,keck_segment_positions,pupil_ZWFS_filled,keck_pupil_filled, final_vals

def logZWFSreconText(ZWFS,LuckyImaging,keepFraction,print_out=True):
    
    ZWFSrecon_text = ' - '+ZWFS.pupilRec+' - '+ZWFS.algo+' - '+str(ZWFS.nIterRec)+' - '+str(ZWFS.doUnwrap)+' - '+str(LuckyImaging)
    if LuckyImaging == True:
        ZWFSrecon_text = ZWFSrecon_text+' - '+str(keepFraction)
    if print_out == True:
        print(' - Pupil Rec - Algo - N iters - unwrap - LuckyImging - keepFraction')
        print(ZWFSrecon_text)

    return ZWFSrecon_text

def takeImageONmask(cam,ZWFS,savedir,nFrames=1,Nimgs=60):

    cam.get(nFrames=1,header=True)
    cam.nFrames = int(nFrames) # 1 for lucky imaging

    # ----- Take data cube -----
    imcube = np.zeros((Nimgs,ZWFS.nPx_img_y,ZWFS.nPx_img_x))
    for i in trange(Nimgs):
        im = ZWFS.getImage()
        imcube[i] = im

    # ----- Save data cube -----
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = 'ZWFS_'+timestamp+'.fits'
    save_fits(imcube,savedir+filename,header_info=cam.header_info)

    return imcube, timestamp

def doLuckyImaging(imcube,LuckyImaging,ZWFS,center_r,keepFraction):

    if LuckyImaging == False:
        img_mean = np.mean(imcube,axis=0)

    elif LuckyImaging == True:
        central_obstruction = makePupil(center_r,ZWFS.nPx_img_y)

        central_sums = np.array([])
        for img in imcube:
            imgR = img[:,ZWFS.nPx_img_y:]
            imgR_cent = imgR*central_obstruction
            central_sums = np.append(central_sums,np.nansum(imgR_cent))

        keepN = int(keepFraction*len(central_sums))
        top_inds_center = np.argpartition(central_sums, -keepN)[-keepN:]
        imcubeLI = imcube[top_inds_center]
        img_mean = np.mean(imcubeLI,axis=0)

    return img_mean

def check_NIRC2_status(savedir,NIRC2,print_output=True):
    if not os.path.exists(savedir+'status.txt'):
        status_file = open(savedir+'status.txt','w')
        status_file.close()
        print("Created Log File: ",savedir+'status.txt')
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    status_file = open(savedir+'status.txt','a')
    status_file.write(ts+':\n')
    status_file.write("- NIRC2 observers: "+NIRC2.get_observers_names()+'\n')
    status_file.write("- NIRC2 object name: "+NIRC2.get_object_name()+'\n')
    status_file.write("- NIRC2 image type: "+NIRC2.get_image_type()+'\n')
    status_file.write("- NIRC2 data saving in: "+'/'.join(NIRC2.get_next_filename().split('/')[:-1])+'/'+'\n')
    status_file.write("- NIRC2 pupil mask: "+NIRC2.get_pupil_mask_name()+'\n')
    status_file.write("- NIRC2 filter: "+NIRC2.get_filters_names()+'\n')
    status_file.write("- NIRC2 exposure time: "+str(NIRC2.get_exposure_time())+'\n')
    status_file.write("- NIRC2 number of coadds: "+str(NIRC2.get_number_of_coadds())+'\n')
    status_file.write("- NIRC2 total exposure time: "+str(NIRC2.get_total_exposure_time())+'\n')
    status_file.write("- NIRC2 window size: "+str(NIRC2.get_roi_size())+'\n')
    status_file.write("- NIRC2 sampmode: "+NIRC2.get_sampling_mode()+'\n')
    status_file.write("\n")
    status_file.close()
    if print_output == True:
        status_file = open(savedir+'status.txt')
        status_lines = np.array(status_file.readlines())
        last_lines = status_lines[np.where(status_lines == ts+':\n')[0][0]:]
        for line in last_lines:
            print(line.strip())
        return last_lines
            
def check_ZWFS_status(savedir,ZWFS,print_output=True):
    if not os.path.exists(savedir+'status.txt'):
        status_file = open(savedir+'status.txt','w')
        status_file.close()
        print("Created Log File: ",savedir+'status.txt')
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    status_file = open(savedir+'status.txt','a')
    status_file.write(ts+':\n')
    status_file.write("- ZWFS model: "+ZWFS.model+'\n')
    status_file.write("- ZWFS wavelength: "+str(ZWFS.wavelength)+'\n')
    status_file.write("- ZWFS diameter: ["+str(ZWFS.diameter_left)+', '+str(ZWFS.diameter_right)+']\n')
    status_file.write("- ZWFS depth: ["+str(ZWFS.depth_left)+', '+str(ZWFS.depth_right)+']\n')
    status_file.write("- ZWFS nPx: "+str(ZWFS.nPx)+'\n')
    status_file.write("- ZWFS Pupil Rec: "+ZWFS.pupilRec+'\n')
    status_file.write("- ZWFS Algo: "+ZWFS.algo+'\n')
    status_file.write("- ZWFS N iters: "+str(ZWFS.nIterRec)+'\n')
    status_file.write("- ZWFS unwrap: "+str(ZWFS.doUnwrap)+'\n')
    status_file.write("\n")
    status_file.close()
    if print_output == True:
        status_file = open(savedir+'status.txt')
        status_lines = np.array(status_file.readlines())
        last_lines = status_lines[np.where(status_lines == ts+':\n')[0][0]:]
        for line in last_lines:
            print(line.strip())
        return last_lines

    

