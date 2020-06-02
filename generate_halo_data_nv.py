#!/usr/bin/env python
# coding: utf-8

# # Load libraries

import pandas as pd
import numpy as np
import MAS_library as MASL
import readfof
import smoothing_library as SL

# # Pick files:

#################################### INPUT ############################################
def generate_data(realization):
    sim = realization
    #raise NotImplementedError("Please update the directory of the data to your system below.")
    #snapshot = '/mnt/ceph/users/fvillaescusa/Quijote/Snapshots/latin_hypercube_HR/%d/snapdir_004/snap_004' % (sim,)
    #snapdir = '/mnt/ceph/users/fvillaescusa/Quijote/Halos/latin_hypercube/%d' % (sim,)
    snapshot = '/projects/QUIJOTE/Snapshots/fiducial_HR/%d/snapdir_004/snap_004' % (sim,)
    snapdir =  '/projects/QUIJOTE/Halos/fiducial_HR/%d' % (sim,)
    snapnum = 4

# parameters for density field
    grid   = 1024  #density field will have grid^3 voxels
    ptypes = [1]   #CDM
    MAS    = 'CIC' #mass assignment scheme
    do_RSD = False #dont do redshift-space distortions
    axis   = 0     #only needed if do_RSD=True

# parameters for smoothing
    BoxSize = 1000.0    #Mpc/h
    R       = 20.0      #Mpc.h
    Filter  = 'Top-Hat' #'Top-Hat' or 'Gaussian'
    threads = 28        #number of openmp threads
#######################################################################################

# # Computing density contrast field:

# compute density field of the snapshot (density constrast d = rho/<rho>-1)
    delta = MASL.density_field_gadget(snapshot, ptypes, grid, MAS, do_RSD, axis)
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

# # Smooth density field:

# smooth the field on a given scale
    W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)
    delta_smoothed = SL.field_smoothing(delta, W_k, threads)

# # Load halo properties:

# read halo catalogue
    z_dict = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
    redshift = z_dict[snapnum]
    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False,
                              swap=False, SFR=False, read_IDs=False)
    pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h  
    mass  = FoF.GroupMass*1e10          #Halo masses in Msun/h   
    vel_h = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s    

# # Find overdensity at each halo:

# interpolate to find the value of the smoothed overdensity field at the position of each halo
# delta_h will contain the value of the smoothed overdensity in the position of each halo
    delta_h = np.zeros(pos_h.shape[0], dtype=np.float32)
    MASL.CIC_interp(delta_smoothed, BoxSize, pos_h, delta_h)

# # Save:

    cur_data = pd.DataFrame({
        'x': pos_h[:, 0],
        'y': pos_h[:, 1],
        'z': pos_h[:, 2],
        'vx': vel_h[:, 0],
        'vy': vel_h[:, 1],
        'vz': vel_h[:, 2],
        'M14': mass/1e14,
        'delta': delta_h
    })

    cur_data.to_hdf('halos_%d.h5' % (sim,), 'df')


