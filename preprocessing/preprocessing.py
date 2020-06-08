import numpy as np
import sys,os,time
import MAS_library as MASL
import readgadget, readfof
import smoothing_library as SL

######################################## INPUT #################################################
# snapshot and halo catalogues parameters
snapshot = '/projects/QUIJOTE/Snapshots/fiducial_HR/0/snapdir_004/snap_004'
snapdir  = '/projects/QUIJOTE/Halos/fiducial_HR/0'
snapnum  = 4
ptype    = [1] #read only dark matter particles

# density field parameters
grid     = 1024 #will create a density field with grid^3 voxels
MAS      = 'CIC'

# smoothing parameters
R        = 20.0 #Mpc/h
Filter   = 'Top-Hat'
threads  = 1
################################################################################################

# read the box size, the number of particles, the masses of them and the redshift (time)
start2 = time.time()
start  = time.time()
header   = readgadget.header(snapshot)
BoxSize  = header.boxsize/1e3  #Mpc/h
Nall     = header.nall         #Total number of particles
Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
redshift = header.redshift     #redshift of the snapshot
print('\nBoxSize             = %.3f Mpc/h'%BoxSize)
print('Number of particles = %d'%Nall[ptype])
print('Masses of particles = %.3e Msun/h'%Masses[ptype])
print('redshift            = %.3f'%redshift)
print('snapshot header read in %.3f seconds\n'%(time.time() - start))

# read the positions of the particles in the snapshot
start = time.time()
pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
print('particle positions read in %.3f seconds\n'%(time.time() - start))

# define the grid containing the density field
density_field = np.zeros((grid,grid,grid), dtype=np.float32)

# compute density field
start = time.time()
MASL.MA(pos,density_field,BoxSize,MAS)
density_field /= np.mean(density_field, dtype=np.float64);  density_field -= 1.0
print('%.3e < density field < %.3e'%(np.min(density_field), np.max(density_field)))
print('< density field > = %.3e'%np.mean(density_field, dtype=np.float64))
print('Density field constructed in %.3f seconds\n'%(time.time()-start))

# smooth the field on a given scale
start = time.time()
W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)
density_field_smoothed = SL.field_smoothing(density_field, W_k, threads)
print('Smoothing done in %.3f seconds\n'%(time.time()-start))

# read halo catalogue
start = time.time()
FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False,
                          swap=False, SFR=False, read_IDs=False)
pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h  
mass  = FoF.GroupMass*1e10          #Halo masses in Msun/h   
vel_h = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s    
print('Halo catalogue read in %.3f seconds\n'%(time.time()-start))

# interpolate to find the value of the smoothed overdensity field at the position of each halo
# delta_h will contain the value of the smoothed overdensity in the position of each halo
start = time.time()
delta_h = np.zeros(pos_h.shape[0], dtype=np.float32)
MASL.CIC_interp(density_field_smoothed, BoxSize, pos_h, delta_h)
print('Smoothed density field interpolated in %.3f seconds\n'%(time.time()-start))
print('Global time = %.3f seconds'%(time.time()-start2))
