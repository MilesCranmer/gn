import numpy as np
import sys,os,time
import MAS_library as MASL
import readgadget

######################################## INPUT #################################################
# snapshot parameters
snapshot = '/projects/QUIJOTE/Snapshots/fiducial_HR/0/snapdir_004/snap_004'
ptype    = [1] #read only dark matter particles

# density field parameters
grid     = 1024 #will create a density field with grid^3 voxels
MAS      = 'CIC'
################################################################################################

# read the box size, the number of particles, the masses of them and the redshift (time)
header   = readgadget.header(snapshot)
BoxSize  = header.boxsize/1e3  #Mpc/h
Nall     = header.nall         #Total number of particles
Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
redshift = header.redshift     #redshift of the snapshot
print('\nBoxSize             = %.3f Mpc/h'%BoxSize)
print('Number of particles = %d'%Nall[ptype])
print('Masses of particles = %.3e Msun/h'%Masses[ptype])
print('redshift            = %.3f\n'%redshift)

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
print('density field constructed')
print('%.3e < density field < %.3e'%(np.min(density_field), np.max(density_field)))
print('< density field > = %.3e'%np.mean(density_field, dtype=np.float64))
print('Time taken = %.3f\n'%(time.time()-start))
