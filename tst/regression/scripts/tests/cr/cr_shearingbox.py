# Regression test based on Newtonian hydro linear wave convergence problem
#
# Runs a linear wave convergence test in 3D including SMR and checks L1 errors (which
# are computed by the executable automatically and stored in the temporary file
# linearwave_errors.dat)

# Modules
import numpy as np
import math
import sys
import scripts.utils.athena as athena
import scripts.utils.comparison as comparison
import os
sys.path.insert(0, '../../vis/python')

# Prepare Athena++
def prepare(*args, **kwargs):
  athena.configure('cr', 'hdf5', *args,
      prob='cr_shear',
      coord='cartesian',
      **kwargs)
  athena.make()

# Run Athena++
def run(**kwargs):
  #Both streaming and siffusion and advection are on, magneticfield along x direction
  arguments = ['mesh/nx1=512', 'mesh/ix1_bc=shear_periodic', 'mesh/ox1_bc=shear_periodic', 
  'mesh/nx2=128', 'mesh/ix2_bc=periodic', 'mesh/ox2_bc=periodic', 'meshblock/nx1=512', 
  'meshblock/nx2=128', 'time/tlim=0.8','cr/vmax=100', 'cr/sigma=10','orbital_advection/Omega0=0.2',
  'orbital_advection/qshear=1','problem/direction=0', 'problem/B0=1','problem/offset1 = 0.5', 
  'problem/offset2 = 0.1', 'problem/cells=8']
  athena.run('cosmic_ray/athinput.cr_diffusion', arguments)

# Analyze outputs
def analyze():
  filename = 'bin/diffusion_error.dat'
  data = []
  with open(filename, 'r') as f:
    raw_data = f.readlines()
    for line in raw_data:
      if line.split()[0][0] == '#':
        continue
      data.append([float(val) for val in line.split()])


  # check absolute error and convergence of all three waves
  #regime1
  if data[0][8] > 7e-5:
    print ("error in static diffusion along x: ", data[0][8])
    return False

    #regime2
  if data[1][8] > 8e-5:
    print ("error in dynamic diffusion along x: ", data[1][8])
    return False

  if data[2][8] > 7e-5:
    print ("error in static diffusion along y: ", data[2][8])
    return False

    #regime2
  if data[3][8] > 8e-5:
    print ("error in dynamic diffusion along y: ", data[3][8])
    return False


  if data[4][8] > 7e-5:
    print ("error in static diffusion along z: ", data[4][8])
    return False

    #regime2
  if data[5][8] > 8e-5:
    print ("error in dynamic diffusion along z: ", data[5][8])
    return False

  return True
