# Regression test based on Newtonian hydro linear wave convergence problem
#
# Runs a linear wave convergence test in 3D including SMR and checks L1 errors (which
# are computed by the executable automatically and stored in the temporary file
# linearwave_errors.dat)

# Modules
import sys
import scripts.utils.athena as athena
sys.path.insert(0, '../../vis/python')


# Prepare Athena++
def prepare(*args, **kwargs):
    athena.configure('cr', 'hdf5', *args,
                     prob='cr_diffusion',
                     coord='cartesian',
                     flux='hllc', **kwargs)
    athena.make()


# Run Athena++
def run(**kwargs):
    # case 1: static diffusion along x direction
    arguments = ['mesh/nx1=256', 'mesh/ix1_bc=outflow', 'mesh/ox1_bc=outflow',
                 'mesh/nx2=4', 'mesh/ix2_bc=periodic', 'mesh/ox2_bc=periodic',
                 'meshblock/nx1=32', 'meshblock/nx2=4', 'problem/direction=0',
                 'problem/v0=0', 'output1/dt=-1', 'time/ncycle_out=10']
    athena.run('cosmic_ray/athinput.cr_diffusion', arguments)

    # case 2: dynamic diffusion along x direction
    arguments = ['mesh/nx1=256', 'mesh/ix1_bc=outflow', 'mesh/ox1_bc=outflow',
                 'mesh/nx2=4', 'mesh/ix2_bc=periodic', 'mesh/ox2_bc=periodic',
                 'meshblock/nx1=32', 'meshblock/nx2=4', 'problem/direction=0',
                 'problem/v0=1', 'output1/dt=-1', 'time/ncycle_out=10']
    athena.run('cosmic_ray/athinput.cr_diffusion', arguments)

    # case 3: static diffusion along y direction
    arguments = ['mesh/nx1=4', 'mesh/ix1_bc=periodic', 'mesh/ox1_bc=periodic',
                 'mesh/nx2=256', 'mesh/ix2_bc=outflow', 'mesh/ox2_bc=outflow',
                 'meshblock/nx1=4', 'meshblock/nx2=32', 'problem/direction=1',
                 'problem/v0=0', 'output1/dt=-1', 'time/ncycle_out=10']
    athena.run('cosmic_ray/athinput.cr_diffusion', arguments)

    # case 4: dynamic diffusion along y direction
    arguments = ['mesh/nx1=4', 'mesh/ix1_bc=periodic', 'mesh/ox1_bc=periodic',
                 'mesh/nx2=256', 'mesh/ix2_bc=outflow', 'mesh/ox2_bc=outflow',
                 'meshblock/nx1=4', 'meshblock/nx2=32', 'problem/direction=1',
                 'problem/v0=1', 'output1/dt=-1', 'time/ncycle_out=10']
    athena.run('cosmic_ray/athinput.cr_diffusion', arguments)

    # case 5: static diffusion along z direction
    arguments = ['mesh/nx1=4', 'mesh/ix1_bc=periodic', 'mesh/ox1_bc=periodic',
                 'mesh/nx2=4', 'mesh/ix2_bc=periodic', 'mesh/ox2_bc=periodic',
                 'mesh/nx3=256', 'mesh/ix3_bc=outflow', 'mesh/ox3_bc=outflow',
                 'meshblock/nx1=4', 'meshblock/nx2=4', 'meshblock/nx3=32',
                 'problem/direction=2', 'problem/v0=0',
                 'output1/dt=-1', 'time/ncycle_out=10']
    athena.run('cosmic_ray/athinput.cr_diffusion', arguments)

    # case 6: dynamic diffusion along z direction
    arguments = ['mesh/nx1=4', 'mesh/ix1_bc=periodic', 'mesh/ox1_bc=periodic',
                 'mesh/nx2=4', 'mesh/ix2_bc=periodic', 'mesh/ox2_bc=periodic',
                 'mesh/nx3=256', 'mesh/ix3_bc=outflow', 'mesh/ox3_bc=outflow',
                 'meshblock/nx1=4', 'meshblock/nx2=4', 'meshblock/nx3=32',
                 'problem/direction=2', 'problem/v0=1',
                 'output1/dt=-1', 'time/ncycle_out=10']
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

    # check absolute error
    if data[0][9] > 8e-5:
        print("error in static diffusion along x: ", data[0][9])
        return False

    if data[1][9] > 9e-5:
        print("error in dynamic diffusion along x: ", data[1][9])
        return False

    if data[2][9] > 8e-5:
        print("error in static diffusion along y: ", data[2][9])
        return False

    if data[3][9] > 9e-5:
        print("error in dynamic diffusion along y: ", data[3][9])
        return False

    if data[4][9] > 8e-5:
        print("error in static diffusion along z: ", data[4][9])
        return False

    if data[5][9] > 9e-5:
        print("error in dynamic diffusion along z: ", data[5][9])
        return False

    return True