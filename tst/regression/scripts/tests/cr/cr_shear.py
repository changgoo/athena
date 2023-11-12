# Regression test for the propagation of cosmic rays in the presence of shear

# Modules
import numpy as np
import sys
import scripts.utils.athena as athena

sys.path.insert(0, "../../vis/python")


# Prepare Athena++
def prepare(*args, **kwargs):
    athena.configure(
        "cr", "hdf5", "b", *args, prob="cr_shear", coord="cartesian", **kwargs
    )
    athena.make()


# Run Athena++
def run(**kwargs):
    # Both streaming and diffusion and advection are on, magnetic field along x direction
    arguments = [
        "mesh/nx1=256",
        "mesh/ix1_bc=shear_periodic",
        "mesh/ox1_bc=shear_periodic",
        "mesh/nx2=64",
        "mesh/ix2_bc=periodic",
        "mesh/ox2_bc=periodic",
        "meshblock/nx1=256",
        "meshblock/nx2=64",
        "time/tlim=0.02",
        "cr/vmax=200",
        "cr/sigma=10",
        "orbital_advection/Omega0=0.5",
        "orbital_advection/qshear=1",
        "problem/direction=0",
        "problem/B0=1",
        "problem/offset1=0.5",
        "problem/cells=8",
    ]
    athena.run("cosmic_ray/athinput.cr_shear", arguments)


# Analyze outputs
def analyze():
    # Solution calculated in the absence of shear
    filename1 = "data/ref_cr_solutions/ShearTest.txt"
    Ec = np.genfromtxt(filename1, usecols=1, skip_header=1)

    # Solution of the test with shearing-periodic BCs
    filename2 = "bin/cr_energy_profile.dat"
    x = np.genfromtxt(filename2, usecols=0, skip_header=1)
    Ec_shear = np.genfromtxt(filename2, usecols=1, skip_header=1)

    err = np.abs(Ec - Ec_shear)

    # check absolute error
    if any(err > 4e-6):
        index = np.where((err) > 4e-6)[0]
        print("error in cosmic-ray propagation =", err[index], "in x =", x[index])
        return False

    return True
