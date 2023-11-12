# Test for uniformly moving tracer/star particles with shear

# Modules
import logging
import numpy as np  # standard Python module for numerics
import sys  # standard Python module to change path
import scripts.utils.athena as athena  # utilities for running Athena++

sys.path.insert(0, "../../vis/python")  # insert path to Python read scripts
import athena_read  # utilities for reading Athena++ data # noqa

athena_read.check_nan_flag = True
logger = logging.getLogger("athena" + __name__[7:])  # set logger name based on module


def read_tab(i, parid=0, base="bin/", pid="pAdvShear"):
    import pandas as pd
    import glob

    parall = []
    tabfiles = glob.glob(
        "{}{}.block*.out3.{:05d}.par{}.tab".format(base, pid, i, parid)
    )
    nblocks = len(tabfiles)
    for nb in range(nblocks):
        tabfile = "{}{}.block{}.out3.{:05d}.par{}.tab".format(base, pid, nb, i, parid)
        par0 = athena_read.partab(tabfile)
        if len(par0) > 0:
            par0.index = par0.pid
        parall.append(par0)
    parall = pd.concat(parall)
    return parall


def prepare(**kwargs):
    """Configure and make the executable."""
    logger.debug("Running test " + __name__)
    athena.configure("mpi", prob="particle_uniform_motion", **kwargs)
    athena.make()


def run(**kwargs):
    """Run the executable."""

    # Construct a list of arguments to the executable.
    arguments = [
        "job/problem_id=pAdvShear",
        "output2/dt=-1",
        "time/ncycle_out=10",
        "time/tlim=2",
        "output3/dt=0.5",
    ]

    # Run the executable.
    athena.mpirun(
        kwargs["mpirun_cmd"],
        kwargs["mpirun_opts"],
        4,
        "particles/athinput.particle_advection_shear",
        arguments,
    )


def analyze():
    """Analyze the output and determine if the test passes."""
    analyze_status = True

    # read history dumps
    h = athena_read.hst("bin/pAdvShear.hst")

    # check mass conservation
    for par in ["p0", "p1"]:
        par_idx = par + "-m"
        pm_idx = "pm" + par[-1] + "-m"
        dm_p = np.mean(np.abs(h[par_idx] - h[par_idx][0]))
        dm_pm = np.mean(np.abs(h[pm_idx] - h[pm_idx][0]))
        dm_ppm = np.mean(np.abs(h[par_idx] - h[pm_idx]))
        logger.info("%s par pm p-pm:   %g %g %g", par, dm_p, dm_pm, dm_ppm)

        if dm_p > 1.0e-16:
            logger.warning("%s particle mass error is too large %g", par, dm_p)
            analyze_status = False
        if dm_pm > 1.0e-16:
            logger.warning("%s particle-mesh mass error is too large %g", par, dm_pm)
            analyze_status = False
        if dm_ppm > 1.0e-16:
            logger.warning("%s P-PM mass error is too large %g", par, dm_ppm)
            analyze_status = False

    # check particles' sheared positions
    i = 4
    dt = 0.5
    for parid in [0, 1]:
        par = read_tab(i, parid=parid)

        vx0 = 1.0
        time = i * dt
        x = np.linspace(-5, 5, 100)
        kappa = np.sqrt(2)
        y = vx0 * (np.cos(kappa * time) - 1) - x * time
        x = vx0 / kappa * np.sin(kappa * time) + x
        Ly = 2

        dythresh = 0.22
        dyexp = (
            np.mod(((par["x2"] - np.interp(par["x1"], x, y)) + dythresh), Ly) - dythresh
        )

        outlier = (dyexp > dythresh) | (dyexp < -dythresh)
        nout = outlier.sum()
        logger.info(
            "p%d -- found %d outliers with offset tolerence %g", parid, nout, dythresh
        )
        if nout > 100:
            logger.warning("p%d  -- more than 100 outliers are found %d", parid, nout)
            analyze_status = False
    return analyze_status
