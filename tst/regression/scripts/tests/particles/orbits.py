# Test for particle integrator under different sources

# Modules
import logging
import numpy as np                           # standard Python module for numerics
import sys                                   # standard Python module to change path
import scripts.utils.athena as athena        # utilities for running Athena++
sys.path.insert(0, '../../vis/python')       # insert path to Python read scripts
import athena_read                           # utilities for reading Athena++ data # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])  # set logger name based on module


def prepare(**kwargs):
    """Configure and make the executable. """
    logger.debug('Running test ' + __name__)
    athena.configure('mpi', prob='particle_orbit', **kwargs)
    athena.make()


def run(**kwargs):
    """Run the executable. """

    # Construct a list of arguments to the executable.
    arguments = []

    # Run the executable.
    athena.mpirun(kwargs['mpirun_cmd'], kwargs['mpirun_opts'], 4,
                  'particles/athinput.particle_orbit', arguments)

    athena.mpirun(kwargs['mpirun_cmd'], kwargs['mpirun_opts'], 4,
                  'particles/athinput.particle_epicycle', arguments)


def analyze():
    """Analyze the output and determine if the test passes. """
    analyze_status = True

    # read history dumps
    h1 = athena_read.hst("bin/par_kep.hst")
    h2 = athena_read.hst("bin/par_epi.hst")

    err = []
    for h in [h1, h2]:
        for Ep in ['Ep1', 'Ep2']:
            E = h[Ep][2:-1]
            dE = E-E[0]
            err.append(np.abs(dE).mean())

    logger.info("%g %g %g %g", err[0], err[1], err[2], err[3])

    if err[0] > 2.00E-6:
        logger.warning("circular orbit energy error is too large %g", err[0])
        analyze_status = False
    if err[1] > 2.00E-5:
        logger.warning("ellipse orbit energy error is too large %g", err[1])
        analyze_status = False
    if err[2] > 2.00E-6:
        logger.warning("epicyclic orbit energy error is too large %g", err[2])
        analyze_status = False
    if err[3] > 3.00E-6:
        logger.warning("off-center epicyclic orbit energy error is too large %g", err[3])
        analyze_status = False

    return analyze_status
