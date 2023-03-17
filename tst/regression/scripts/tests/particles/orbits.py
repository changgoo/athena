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
    arguments = dict(kep_vl2=['job/problem_id=par_kep_vl2',
                              'output1/data_format=%12.10e',
                              'time/integrator=vl2'],
                     kep_rk2=['job/problem_id=par_kep_rk2',
                              'output1/data_format=%12.10e',
                              'time/integrator=rk2'],
                     epi_vl2=['job/problem_id=par_epi_vl2',
                              'output1/data_format=%12.10e',
                              'time/integrator=vl2'],
                     epi_rk2=['job/problem_id=par_epi_rk2',
                              'output1/data_format=%12.10e',
                              'time/integrator=rk2'])
    for test, arg in arguments.items():
        if 'kep' in test:
            finput = 'particles/athinput.particle_orbit'
        elif 'epi' in test:
            finput = 'particles/athinput.particle_epicycle'
        # Run the executable.
        athena.mpirun(kwargs['mpirun_cmd'], kwargs['mpirun_opts'], 4,
                      finput, arg)


def analyze():
    """Analyze the output and determine if the test passes. """
    analyze_status = True

    # read history dumps
    h1 = athena_read.hst("bin/par_kep_vl2.hst")
    h2 = athena_read.hst("bin/par_epi_vl2.hst")
    h3 = athena_read.hst("bin/par_kep_rk2.hst")
    h4 = athena_read.hst("bin/par_epi_rk2.hst")

    err = []
    for h in [h1, h2, h3, h4]:
        for Ep in ['Ep1', 'Ep2']:
            E = h[Ep][2:-1]
            dE = E-E[0]
            err.append(np.abs(dE).mean())

    logger.info("vl2: %g %g %g %g", err[0], err[1], err[2], err[3])
    logger.info("rk2: %g %g %g %g", err[4], err[5], err[6], err[7])

    # vl2
    if err[0] > 2.00E-6:
        logger.warning("vl2: circular orbit energy error is too large %g", err[0])
        analyze_status = False
    if err[1] > 1.50E-5:
        logger.warning("vl2: ellipse orbit energy error is too large %g", err[1])
        analyze_status = False
    if err[2] > 2.50E-6:
        logger.warning("vl2: epicyclic orbit energy error is too large %g", err[2])
        analyze_status = False
    if err[3] > 2.50E-6:
        logger.warning("vl2: off-center epicyclic orbit energy error is too large %g",
                       err[3])
        analyze_status = False
    # rk2
    if err[4] > 4.00E-6:
        logger.warning("rk2: circular orbit energy error is too large %g", err[0])
        analyze_status = False
    if err[5] > 6.00E-5:
        logger.warning("rk2: ellipse orbit energy error is too large %g", err[1])
        analyze_status = False
    if err[6] > 2.50E-6:
        logger.warning("rk2: epicyclic orbit energy error is too large %g", err[2])
        analyze_status = False
    if err[7] > 1.50E-5:
        logger.warning("rk2: off-center epicyclic orbit energy error is too large %g",
                       err[3])
        analyze_status = False

    return analyze_status
