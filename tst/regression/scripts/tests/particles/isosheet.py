# Test for gas-particle with gravity using self-gravitating isothermal sheet

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
    athena.configure('p', 'mpi', 'fft',
                     grav='blockfft',
                     eos='isothermal',
                     prob='particle_isosheet',
                     **kwargs)
    athena.make()


def run(**kwargs):
    """Run the executable. """

    # Construct a list of arguments to the executable.
    arguments = ['job/problem_id=IsoSheet_N3',
                 'output2/dt=-1',
                 'time/ncycle_out=100',
                 'particle1/npartot=1000',
                 ]

    # Run the executable.
    athena.mpirun(kwargs['mpirun_cmd'], kwargs['mpirun_opts'], 4,
                  'particles/athinput.particle_isosheet', arguments)

    # Construct a list of arguments to the executable.
    arguments = ['job/problem_id=IsoSheet_N4',
                 'output2/dt=-1',
                 'time/ncycle_out=100',
                 'particle1/npartot=10000',
                 ]

    # Run the executable.
    athena.mpirun(kwargs['mpirun_cmd'], kwargs['mpirun_opts'], 4,
                  'particles/athinput.particle_isosheet', arguments)

    # # Construct a list of arguments to the executable.
    # arguments = ['job/problem_id=IsoSheet_N5',
    #              'output2/dt=-1',
    #              'time/ncycle_out=100',
    #              'particle1/npartot=100000',
    #              ]
    #
    # # Run the executable.
    # athena.mpirun(kwargs['mpirun_cmd'], kwargs['mpirun_opts'], 4,
    #               'particles/athinput.particle_isosheet', arguments)


def analyze():
    """Analyze the output and determine if the test passes. """
    analyze_status = True

    # read history dumps
    h1 = athena_read.hst("bin/IsoSheet_N3.hst")
    h2 = athena_read.hst("bin/IsoSheet_N4.hst")
    # h3 = athena_read.hst("bin/IsoSheet_N5.hst")

    err = []
    for drho in ['drhog', 'drhop']:
        for h in [h1, h2]:
            err.append(h[drho].mean())

    logger.info("gas: %g %g", err[0], err[1])
    logger.info("star: %g %g", err[2], err[3])

    if err[0] > 1.00E-5*np.sqrt(10):
        logger.warning("np=1e3: gas density error is too large %g", err[0])
        analyze_status = False
    if err[1] > 1.00E-5:
        logger.warning("np=1e4: gas density error is too large %g", err[1])
        analyze_status = False
    # if err[2] > 1.00E-5/np.sqrt(10):
    #     logger.warning("np=1e5: gas density error is too large %g", err[2])
        analyze_status = False
    if err[2] > 2.e-4*np.sqrt(10):
        logger.warning("np=1e3: star density error is too large %g", err[2])
        analyze_status = False
    if err[3] > 2.e-4:
        logger.warning("np=1e4: star density error is too large %g", err[3])
        analyze_status = False
    # if err[5] > 2.e-4/np.sqrt(10):
    #     logger.warning("np=1e5: star density error is too large %g", err[5])
    #     analyze_status = False

    return analyze_status
