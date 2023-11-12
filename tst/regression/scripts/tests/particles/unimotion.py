# Test for uniformly moving tracer particles with multiple containers

# Modules
import logging
import numpy as np  # standard Python module for numerics
import sys  # standard Python module to change path
import scripts.utils.athena as athena  # utilities for running Athena++

sys.path.insert(0, "../../vis/python")  # insert path to Python read scripts
import athena_read  # utilities for reading Athena++ data # noqa

athena_read.check_nan_flag = True
logger = logging.getLogger("athena" + __name__[7:])  # set logger name based on module


def prepare(**kwargs):
    """Configure and make the executable."""
    logger.debug("Running test " + __name__)
    athena.configure("mpi", prob="particle_uniform_motion", **kwargs)
    athena.make()


def run(**kwargs):
    """Run the executable."""

    # Construct a list of arguments to the executable.
    arguments = [
        "job/problem_id=pUniMotion_N2",
        "output2/dt=-1",
        "time/ncycle_out=10",
        "problem/npartot=100",
    ]

    # Run the executable.
    athena.mpirun(
        kwargs["mpirun_cmd"],
        kwargs["mpirun_opts"],
        4,
        "particles/athinput.particle_uniform_motion",
        arguments,
    )

    # Construct a list of arguments to the executable.
    arguments = [
        "job/problem_id=pUniMotion_N3",
        "output2/dt=-1",
        "time/ncycle_out=10",
        "problem/npartot=1000",
    ]

    # Run the executable.
    athena.mpirun(
        kwargs["mpirun_cmd"],
        kwargs["mpirun_opts"],
        4,
        "particles/athinput.particle_uniform_motion",
        arguments,
    )

    # Construct a list of arguments to the executable.
    arguments = [
        "job/problem_id=pUniMotion_N4",
        "output2/dt=-1",
        "time/ncycle_out=10",
        "problem/npartot=10000",
    ]

    # Run the executable.
    athena.mpirun(
        kwargs["mpirun_cmd"],
        kwargs["mpirun_opts"],
        4,
        "particles/athinput.particle_uniform_motion",
        arguments,
    )


def analyze():
    """Analyze the output and determine if the test passes."""
    analyze_status = True

    # read history dumps
    h1 = athena_read.hst("bin/pUniMotion_N2.hst")
    h2 = athena_read.hst("bin/pUniMotion_N3.hst")
    h3 = athena_read.hst("bin/pUniMotion_N4.hst")

    errg = []
    errp0 = []
    errp1 = []
    for h in [h1, h2, h3]:
        errg.append(h["drhog"][-1])
        errp0.append(h["drhop"][0])
        errp1.append(h["drhop"][-1])

    logger.info("gas  final:   %g %g %g", errg[0], errg[1], errg[2])
    logger.info("star initial: %g %g %g", errp0[0], errp0[1], errp0[2])
    logger.info("star final:   %g %g %g", errp1[0], errp1[1], errp1[2])

    for i, err in enumerate(errg):
        if err > 1.0e-16:
            logger.warning(
                "np=%d: gas density error is too large %g", 10 ** (i + 2), err
            )
            analyze_status = False

    for i, erri, errf in zip(range(3), errp0, errp1):
        if np.abs(erri - errf) > 1.0e-16:
            logger.warning(
                "np=%d: initial - final is too large %g", 10 ** (i + 2), erri - errf
            )
            analyze_status = False
        if errf > 2.0e-2 * np.sqrt(10 ** (2 - i)):
            logger.warning(
                "np=%d: final particle error is too large %g", 10 ** (i + 2), errf
            )
            analyze_status = False

    return analyze_status
