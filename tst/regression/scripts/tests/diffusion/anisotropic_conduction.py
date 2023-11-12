# Regression test based on the diffusion of a ring segment by anistropic conduction.
# L1 error in comparison with analytic solution is calculated and stored in history.
# An expected error at different time and resolution is then compared.

# Modules
import logging
import numpy as np
import sys
import scripts.utils.athena as athena

sys.path.insert(0, "../../vis/python")
import athena_read  # noqa

athena_read.check_nan_flag = True
logger = logging.getLogger("athena" + __name__[7:])  # set logger name based on module

resolution_range = [32, 64, 128]
error_tol_high = 4.0e-3
method = "EXPLICIT"


def prepare(*args, **kwargs):
    logger.debug("Running test " + __name__)
    athena.configure("b", *args, prob="ring_diff", **kwargs)
    athena.make()


def run(**kwargs):
    for i in resolution_range:
        arguments = [
            "output1/dt=1.e-3",
            "output2/dt=-1",  # disable .hdf5 outputs
            "time/tlim=0.1",
            "time/ncycle_out=0",
            "mesh/nx1=" + repr(i),
            "mesh/nx2=" + repr(i),
            "mesh/nx3=1",
            "meshblock/nx1=" + repr(i),
            "meshblock/nx2=" + repr(i),
            "meshblock/nx3=1",
            "job/problem_id=RingDiff-{}".format(i),
        ]
        athena.run("mhd/athinput.ring_diff", arguments)


def analyze():
    analyze_status = True

    for nx in resolution_range:
        logger.info(
            "[Anisotropic Ring Diffusion {}]: " "Mesh size {} x {} x {}".format(
                method, nx, nx, 1
            )
        )
        filename = "bin/RingDiff-{}.hst".format(nx)
        hst_data = athena_read.hst(filename)
        tt = hst_data["time"][1:]
        l1_error = hst_data["L1"][1:]
        l1_tol = error_tol_high * (nx / 256) ** (-0.5) * (tt / 0.01) ** 0.4

        logger.info(
            "[Anisotropic Ring Diffusion {}]: "
            "L1 error for nx={} at t={} is {}"
            " ".format(method, nx, tt[-1], l1_error[-1])
        )

        error_tol = l1_tol > l1_error
        if np.all(error_tol):
            logger.info("passed")
        else:
            logger.warning(
                "[Anisotropic Ring Diffusion {}]: " "L1 error is too large -- ".format(
                    method
                ),
                error_tol,
            )
            analyze_status = False

    return analyze_status
