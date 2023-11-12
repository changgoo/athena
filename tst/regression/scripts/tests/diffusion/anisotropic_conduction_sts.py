# Regression test based on the diffusion of a ring segment by anistropic conduction.
# L1 error in comparison with analytic solution is calculated and stored in history.
# An expected error at different time and resolution is then compared.
# This test employs STS.

# Modules
# (needed for global variables modified in run_tests.py, even w/o athena.run(), etc.)
import scripts.utils.athena as athena  # noqa
import scripts.tests.diffusion.anisotropic_conduction as anisotropic_conduction
import logging

anisotropic_conduction.method = "STS"

# lower bound on convergence rate at final (Nx1=64) asymptotic convergence regime
anisotropic_conduction.logger = logging.getLogger("athena" + __name__[7:])


def prepare(*args, **kwargs):
    anisotropic_conduction.prepare("sts", *args, **kwargs)


def run(**kwargs):
    return anisotropic_conduction.run(**kwargs)


def analyze():
    return anisotropic_conduction.analyze()
