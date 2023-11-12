# radiative snr test (Kim & Ostriker 2015)
# test based on tigress cooling (or Sutherland & Dopita) with correct mu

# Modules
import numpy as np  # standard Python module for numerics
import sys  # standard Python module to change path
import scripts.utils.athena as athena  # utilities for running Athena++

sys.path.insert(0, "../../vis/python")  # insert path to Python read scripts
import athena_read  # utilities for reading Athena++ data # noqa

athena_read.check_nan_flag = True  # raise exception when encountering NaNs

# tolerance in relative error
err_tol = 0.05


def prepare(**kwargs):
    """
    Configure and make the executable.

    This function is called first. It is responsible for calling the configure script and
    make to create an executable. It takes no inputs and produces no outputs.
    """

    # Configure as though we ran
    athena.configure(prob="radiative_snr", **kwargs)

    # Call make as though we ran
    #     make clean
    #     make
    # from the athena/ directory.
    athena.make()


def run(**kwargs):
    # Run the executable.
    arguments_def = [
        "time/ncycle_out=10",
        "time/dt_diagnostics=-1",
        "time/integrator=rk2",
        "cooling/coolftn=tigress",
        "hydro/neighbor_flooring=true",
        "output2/dt=-1",  # turn off hdf5 outputs
        "output3/dt=-1",  # turn off hdf5 outputs
        "time/tlim=0.05",  # stop after shell formation
    ]

    # Run Athena++ as though we called
    # test for enrolled cooling
    # arguments = arguments_def + \
    #     ['cooling/cooling=enroll',
    #      'job/problem_id=snr_enroll']
    # athena.run('cooling/athinput.radiative_snr', arguments)

    # test for operator split cooling (default)
    arguments = arguments_def + [
        "cooling/cooling=op_split",
        "job/problem_id=snr_op_split",
    ]
    athena.run("cooling/athinput.radiative_snr", arguments)
    # No return statement/value is ever required from run(), but returning anything other
    # than default None will cause run_tests.py to skip executing the optional Lcov cmd
    # immediately after this module.run() finishes, e.g. if Lcov was already invoked by:
    # athena.run('cooling/athinput.cooling_test', arguments, lcov_test_suffix='mb_1')
    return "skip_lcov"


def analyze():
    # Analyze the output and determine if the test passes.

    # Read in reference data. Which is in code units
    # This is a tabulated result of the analytic exponential growth expected for this
    # eigenmode perturbation with the growth rate output from the code's calculation

    # initialize analyze_status to True (remains true if all tests are passed)
    analyze_status = True

    # loop over cooling sovers
    # Read in the data produced during this test.
    filename = "bin/snr_op_split.hst"
    h = athena_read.hst(filename)
    # define shell formation using hot gas mass
    isf = h["M_hot"].argmax()
    tsf = h["time"][isf]
    Msf = h["M_hot"][isf] * 0.03526938739965538  # to M_sun
    psf = h["pr"][isf] * 0.03526938739965538  # to M_sun*km/s

    if (np.abs(tsf / 0.0344 - 1) > err_tol) or np.isnan(tsf):
        print("shell formation time is off: ", tsf)
        analyze_status = False

    if (np.abs(Msf / 1294 - 1) > err_tol) or np.isnan(Msf):
        print("Mhot at shell formation is off: ", Msf)
        analyze_status = False
    if (np.abs(psf / 1.788e5 - 1) > err_tol) or np.isnan(psf):
        print("momentum at shell formation is off: ", psf)
        analyze_status = False

    return analyze_status
