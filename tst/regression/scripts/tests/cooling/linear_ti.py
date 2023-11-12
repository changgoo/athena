# Regression test for the growth of a thermal instability with initial density of
# nH = 5 cm^-3 with a perturbation amplitude of 1e-4. Eigenmode perturbation taken
# from Jennings & Li 2020 (https://arxiv.org/pdf/2012.05252.pdf). This regression
# test uses the TIGRESS classic cooling function with an Euler integration scheme.

# Modules
import numpy as np  # standard Python module for numerics
import sys  # standard Python module to change path
import scripts.utils.athena as athena  # utilities for running Athena++

sys.path.insert(0, "../../vis/python")  # insert path to Python read scripts
import athena_read  # utilities for reading Athena++ data # noqa

athena_read.check_nan_flag = True  # raise exception when encountering NaNs

# amplitude of eigenmode linear instability
_amp = 1.0e-4
# resolutions to be tested
resolution_range = [32, 64, 128]
# cooling function to be used
coolfnc = "tigress"
# tolerance in relative error for each resolution
error_rel_tols = [0.008, 0.004, 0.001]

# tolerance in convergance rate
rate_tols = [1.8, 1.9]


def prepare(**kwargs):
    """
    Configure and make the executable.

    This function is called first. It is responsible for calling the configure script and
    make to create an executable. It takes no inputs and produces no outputs.
    """

    # Configure as though we ran
    #     python configure.py -hdf5 --prob=TI_test
    athena.configure(prob="linear_ti_test", **kwargs)

    # Call make as though we ran
    #     make clean
    #     make
    # from the athena/ directory.
    athena.make()


def run(**kwargs):
    # Run the executable.
    # for enrolled cooling
    for N in resolution_range:
        arguments = [
            "job/problem_id=linear_ti_test-euler-{}".format(N),
            "cooling/cooling=enroll",
            "cooling/coolftn=" + coolfnc,
            "cooling/cfl_cool=0.5",
            "cooling/cfl_cool_sub=0.5",
            "mesh/nx1=" + repr(N),
            "mesh/x2min=" + repr(-0.00625 * 32 / N),
            "mesh/x2max=" + repr(0.00625 * 32 / N),
            "mesh/x3min=" + repr(-0.00625 * 32 / N),
            "mesh/x3max=" + repr(0.00625 * 32 / N),
            "problem/nH_0=5.0",
            "problem/kn=1",
            "problem/alpha={}".format(_amp),
        ]

        # Run Athena++ as though we called
        # ./athena -i ../inputs/cooling/athinput.thermal_instability_test
        athena.run("cooling/athinput.linear_ti_test", arguments)

    # for operater split cooling
    for N in resolution_range:
        arguments = [
            "job/problem_id=linear_ti_test-op-{}".format(N),
            "cooling/cooling=op_split",
            "cooling/coolftn=" + coolfnc,
            "cooling/cfl_cool=5.0",
            "cooling/cfl_cool_sub=0.5",
            "mesh/nx1=" + repr(N),
            "mesh/x2min=" + repr(-0.00625 * 32 / N),
            "mesh/x2max=" + repr(0.00625 * 32 / N),
            "mesh/x3min=" + repr(-0.00625 * 32 / N),
            "mesh/x3max=" + repr(0.00625 * 32 / N),
            "problem/nH_0=5.0",
            "problem/kn=1",
            "problem/alpha={}".format(_amp),
        ]

        # Run Athena++ as though we called
        # ./athena -i ../inputs/cooling/athinput.thermal_instability_test
        athena.run("cooling/athinput.linear_ti_test", arguments)
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
    ref_file = "data/ref_cooling_soltuions/ti_test_tigress_nH5.txt"
    (t_ref, alpha_ref) = np.loadtxt(ref_file).T

    # keep track of absolute errors
    errors_abs = []
    # initialize analyze_status to True (remains true if all tests are passed)
    analyze_status = True

    # loop over cooling sovers
    for csolver in ["euler", "op"]:
        # loop over different resolutions that have been tested
        for nx, err_tol in zip(resolution_range, error_rel_tols):
            # Read in the data produced during this test.
            filename = "bin/linear_ti_test-{}-{}.hst".format(csolver, nx)
            hst_data = athena_read.hst(filename)
            t_sol = hst_data["time"]
            alpha_sol = hst_data["rho_max"]

            # Next we compute the differences between the reference arrays and
            # the newly created ones in the L^1 sense.
            error_abs = np.trapz(
                abs(alpha_sol - np.interp(t_sol, t_ref, alpha_ref)), t_sol
            )
            errors_abs += [error_abs]

            # Compute the relative error
            error_rel = error_abs / np.trapz(np.interp(t_sol, t_ref, alpha_ref), t_sol)

            # if error does not meet tolerance, test is failed
            if error_rel > err_tol or np.isnan(error_rel):
                print("Relative error is ", error_rel)
                analyze_status = False

        # Finally, we want to check the 2nd order convergence rate of solver
        print("[{} solver]: convergence rates".format(csolver))
        for j in range(len(resolution_range) - 1):
            rate = np.log(errors_abs[j] / errors_abs[j + 1]) / (
                np.log(resolution_range[j + 1] / resolution_range[j])
            )
            print(
                "{} --> {}: {}".format(
                    resolution_range[j], resolution_range[j + 1], rate
                )
            )
            if rate < rate_tols[j]:
                print("Desired Convergance not achieved with {} solver".format(csolver))
                analyze_status = False
    return analyze_status
