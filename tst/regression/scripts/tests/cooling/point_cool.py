"""
Example test script.

This is a complete, working example that can be run as part of the test suite. It does a
simple test of a relativistic shock tube using the GR framework. There are many comments
in order to make this file self-explanatory, but the actual working code is only 28
lines long.

There are three functions defined here:
    prepare(**kwargs)
    run(**kwargs)
    analyze()
All three must be defined with the same names and no required inputs in order to make a
working script. They are called in sequence from the main test script run_tests.py.
Additional support functions can be defined here, to be called by the three primary fns.

Heavy use is made of support utilities defined in scripts/utils/athena.py. These are
general-purpose Python scripts that interact with Athena++. They should be used whenever
possible, since they work together to compile and run Athena++ and read the output data.
In particular, proper use of them will result in all files outside tst/regression/ being
in the same state after the test as they were before (including whatever configured
version of Athena++ existed in athena/bin/), as well as cleaning up any new files
produced by the test.
"""

# Modules
import numpy as np                             # standard Python module for numerics
import sys                                     # standard Python module to change path
import scripts.utils.athena as athena          # utilities for running Athena++
import scripts.utils.comparison as comparison  # more utilities explicitly for testing
sys.path.insert(0, '../../vis/python')         # insert path to Python read scripts
import athena_read                             # utilities for reading Athena++ data # noqa
athena_read.check_nan_flag = True              # raise exception when encountering NaNs


def prepare(**kwargs):
    """
    Configure and make the executable.

    This function is called first. It is responsible for calling the configure script and
    make to create an executable. It takes no inputs and produces no outputs.
    """

    # Configure as though we ran
    #     python configure.py -hdf5 --prob=cooling
    # from the athena/ directory. Note that additional -<flag> command-line arguments can
    # be specified as additional '<flag>' arguments before the <key>='<value>' arguments
    # to athena.configure(). Any number of --<key>=<value> command-line arguments can also
    # be supplied. Note athena.configure() expects the values only to be quoted, e.g.
    # --<key>='<value>'.
    athena.configure('-hdf5',prob='cooling',**kwargs)

    # Call make as though we ran
    #     make clean
    #     make
    # from the athena/ directory.
    athena.make()


def run(**kwargs):
    """
    Run the executable.

    This function is called second. It is responsible for calling the Athena++ binary in
    such a way as to produce testable output. It takes no inputs and produces no outputs.
    """

    # Create list of runtime arguments to override the athinput file. Each element in the
    # list is simply a string of the form '<block>/<field>=<value>', where the contents of
    # the string are exactly what one would type on the command line run running Athena++.
    arguments = ['time/ncycle_out=1',
                 'job/problem_id=cooling',
                 'output1/file_type=hst',
                 'output1/dt=1e-8',
                 'time/cfl_number=0.3',
                 'time/tlim=2.0',
                 'mesh/nx1=8',
                 'mesh/nx2=1',
                 'mesh/nx3=1',
                 'cooling/coolftn=tigress',
                 'cooling/solver=rk4',
                 'problem/turb_flag=0',
                 'problem/rho_0=1.0',
                 'problem/pgas_0=3000']

    # Run Athena++ as though we called
    #     ./athena -i ../inputs/cooling/athinput.cooling_test job/problem_id=cooling <...>
    # from the bin/ directory. Note we omit the leading '../inputs/' below when specifying
    # the athinput file.)
    athena.run('cooling/athinput.cooling_test', arguments)
    # No return statement/value is ever required from run(), but returning anything other
    # than default None will cause run_tests.py to skip executing the optional Lcov cmd
    # immediately after this module.run() finishes, e.g. if Lcov was already invoked by:
    # athena.run('cooling/athinput.cooling_test', arguments, lcov_test_suffix='mb_1')
    return 'skip_lcov'


def analyze():
    """
    Analyze the output and determine if the test passes.

    This function is called third; nothing from this file is called after it. It is
    responsible for reading whatever data it needs and making a judgment about whether or
    not the test passes. It takes no inputs. Output should be True (test passes) or False
    (test fails).
    """

    # Read in reference data. Which is in Physical units of Myrs (for t_ref) and Kelvin 
    # for T_ref (which is actually T_mu when doing the comparison of the TIGRESS classic
    # cooling function). 
    # This is the result of an Euler cooling integration done separately in 
    # Python with a much smaller time step. Make sure the that file that is being 
    # compared matches the inputs given above in the "run()" function
    (t_ref, T_ref) = np.loadtxt('data/tigress_pok3e3_nH1e0.txt')


    # Read in the data produced during this test. This will usually be stored in the
    # tst/regression/bin/ directory, but again we omit the first part of the path. Note
    # the file name is what we expect based on the job/problem_id field supplied in run().
    (t_sol,Etot_sol) = np.loadtxt('bin/cooling.hst',usecols=(0,9))
    # default volume of the simulation domain
    vol = 8 
    # default conversion factor from code to kB K cm^-3 if using TIGRESS Units
    Pconv = 1.696742e+02
    # density, pressure, and T_mu of the solution
    rho = mass/vol
    P = (2./3)*Pconv*Etot/vol
    T_sol = P/rho

    # Next we compute the differences between the reference arrays and the newly created
    # ones in the L^1 sense. That is, given functions f and g, we want
    #     \int |f(x)-g(x)| dx.
    # The utility script comparison.l1_diff() does this exactly, conveniently taking N+1
    # interface locations and N volume-averaged quantities. The two datasets can have
    # different values of N.
    error_abs_T = comparison.l1_diff(t_ref, T_ref, t_sol, T_sol)
 
    # The errors are more meaningful if we account for the length of the domain and the
    # typical magnitude of the function itself. Fortunately, comparison.l1_norm() computes
    #     \int |f(x)| dx.
    # (Note neither comparison.l1_diff() nor comparison.l1_norm() divides by the length of
    # the domain.)
    error_rel_T = error_abs_T / comparison.l1_norm(t_ref, T_ref)

    # Finally, we test that the relative errors in the two quantities are no more than 1%.
    # If they are, we return False at the very end of the function and file; otherwise
    # we return True. NumPy provides a way of checking if the error is NaN, which also
    # indicates something went wrong. The same check can (and should) be enabled
    # automatically at the point of reading the input files via the athena_read.py
    # functions by setting "athena_read.check_nan_flag=True" (as done at the top of this
    # file). Regression test authors should keep in mind the caveats of floating-point
    # calculations and perform multiple checks for NaNs when necessary.

    # The main test script will record the result and delete both tst/regression/bin/ and
    # obj/ folders before proceeding on to the next test.
    analyze_status = True
    if error_rel_T > 0.01 or np.isnan(error_rel_e):
        analyze_status = False

    # Note, if the problem generator in question outputs a unique CSV file containing
    # quantitative error measurements (e.g. --prob=linear_wave outputs
    # linearwave-errors.dat when problem/compute_error=true at runtime), then these values
    # can also be input and used in this analyze() function. It is recommended to use:
    # athena_read.error_dat('bin/linearwave-errors.dat')
    # This wrapper function to np.loadtxt() can automatically check for the presence of
    # NaN values as in the other athena_read.py functions.

    return analyze_status
