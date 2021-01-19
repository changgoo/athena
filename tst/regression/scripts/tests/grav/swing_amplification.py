# Regression test based on the swing amplification test
# Convergence of L2 norm of the density error is tested

# Modules
import logging
import scripts.utils.athena as athena
import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.insert(0, '../../vis/python')
import athena_read  # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])  # set logger name based on module

resolution_range = [32, 64]
rate_tols = -1.99

# Set dimensionless input parameters
Lx = 1.0        # box size
Omega0 = 1.0    # angular freqency
qshear = 1.0    # shear rate
Q = 2           # Toomre Q
nJ = 2.5        # Jeans number
nwx = -3        # number of waves in x
nwy = 1         # number of waves in y
amp = 1e-7      # perturbation amplitude

# derived parameters
kappa = np.sqrt(4-2*qshear)*Omega0
cs = np.sqrt(4.0-2.0*qshear)/np.pi/nJ/Q
cs2 = cs**2
kx = 2*np.pi*nwx/Lx
ky = 2*np.pi*nwy/Lx
gconst = nJ*cs2

def prepare(*args, **kwargs):
    logger.debug('Running test ' + __name__)
    athena.configure('mpi','fft','hdf5','h5double',
                     prob='msa',
                     eos='isothermal', flux='hlle',
                     grav='fft-periodic', *args, **kwargs)
    athena.make()

def run(**kwargs):
    for n in resolution_range:
        arguments = ['job/problem_id=SwingAmplification_'
                     + repr(n),
                     'output1/file_type=hdf5', 'output1/variable=prim',
                     'output1/dt=2',
                     'time/cfl_number=0.3',
                     'time/tlim=2', 'time/nlim=10000',
                     'time/integrator=vl2',
                     'time/xorder=2',
                     'time/ncycle_out=0',
                     'mesh/nx1=' + repr(n),
                     'mesh/x1min={}'.format(-Lx/2.),
                     'mesh/x1max={}'.format(Lx/2.),
                     'mesh/ix1_bc=shear_periodic', 'mesh/ox1_bc=shear_periodic',
                     'mesh/nx2=' + repr(n),
                     'mesh/x2min={}'.format(-Lx/2.),
                     'mesh/x2max={}'.format(Lx/2.),
                     'mesh/ix2_bc=periodic', 'mesh/ox2_bc=periodic',
                     'mesh/nx3=4',
                     'mesh/x3min={}'.format(-Lx/2.),
                     'mesh/x3max={}'.format(Lx/2.),
                     'mesh/ix3_bc=periodic', 'mesh/ox3_bc=periodic',
                     'meshblock/nx1=' + repr(n),
                     'meshblock/nx2=' + repr(n),
                     'meshblock/nx3=4',
                     'hydro/iso_sound_speed={}'.format(cs),
                     'orbital_advection/OAorder=0',
                     'orbital_advection/qshear={}'.format(qshear),
                     'orbital_advection/Omega0={}'.format(Omega0),
                     'orbital_advection/shboxcoord=1',
                     'problem/Q={}'.format(Q),
                     'problem/nJ={}'.format(nJ),
                     'problem/amp={}'.format(amp),
                     'problem/nwx={}'.format(nwx),
                     'problem/nwy={}'.format(nwy)]
        athena.run('mhd/athinput.msa', arguments)

def analyze():
    def ode(t,y,kx,ky,kappa,qshear,cs2,gconst):
        kxt = kx + qshear*ky*t
        kmag = np.sqrt(kxt**2+ky**2)
        f0 = -kxt*y[1]-ky*y[2]
        f1 = 2*y[2]+kxt*(cs2-4*np.pi*gconst/kmag**2)*y[0]
        f2 = -0.5*kappa**2*y[1]+ky*(cs2-4*np.pi*gconst/kmag**2)*y[0]
        return np.array([f0,f1,f2])

    l2ERROR = []

    # solve ODE to get the reference solution
    y0 = 0.5*amp*np.array([1, kx/ky, 1])
    odesol = solve_ivp(ode, (0,2), y0, atol=1e-30, rtol=1e-13,
                       t_eval = [2,],
                       args=[kx,ky,kappa,qshear,cs2,gconst])
    kxt = kx+qshear*ky*odesol.t

    # compute L2 norms
    for n in resolution_range:
        ds = athena_read.athdf('bin/SwingAmplification_' + str(n)
                               + '.out1.00001.athdf')
        analytic = 1.0 + 2*odesol.y[0]*np.cos(ky*ds['x2v'][None,:,None]\
                 + kxt*ds['x1v'][None,None,:]) + 0*ds['x3v'][:,None,None]
        dx = Lx/n
        l2ERROR.append(np.sqrt(((ds['rho']-analytic)**2*dx**2).sum()))

    # estimate L2 convergence
    analyze_status = True
    conv = (np.diff(np.log(np.array(l2ERROR)))
                / np.diff(np.log(np.array(resolution_range))))
    logger.info('[Swing Amplification]: Convergence order = {}'
                .format(conv))

    if conv > rate_tols:
        logger.warning('[Swing Amplification]: '
                       'Scheme NOT converging at expected order.')
        analyze_status = False
    else:
        logger.info('[Swing Amplification]: '
                    'Scheme converging at expected order.')

    return analyze_status
