from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

MSUN_IN_MP = 1.188813e57
KELVIN_TIMES_KB_ERG = 1.38065e-16
PC3_IN_CM3 = 2.938e55
MSUN_IN_MP = 1.188813e57

# Multiply by this to get erg/cm^3 to code units
ERG_CM3_CODE = 4.27e13

# multiply code units by this to get time in myr
TIME_CODE_UNIT_MYR = 0.97779

kb = 1.38064852e-16 # cgs
m_p = 1.672621898e-24 # grams
pc_in_cm = 3.08568e+18

def get_pressure_from_code_units(P_code):
    '''
    assumes you'll feed an array of pressure in code units.
    returns P/k in K*cm^3
    '''
    return P_code/ERG_CM3_CODE/KELVIN_TIMES_KB_ERG

def get_time_from_code_units(t_code):
    '''
    t_code: time in code units
    returns: time in yrs (not Myr!)
    '''
    return t_code*TIME_CODE_UNIT_MYR*1e6

def get_n_from_code_pressure(rho_code, neutral = False):
    '''
    rho_code: rho in code units
    returns n_h (number density of hydrogen)
            n (number density of everything)
    assumes neutral
    '''
    n_H = rho_code.copy()
    if neutral:
        n = 1.1*n_H
    else:
        n = (1.4/0.62)*n_H
    return n_H, n

def make_cells(r_min = 2, r_max = 70, Nsteps = 256, xrat = 1.01):
    '''
    to see what the code is doing
    r_n are the *boundaries* of the cells.
    This sets them such that the width of the cells also increase by a factor of xrat.

    '''
    summ = np.sum(xrat**np.arange(Nsteps))
    dr1 = (r_max - r_min)/summ
    r_n = np.concatenate([[r_min], r_min + np.cumsum(dr1 * xrat**np.arange(Nsteps))])
    return r_n

def cooling_curve(T):
    '''
    taken from Drummond
    '''
    scale_Ti = [0.99999999e2, 1.0e+02, 6.0e+03, 1.75e+04, 4.0e+04, 8.7e+04, 2.30e+05,
        3.6e+05, 1.5e+06, 3.50e+06, 2.6e+07, 1.0e+12, np.inf]
    exps_i =   [1e10, 0.73167566,  8.33549431, -0.26992783, 1.89942352, -0.00984338,
        -1.8698263, -0.41187018, -1.50238273, -0.25473349, 0.5000359, 0.5]
    coeffs_i =  [3.720076376848256e-71, 1.00e-27, 2.00e-26, 1.50e-22, 1.20e-22, 5.25e-22,
        5.20e-22, 2.25e-22, 1.25e-22, 3.50e-23, 2.10e-23,   4.12e-21]

    Lambda = np.zeros(len(T))
    Lambda[T < scale_Ti[0]] = 1.0e-50
    for i, s in enumerate(scale_Ti[:-1]):
        msk = (T > scale_Ti[i]) & (T < scale_Ti[i+1])
        Lambda[msk] = coeffs_i[i]*(T[msk]/scale_Ti[i])**(exps_i[i])
    return Lambda

def tcool(T, nH):
    '''
    also from Drummond
    in seconds, presumably
    '''
    T_PE = 1e4 # no photoelectric heating above 10000 K because there's no dust
    kb = 1.38065e-16 # erg/K
    gamma = 5/3
    muH, mu = 1.4, 0.62
    Gamma0 = 2e-26 # erg/s for density of 1.4 m_p cm3
    Gamma = np.zeros(len(T))
    Gamma[T < T_PE] = (nH/2)*Gamma0
    #Gamma[T < T_PE] = Gamma0
    Gamma[T >= T_PE] = 0
    tcool = kb*T/((gamma - 1) * (mu/muH)*(nH * cooling_curve(T) - Gamma0))
    return tcool

def read_in_1d_output(path, get_volume = False):
    '''
    get quantities as a function of radius.
    also return time if hdf5
    '''
    import athena_read

    if path.split('.')[-1] == 'vtk':
        data_1d = athena_read.vtk(filename = path)
        r_edges_pc = data_1d[0]
        r_pc = r_edges_pc[:-1] + 0.5*np.diff(r_edges_pc)
        press = data_1d[3]['press'][0][0]
        rho = data_1d[3]['rho'][0][0]
        vr = data_1d[3]['vel'][0][0][:, 0]
    elif path.split('.')[-1] == 'athdf':
        data_1d = athena_read.athdf(filename = path,
            quantities=['rho', 'press', 'vel1', 'vel2', 'vel3'])
        r_pc = data_1d['x1v']
        press = data_1d['press'][0][0]
        rho = data_1d['rho'][0][0]
        vr = data_1d['vel1'][0][0]
        time = data_1d['Time']
        if get_volume:
            r_edge = data_1d['x1f']
            shell_vol = 4*np.pi/3*(r_edge[1:]**3 - r_edge[:-1]**3)

    else:
        raise ValueError('data format not good.')

    P_over_kb = get_pressure_from_code_units(P_code = press)
    nH_cm3, n_cm3 = get_n_from_code_pressure(rho_code = rho)
    T_K = P_over_kb/n_cm3
    if get_volume:
        return r_pc, nH_cm3, P_over_kb, T_K, vr, shell_vol, time
    return r_pc, nH_cm3, P_over_kb, T_K, vr

def read_in_user_output_variables(path_uov):
    '''
    just a test for conduction.
    '''
    import athena_read

    if path_uov.split('.')[-1] == 'vtk':
        data_uov = athena_read.vtk(filename = path_uov)
        qcond = data_uov[3]['user_out_var0'][0][0]
        qsat = data_uov[3]['user_out_var1'][0][0]
    return qcond, qsat

def read_in_history_file(path, variables = ['time', 'Vbubble']):
    '''
    useful because the order changes whenever we change something
    '''
    f = open(path)
    line = f.readlines()[1]
    f.close()

    all_vars = np.array([var.split('[')[0].strip() for var in line.split(']=')[1:]])
    idx = [np.arange(len(all_vars))[all_vars == var][0] for var in variables]
    if len(idx) != len(variables):
        raise ValueError('could not find desired variable!')

    data = np.genfromtxt(path)
    values = [data[:, idxx] for idxx in idx]
    return values

def sedov_taylor_radius(t_Myr, E51 = 1, n_amb0 = 1):
    '''
    sedov-taylor solution
    KO15 eq. 3
    '''
    r_st = 5.0*(E51/n_amb0)**(1/5.) * (t_Myr*1.e3)**(2/5.)
    return r_st

def sedov_taylor_velocity(t_Myr, E51 = 1, n_amb0 = 1):
    '''
    sedov-taylor solution
    KO15 eq. 4
    '''
    v_st = 1.95e3*(E51/n_amb0)**(1/5.) * (t_Myr*1.e3)**(-3/5.)
    return r_st

def sedov_taylor_momentum(t_Myr, E51 = 1, n_amb0 = 1):
    '''
    sedov-taylor solution
    KO15 eq. 4
    '''
    t3=t_Myr*1.e3
    p_st = 2.21e4*E51**0.8*n_amb0**0.2*t3**0.6
    return p_st

def t_shell_formation(E51 = 1, n_amb0 = 1):
    '''
    KO15 eq. 7
    '''
    tsf_Myr = 4.4e-2 * E51**0.22 * n_amb0**(-0.55)
    return tsf_Myr

def pressure_driven_snowplow_radius(t_Myr, E51 = 1, delta_t = 0.1, n_amb0 = 1):
    '''
    from eq 16 of KOR+17
    delta_t is the time spacing between SNe, in Myr
    '''
    r_pds = 52*(E51/(delta_t * n_amb0))**(1/5) * t_Myr**(3/5)
    return r_pds

def momentum_driven_snowplow_radius(t_Myr, pstar5 = 1, delta_t = 0.1, n_amb0 = 1):
    '''
    from eq 18 of KOR+17
    delta_t is the time spacing between SNe, in Myr
    pstar5 = momentum per sn / 1e5 Msun km/s
    '''
    r_mds = 34*(pstar5/(delta_t * n_amb0))**(1/4) * t_Myr**(1/2)
    return r_mds

def weaver_R2(Lw, rho0, t):
    '''
    eq 21
    '''
    return (250/(308*np.pi))**(1/5) * Lw**(1/5) * rho0**(-1/5) * t**(3/5)

def weaver_P(Lw, rho0, t):
    '''
    eq 22
    pressure, not momentum
    '''
    return 7/(3850*np.pi)**(2/5) * Lw**(2/5) * rho0**(3/5) * t**(-4/5)


def weaver_p(Lw, rho0, t):
    '''
    momentum. Just rdot times Mshell.
    '''
    B = (250/(308*np.pi))**(1/5) * Lw**(1/5) * rho0**(-1/5)
    v = 3/5*B*t**(-2/5)
    R2 = B*t**(3/5)
    Mshell = 4*np.pi/3 * R2**3 * rho0
    return v*Mshell

def weaver_dMbdt(Lw, rho0, t):
    '''
    eq 33
    '''
    C = 6e-7
    m_p = 1.672621898e-24 # grams
    mu = 0.62 * m_p
    kb = 1.38064852e-16 # cgs
    R2 = weaver_R2(Lw = Lw, rho0 = rho0, t = t)
    P = weaver_P(Lw = Lw, rho0 = rho0, t = t)
    A = 1.646
    return (12/75)*A**(5/2)*4*np.pi*R2**3 * mu/(kb * t) * (t*C/R2**2)**(2/7)*P**(5/7)

def weaver_Mb(Lw, rho0, t):
    '''
    bubble mass
    '''
    C = 6e-7 # cgs
    m_p = 1.672621898e-24 # grams
    mu = 0.62 * m_p
    kb = 1.38064852e-16 # cgs
    A = 1.646 # dimensionless
    R2 = weaver_R2(Lw = Lw, rho0 = rho0, t = t)
    P = weaver_P(Lw = Lw, rho0 = rho0, t = t)
    return 28/205*A**(5/2)*4*np.pi*R2**3*(mu/kb)*(t*C/R2**2)**(2/7) * P**(5/7)

def cond_mass_flux(R, T):
    '''
    from cowie and mckee
    '''
    m_p = 1.672621898e-24 # grams
    mu = 0.62 * m_p
    C = 6e-7 # cgs
    kb = 1.38064852e-16 # cgs
    dmdt = 16*np.pi*mu/(25*kb)*C*T**(5/2)*R
    return dmdt


def plot_weaver_mass_flux():
    '''
    to compare with mac low and mccray
    '''
    mu = 0.62
    C = 6e-7 # cgs
    A = 1.646
    m_p = 1.672621898e-24 # grams
    dt_myr = 0.1 # myr
    kb = 1.38064852e-16 # cgs
    ts = np.linspace(0.001, 1, 1000)*1e6*3.15e7 # 10 myr in sec
    Lw = 1e51/dt_myr / (1e6*3.15e7) # erg/s for 1 sne every 0.1 Myr
    rho0 = 1.4*m_p # g/cm3
    n0 = rho0/(mu*m_p)
    dMdt = weaver_dMbdt(Lw = Lw, rho0 = rho0, t = ts) # in g/s
    R2 = weaver_R2(Lw = Lw, rho0 = rho0, t = ts)
    P = weaver_P(Lw = Lw, rho0 = rho0, t = ts)

    T = (P*R2**2/(ts*C))**(2/7)*A
    dmdt_cond = cond_mass_flux(R = R2, T = T)

    c_hot = np.sqrt(kb*T/(mu*mu*m_p))
    mdot_sat = 12*np.pi/5 * P/c_hot * R2**2

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(ts/(1e6*3.15e7), dMdt*(1e6*3.15e7/2e33), 'k', label = r'$\rm unsaturated$')
    ax.plot(ts/(1e6*3.15e7), mdot_sat*(1e6*3.15e7/2e33), 'r--', label = r'$\rm saturated$')
    ax.set_xlabel(r'$\rm time\,\,[Myr]$', fontsize = 20)
    ax.set_ylabel(r'${\rm d}M_{{\rm b}}/{\rm d}t\,\,\left[{\rm M_{\odot}\,Myr^{-1}}\right]$', fontsize = 20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 10000)
    ax.tick_params(labelsize = 18)
    ax.legend(loc = 'best', frameon = False)

def plot_weaver_kinetic_and_internal_energy():
    '''
    mostly to see the ratio of kinetic to thermal energy
    '''
    mu = 0.62
    C = 6e-7 # cgs
    A = 1.646
    m_p = 1.672621898e-24 # grams
    dt_myr = 0.1 # myr
    kb = 1.38064852e-16 # cgs
    ts = np.linspace(0.001, 10, 1000)*1e6*3.15e7 # 10 myr in sec
    Lw = 1e51/dt_myr / (1e6*3.15e7) # erg/s for 1 sne every 0.1 Myr
    rho0 = 1.4*m_p # g/cm3
    n0 = rho0/(mu*m_p)
    dMdt = weaver_dMbdt(Lw = Lw, rho0 = rho0, t = ts) # in g/s
    R2 = weaver_R2(Lw = Lw, rho0 = rho0, t = ts)
    P = weaver_P(Lw = Lw, rho0 = rho0, t = ts)
    p = weaver_p(Lw = Lw, rho0 = rho0, t = ts)
    M_shell = 4*np.pi/3*R2**3 * rho0
    E_kin = p**2/(2*M_shell)
    E_int = 5/11 * Lw * ts

    f = plt.figure(figsize = (6, 4))
    ax = f.add_subplot(111)
    ax.plot(ts/(1e6*3.15e7), E_kin, 'b:', label = 'kinetic')
    ax.plot(ts/(1e6*3.15e7), E_int, 'r--', label = 'thermal')
    ax.plot(ts/(1e6*3.15e7), E_int+E_kin, 'g-.', label = 'thermal + kinetic')

    ax.plot(ts/(1e6*3.15e7), Lw*ts, 'k', label = r'$L_{w} \times t$')
    ax.legend(loc = 'best', frameon = False)
    ax.set_xlabel('time (Myr)')
    ax.set_ylabel('energy (erg)')

def weaver_interior_structure(x, Lw, rho0, time_myr):
    '''
    taken from MacLow & Mccray 1988
    returns n and T as a function of r/R
    actually, just taken from weaver eventualy
    '''
    m_p = 1.672621898e-24 # grams
    t = time_myr*3.15e7*1e6

    kb = 1.38064852e-16 # cgs
    mu = 0.62
    C = 6e-7 # cgs
    A = 1.646
    m_p = 1.672621898e-24 # grams
    P = weaver_P(Lw = Lw, rho0 = rho0, t = t)
    R2 = weaver_R2(Lw = Lw, rho0 = rho0, t = t)
    T_c = A*(P*R2**2/(t*C))**(2/7)
    T_x = T_c * (1 - x)**(2/5)

    # we know P= nkT= const, so
    n_x = P/(kb*T_x)
    return n_x, T_x

def get_cs_ionized(P_k, nH):
    '''
    take P_over_kb and nH and return cs
    this is the isothermal sound speed (no gamma)
    '''
    m_p = 1.672621898e-24 # grams
    kb = 1.38064852e-16 # cgs
    return np.sqrt(kb*P_k/(nH*m_p))

def locate_shock(r_pc, P_k, nH, vr, last_r, last_vr, dt_myr):
    '''
    assume that in the last snapshot, the shock was at radius last_r
    P_k is a one-d array of pressure
    last_vr radial velocity in km/s; providing a guess of where the shock is.
    return r, vr, cs
    '''

    from scipy.signal import argrelextrema
    dt_sec = dt_myr*3.15e7*1e6
    dr_pc = last_vr*dt_sec/(3.086e13)
    guess_r = last_r + dr_pc
    max_argP = argrelextrema(P_k, np.greater)[0]
    max_argP = max_argP[(r_pc[max_argP] > last_r) & (np.abs(r_pc[max_argP] - guess_r) < 2)]

    try:
        best_idx = max_argP[np.argmax(P_k[max_argP])]
        new_r, new_vr = r_pc[best_idx], vr[best_idx]

        msk = (np.arange(len(P_k)) > best_idx) & (np.abs(vr) > 1) & (nH < 1) # in front of shock
        cs = get_cs_ionized(P_k = P_k[msk], nH = nH[msk])
    except ValueError:
        new_r, new_vr, cs = np.nan, np.nan, np.nan
    return new_r, new_vr, np.median(cs)

def locate_shock_T_gradient(r_pc, P_k, nH, dTdr, vr, last_r, last_vr, dt_myr):
    '''
    assume that in the last snapshot, the shock was at radius last_r
    P_k is a one-d array of pressure
    last_vr radial velocity in km/s; providing a guess of where the shock is.
    return r, vr, cs
    use the negative temperature gradient
    '''

    from astropy.convolution import convolve, Gaussian1DKernel
    g = Gaussian1DKernel(2)
    dTdr = convolve(dTdr, g)

    from scipy.signal import argrelextrema
    dt_sec = dt_myr*3.15e7*1e6
    dr_pc = last_vr*dt_sec/(3.086e13)
    guess_r = last_r + dr_pc
    max_arg = argrelextrema(dTdr, np.greater)[0]
    max_arg = max_arg[(r_pc[max_arg] > last_r) & (np.abs(r_pc[max_arg] - guess_r) < 2)]

    try:
        best_idx = max_arg[np.argmax(dTdr[max_arg])]
        new_r, new_vr = r_pc[best_idx], vr[best_idx]

        msk = (np.arange(len(P_k)) > best_idx) & (np.abs(vr) > 0.5) & (nH < 0.5) # in front of shock
        cs = get_cs_ionized(P_k = P_k[msk], nH = nH[msk])
    except ValueError:
        new_r, new_vr, cs = np.nan, np.nan, np.nan
    return new_r, new_vr, np.median(cs)

def spitzer_conductivity(T, ne = 0.01):
    '''
    can also compare to the Parker one
    T is in K
    answer in erg/s/cm/K
    '''
    return 1.84e-5*T**(5/2)/(29.7 + np.log(ne**(-1/2) + T/1e6))

def parker_conductivity(T):
    '''
    from eq 31 of Parker 1953
    '''
    return 2.5e3*T**0.5

def expected_mdot_profile_Weaver(Lw, rho0, t):
    '''
    plug the weaver density profile into the Cowie and Mckee formula
    '''
    mu, muH = 0.62, 1.4
    m_p = 1.672621898e-24 # grams
    kb = 1.38064852e-16 # cgs
    R2 = weaver_R2(Lw = Lw, rho0 = rho0, t=t)
    x = np.linspace(0, 0.999999, 1000)
    n_x, T_x = weaver_interior_structure(x = x, Lw = Lw, rho0 = rho0, time_myr = t/3.15e7/1e6)
    #kappa_s = spitzer_conductivity(T = T_x, ne = 1.2*mu/muH*n_x)
    # to be consistent with everything else.
    kappa_s = 6e-7*T_x**(5/2)
    r = x*R2
    mdot = 2/5*4*np.pi*r**2 * kappa_s * (1 - r/R2)**(-1) * mu * m_p /(2.5*kb*R2)
    return r, mdot

def mdot_assuming_weaver_profile(r_cm, R_cm, kappa_cgs):
    '''
    based on local temperature and kappa
    '''
    mu = 0.62
    m_p = 1.672621898e-24 # grams
    kb = 1.38064852e-16 # cgs
    mdot = 2/5*4*np.pi*r_cm**2*kappa_cgs*(1 - r_cm/R_cm)**(-1) * mu* m_p/(2.5*kb*R_cm)
    return mdot

def get_kappa_cgs(r_cm, T_K, P_over_k, nH_cm3, lambda_dv = 1):
    '''
    account for saturation, Parker conductivity, nonlinear mixing, etc.
    '''
    mu = 0.62
    dTdr = np.diff(T_K)/np.diff(r_cm)
    r_cen = r_cm[:-1] + 0.5*np.diff(r_cm)
    dTdr = np.interp(r_cm, r_cen, dTdr)
    P_cgs = P_over_k*kb # erg/cm^3
    rho_cgs = 1.4*m_p*nH_cm3 # in g/cm3

    qsat = 1.5*rho_cgs*(P_cgs/rho_cgs)**(3/2) # erg/s/cm^2


    msk = T_K < 6.6e4
    kappa = spitzer_conductivity(T = T_K, ne = 1.2*nH_cm3)
    kappa[msk] = parker_conductivity(T = T_K[msk])

    kappa_eff = 1/((np.abs(dTdr)/qsat) + (1/kappa))

    lambda_dv_cgs = lambda_dv * pc_in_cm * 1e5
    kappa_nonlinear = lambda_dv_cgs * rho_cgs * kb /(mu * m_p)
    kappa[kappa_nonlinear > kappa] = kappa_nonlinear[kappa_nonlinear > kappa]
    return kappa
