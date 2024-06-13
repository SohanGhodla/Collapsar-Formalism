# --------- This notebook assumes a Magentically arrested accretion flow -----------

import pylab as pl
import numpy as np
from math import sqrt, log10, pi, log, cos, floor
from scipy.interpolate import interp1d
import scipy.integrate as integrate
# from mplchange import *
import mesa_reader as mr
import sys
import pandas as pd

dir_path = '../data/MESA_LOGS/0.45/'
# dir_path = '/nesi/nobackup/uoa03218/Remnant_mass_spectrum_MESA_1/Makemodels/zsun_50/rot-0.05/35/LOGS/'

# adjustable parameters
s_PL = 0.5    # power-law index for the radial accretion rate profile in the advective regime
alp_ss = 0.01    # viscosity parameter
# themin = 30*pi/180   # the minimum polar angle [rad] below which fallback is impeded by feedback
dt_over_tvis = 0.005    # time resolution factor

# some constants
c = 2.99792e10  # speed of light
G = 6.674e-8     # Newton's constant
rsun = 6.96e10         # solar radius
msun = 1.98847e33      # solar mass
R_unit = G/c**2
J_unit = G*msun**2/c

prof = mr.MesaData(file_name= dir_path + 'profile82' + '.data')
# print(prof.bulk_names)

rhodat = np.flip(prof.logRho)       # density in each shell -- already in log10
rdat = np.log10(np.flip(prof.radius)*rsun)   # radius (right boundary)
Omgdat = np.log10(np.flip(prof.omega))       # angular frequency of each radial shell
massdat = np.log10(np.flip(prof.mass)*msun)
AMdat = np.flip(prof.log_J_inside)   # -- already in log10

unique_data = {}
for ai, yi, bi, ci, di in zip(rdat, massdat, Omgdat, rhodat, AMdat):
    yi = round(yi, 12)
    if yi not in unique_data:
        unique_data[yi] = (ai, bi, ci, di)

# Extract the unique x, z, and w values from the dictionary using the unique y-values as keys
massdat = np.array(list(unique_data.keys()))
unique_a, unique_b, unique_c, unique_d = zip(*[unique_data[yi] for yi in massdat])

# convert to arrays
rdat   = np.array(unique_a)
Omgdat = np.array(unique_b)
rhodat = np.array(unique_c)
AMdat  = np.array(unique_d)
# Marr = np.empty(Nr, dtype=float)    # enclosed mass

intp_lgrho = interp1d(rdat, rhodat, fill_value='extrapolate')
intp_lgomg = interp1d(rdat, Omgdat, fill_value='extrapolate')
intp_lgmass = interp1d(rdat, massdat, fill_value='extrapolate')
intp_lgAM = interp1d(rdat, AMdat, fill_value='extrapolate')

# interpolate these profiles to a finer grid
Nr = 2000
rarr = np.logspace((rdat[0]), (rdat[-1]), Nr)  # right boundary of shell
# rhoarr = np.array([10**intp_lgrho(log10(r)) for r in rarr])
Omgarr = np.array([10**intp_lgomg(log10(r)) for r in rarr])
Marr = np.array([10**intp_lgmass(log10(r)) for r in rarr])
Jarr = np.array([10**intp_lgAM(log10(r)) for r in rarr])


# ----------------------------
def delta_t(M, a, r_isco, r_0):
    def inner_expression(r):
        term1 = 2 * M * log((sqrt(r) - sqrt(2*M)) / (sqrt(r) + sqrt(2*M)))
        term2 = 2 * sqrt(2 * M * r)
        term3 = (3 * a**2 * sqrt(2 * M / r) + 2 * sqrt(2 * M * r**3)) / (6 * M)
        # ----------- added extra c ------------
        return (term1 + term2 + term3)/c

    delta_t_r_0 = inner_expression(r_0)
    delta_t_r_isco = inner_expression(r_isco)

    return delta_t_r_0 - delta_t_r_isco

# ----------------------------
def u_t(M, r, a):
    numerator = r**(3/2) - 2 * M * r**(1/2) + a * sqrt(M)
    denominator = r**(3/4) * sqrt(r**(3/2) - 3 * M * r**(1/2) + 2 * a * sqrt(M))
    return numerator / denominator

# ----------------------------
def dot_J_wind(M_d, M, tau_vis, s, r_d, a, r_t):

    integral_term_1 = (2 * a * M * r_d**(s-1) / (1 - s)) + (2 * sqrt(M) * r_d**(s + 0.5) / (2 * s + 1))
    integral_term_2 = (2 * a * M * r_t**(s-1) / (1 - s)) + (2 * sqrt(M) * r_t**(s + 0.5) / (2 * s + 1))
    result = (M_d / tau_vis) * (s / r_d**s) * (integral_term_1 - integral_term_2) * c
    return result


# ---------------------------
def calculate_risco(M, a):
    z1 = 1 + (1 - a**2 / M**2)**(1/3) * ((1 + a / M)**(1/3) + (1 - a / M)**(1/3))
    z2 = sqrt(3 * a**2 / M**2 + z1**2)
    r_isco = M * (3 + z2 - sqrt((3 - z1) * (3 + z1 + 2 * z2)))
    return r_isco


# ----------------------------
def u_phi(M, r, a):
    numerator = sqrt(M) * (r**2 - 2 * a * sqrt(M) * r**(1/2) + a**2)
    denominator = r**(3/4) * sqrt(r**(3/2) - 3 * M * r**(1/2) + 2 * a * sqrt(M))
    u_phi_r = numerator / denominator
    return u_phi_r * c


tffarr = pi/2**1.5 * np.sqrt(rarr**3/G/Marr)
ellarr = np.diff(Jarr)/np.diff(Marr)

# need to find the time when accretion disk forms
Mbh0, abh0, Jbh0, i_disk = 0., 0., 0., 0.
for i in range(Nr-2):
    # ----- assume that the innermost 1 Msun rest mass always forms a black hole  --------
    if Marr[i] < msun:   
        continue

    Mbh = Marr[i]
    abh = c*Jarr[i]/(G*Mbh**2)
    
    if abh >= 0.9994:  # ------ Need to do this to respect cosmic censorship during the initial 1msun collapse bit  --------
        abh = 0.9994
        Jbh = abh * (G*Mbh**2)/c

    # \ell values below
    ell = ellarr[i+1]

    M_ = Mbh * R_unit
    a_ = abh * M_
    Risco = calculate_risco(M_, a_)
    jisco = u_phi(M_, Risco, a_) 

    # ------ the condition for the formation of the disk --------
    if ell >= jisco:
        i_disk = i+1
        Mbh0 = Marr[i]
        Jbh0 =  abh*(G*Mbh**2)/c   
        abh0 = abh
        # print(i/Nr)
        break
    # --- the disk never forms ------
    if i == Nr-3:
        print('no disk forms for this star!')
        print('final Mbh=%.3f Msun, abh=%.3f' % (Mbh/msun, abh))
        exit()

# the i_disk-th shell first forms the disk
tdisk = tffarr[i_disk]

# fallback rate profile 
Mfbdot = np.abs(np.diff(Marr))/np.abs(np.diff(tffarr)) 
Jfbdot = np.abs(np.diff(Jarr))/np.abs(np.diff(tffarr)) 

tmid = np.array([(tffarr[i]+tffarr[i+1])/2 for i in range(Nr-1)])
# -------- interplote for a finner grid. Also on a log scale  -----------
intp_lgMfbdot = interp1d(tmid, np.log10(Mfbdot), fill_value='extrapolate')
intp_lgJfbdot = interp1d(tmid, np.log10(Jfbdot), fill_value='extrapolate')

# ----- Go to 2x ---------
tmax = tffarr[-1]*2

# ----- number of time iterations ------
Ntgrid = 2000      
#  ----- new time grid -----  
tgrid = np.linspace(tdisk, tmax, Ntgrid)

# interpolate lgMfbdot and lgJfbdot on a regular grid
dtgrid = tgrid[1] - tgrid[0]
lgMfbdotgrid = np.empty(Ntgrid, dtype=float)
lgJfbdotgrid = np.empty(Ntgrid, dtype=float)
for i in range(Ntgrid):
    t = tgrid[i]
    if t > tmid[-1]:  # the disk never formed 
        lgMfbdotgrid[i] = -10.   # zero fallback rate
        lgJfbdotgrid[i] = -10.
    else:
        lgMfbdotgrid[i] = intp_lgMfbdot(t)
        lgJfbdotgrid[i] = intp_lgJfbdot(t)


def MJfbdot(t, tgrid, lgMJdot_grid):    # M/J fallback rate any time
    i_grid = min(Ntgrid-1, max(0, int(floor((t-tgrid[0])/dtgrid))))
    # t is usually between tgrid[i_grid] and tgrid[i_grid+1]
    slope = (lgMJdot_grid[i_grid+1] - lgMJdot_grid[i_grid])/(tgrid[i_grid+1] - tgrid[i_grid])
    lgMJdot = lgMJdot_grid[i_grid] + (t - tgrid[i_grid])*slope
    return 10**lgMJdot


def calc_grid_pt(t, tgrid):  # M/J fallback rate at a given time t
    i_grid = min(Ntgrid-1, max(0, int(floor((t-tgrid[0])/dtgrid))))
    return  i_grid

Mbh = Mbh0
Jbh = Jbh0
abh = c*Jbh/(G*Mbh**2)

if abh >= 0.9994:  # ------ Need to do this to respect cosmic censorship during the initial 1msun collapse bit  --------
    abh = 0.9994
    Jbh = abh * (G*Mbh**2)/c

M_ = Mbh * R_unit
a_ = abh * M_
Risco = calculate_risco(M_, a_)
jisco = u_phi(M_, Risco, a_) 

# --------- --------- --------- These are only for initization --------- ---------
OmgKisco = sqrt(G*Mbh/Risco**3)
tvis_isco = 1/alp_ss/OmgKisco

# initialize the disk properties (unimportant for the total energetics)
Md0 = MJfbdot(tdisk, tgrid, lgMfbdotgrid)*tvis_isco*0.5
Rd0 = 1.01*Risco
Rd = Rd0
Jd0 = Md0 * u_phi(M_, Rd0, a_)  # initial circular disk amgular momentum
Md = Md0
Jd = Jd0
t = tdisk

Mfb = 0
Jfb = 0
Mfb_rest = 0
Jfb_rest = 0

# ------ To find Rd (radius at which the orbit becomes circlar) given Jd and Md --------
def find_closest_r(M_, a_, Md, Jd, r_min, r_max,  max_iter=500):
    iter_count = 0
    X = Jd / Md
    tol=1e-3 * X
    while iter_count < max_iter:
        r_mid = (r_min + r_max) / 2
        j_mid = u_phi(M_, r_mid, a_)

        if np.abs(j_mid - X) < tol:
            # print('tol = ', tol / X, r_mid / r_min, j_mid / X)
            return r_mid
        
        if j_mid < X:
            r_min = r_mid
        else:
            r_max = r_mid
        iter_count += 1

    if X < u_phi(M_, r_min, a_):
        # print('yes')
        return r_min
        
    print('error')
    sys.exit(1)


# ------ To find R_in given u_t = 0.97 --------
def find_closest_r_in(M_, a_, ut, r_min, r_max,  max_iter=100):
    iter_count = 0
    X = ut
    tol = 1e-3 * X
    while iter_count < max_iter:
        r_mid = (r_min + r_max) / 2
        u_t_mid = u_t(M_, r_mid, a_)

        if np.abs(u_t_mid - X) < tol:
            # print('tol = ', r_mid / r_min, u_t_mid/ X)
            # print(u_t_mid, r_mid/ r_min)
            return r_mid
        
        if u_t_mid < X:
            r_min = r_mid
        else:
            r_max = r_mid
        iter_count += 1

    print('error')
    sys.exit(1)


# ------- disk evolution --------  

print('Mass of star at collapse [Msun] = %.3f' % (Marr[-1]/msun))
print('disk formation time [s] = %.3f' %  tdisk)

temp = [ [] for i in range(10)]
fac1 = 1
fac2 = 1
Eacc = 0.
Eaccj = 0.

FLAG = True
while t < tmax:
    abh = c*Jbh/(G*Mbh**2)
    if abh >= 0.9994:  # ------ Need to do this to respect cosmic censorship during the initial 1 msun collapse bit  --------
        abh = 0.9994
        Jbh = abh * (G*Mbh**2)/c

    M_ = Mbh * R_unit
    a_ = abh * M_
    Risco =  calculate_risco(M_, a_)
   
    # OmgK = sqrt(G*Mbh/Rd**3)
    OmgK = c * sqrt(M_) / (Rd**(3/2) + a_ * sqrt(M_))
    omega = c * 4*a_*M_ / (Rd**3 + Rd*a_**2 + 2*M_*a_**2)
    tvis = 1/alp_ss/np.abs(OmgK - omega)

    Rg = G*Mbh/c**2
    Rin = find_closest_r_in(M_, a_, 0.97,  Risco, rarr[-1])
    if Rin > Rd:
        Rin = Rd
    Rt = max(Rin, min(Rd, (2*Rg/Rd**s_PL * Md/ tvis / msun * 10**2.5)**(1./(1-s_PL))))

    gtt_factor = - (Rt**2 + a_**2 + 2*(a_**2)*M_/Rt) / (Rt**2 - 2*M_*Rt + a_**2)  # --- becomes important only when Rd is close to Risco.
    Mdotacc = Md / np.sqrt(-gtt_factor) / tvis

    eta = 1.063*abh**4 + 0.395*abh**2
    Macc = Mdotacc * (Rt/Rd)**s_PL
    P_BZ = eta * Macc  * c**2

    gtt_fact_in_var_form = lambda r_: -(2*a_**2*M_ + a_**2*r_ + r_**3) / ((a_**2 -2*M_*r_ + r_**2) * r_)
    result, error = integrate.quad(lambda r_: r_**(s_PL-1)/np.sqrt(-gtt_fact_in_var_form(r_)), Rt,Rd)
                                                                                      
    Mdot_wind = (result * Md * s_PL / tvis / Rd**s_PL)
    Mdot_lost = Mdotacc * (Rt/Rd)**s_PL  +  Mdot_wind
    
    result, error = integrate.quad(lambda r_: r_**(s_PL-1) * u_phi(M_, r_, a_) /np.sqrt(-gtt_fact_in_var_form(r_)), Rt,Rd)
    Jwdot = result * Md * s_PL / tvis / Rd**s_PL
    

    # --- ends up in the hole ----
    k = min(0.1  + 0.5*abh, 0.35)
    r_H = calculate_risco(M_, 1*M_)  #critically rotating 
    # r_H = G/c**2 *(Mbh + sqrt(M**2 - abh**2))
    # Omega_H = c * sqrt(M_) / (r_H**(3/2) + a_ * sqrt(M_))
    Omega_H = c*a_ / (2*M_*r_H)

    if Rt > Rin:
        Jbhdot =  (0.86 * Mbh * G / c) * Macc / fac1   - P_BZ / (k*Omega_H)
        Jnutrino_dot =  u_phi(M_, Rt, a_) * Mdotacc * (Rt/Rd)**s_PL / fac1  #- (0.86 * Mbh * G / c)  * Macc / fac1 +
    else:
        Jbhdot =  (0.86 * Mbh * G / c) * Macc / fac1   - P_BZ / (k*Omega_H)
        Jnutrino_dot =  (0.86 * Mbh * G / c)  * Macc / fac1 

    R_ergo = G/c**2 * (Mbh + sqrt(Mbh**2 - 0*abh**2))
    R_avg = 1.01*R_ergo
    gtt_fac =  np.sqrt(1 - 2*M_/R_avg) 
    # gtt_fac = sqrt(R_avg**2 - 2*M_*R_avg + a_**2) / sqrt(R_avg**2 + a_**2 + 2*(a_**2)*M_/R_avg)  # --- becomes important only when Rd is close to Risco.
    
    Mbhdot =  0.97 * Macc / fac1 -    P_BZ / c**2
    eta_acc = 0.03

    R_avg = 2.1*Risco   # to account for gravitional redshift
    gtt_fac_2 =  np.sqrt(1 - 2*M_/R_avg)
    # gtt_fac_2 = np.round(sqrt(R_avg**2 - 2*M_*R_avg + a_**2) / sqrt(R_avg**2 + a_**2 + 2*(a_**2)*M_/R_avg),4)  # --- becomes important only when Rd is close to Risco.
    Lacc = (0.01*eta_acc * gtt_fac_2* Macc / fac1 * c**2)  + P_BZ * gtt_fac
    Lj = (eta_acc * gtt_fac_2* Macc / fac1 * c**2)  + P_BZ * gtt_fac

    i_grid = calc_grid_pt(t, tgrid) 
    Rd_temp = rarr[i_grid]
    gamma = np.sqrt( (a_**2 / 2 + Rd_temp**2) / (a_**2 /2 - 2*M_*Rd_temp + Rd_temp**2) ) 
    fac1_ = gamma**3 * (a_**2 / Rd_temp**2 - 6*M_ / Rd_temp + 3)/3 
    fac2_ = gamma**2 * ( (a_/Rd_temp)**4 + 6*(a_/Rd_temp)**2 + 5 + 8*M_*a_**2/Rd_temp**3  - 10*a_*M_/Rd_temp**3/Omgarr[i_disk]) / 5

    Mfbdot = fac1_ * MJfbdot(t, tgrid, lgMfbdotgrid)
    Jfbdot = fac2_ * MJfbdot(t, tgrid, lgJfbdotgrid)

    Mddot = Mfbdot - Mdot_lost
    Jddot = Jfbdot - Jwdot  - (0.86 * Mbh * G / c) * Macc / fac1 #- Jnutrino_dot  # The last term considers Jbhdot as well (minus the jets)


    Mfbdot_rest_mass = MJfbdot(t, tgrid, lgMfbdotgrid)
    Jfbdot_rest_mass = MJfbdot(t, tgrid, lgJfbdotgrid)

    dt = dt_over_tvis * tvis
    Mfb_rest += Mfbdot_rest_mass * dt
    Jfb_rest += Jfbdot_rest_mass * dt

    Mfb += Mfbdot * dt
    Jfb += Jfbdot * dt

    fac1 = Mfb / Mfb_rest
    fac2 = Jfb / Jfb_rest

    # the upper disk radius rarr[-1] is set below by the radius of the collapsing star
    Rd =  find_closest_r(M_, a_, Md / fac1, Jd/ fac2, Risco, rarr[-1]) 
   

    temp[0].append(Risco / (G*Mbh/c**2))
    temp[1].append(Rd / (G*Mbh/c**2) )
    temp[2].append(Mdot_wind)
    temp[3].append(Jwdot)
    temp[5].append(Mbh)
    temp[4].append(t - tdisk)
    temp[6].append(Rt / (G*Mbh/c**2))
    temp[7].append(abh)    
    temp[8].append(Md)
    temp[9].append(Mfbdot)

    # dt = dt_over_tvis * tvis
    t += dt
    Md += Mddot * dt
    Jd += Jddot * dt
    Mbh += Mbhdot * dt
    Jbh += Jbhdot * dt
    Eacc += Lacc * dt
    Eaccj += Lj*dt
  
    if FLAG == True and t > 1.0025*tdisk:
        Lj = Eaccj /(t - tdisk)  # taking average
        rho_mean = Marr[-1] / (4/3 * np.pi * rarr[-1]**3)
        themin = (10/3)**(3/4) * (Lj/ (3*rarr[-1]**2 * rho_mean * c**3))**(1/4) 
        print('theta_cocoon [deg] = %.3f' %  (themin * 180/ np.pi), "L_jets [erg] = %.3e" % (Lj))
        
        fb_frac = cos(themin)   # fraction of mass outside 30-deg polar cones
        Mfbdot_arr = np.abs(np.diff(Marr))/np.abs(np.diff(tffarr)) * fb_frac
        Jfbdot_arr = np.abs(np.diff(Jarr))/np.abs(np.diff(tffarr)) * (cos(themin) - 1./3 * (cos(themin))**3) * 3./2
        # -------- interplote for a finner grid. Also on a log scale  -----------
        intp_lgMfbdot = interp1d(tmid, np.log10(Mfbdot_arr), fill_value='extrapolate')
        intp_lgJfbdot = interp1d(tmid, np.log10(Jfbdot_arr), fill_value='extrapolate')

        lgMfbdotgrid = np.empty(Ntgrid, dtype=float)
        lgJfbdotgrid = np.empty(Ntgrid, dtype=float)
        for i in range(Ntgrid):
            t_index = tgrid[i]
            if t_index > tmid[-1]:  # the disk never formed 
                lgMfbdotgrid[i] = -10.   # zero fallback rate
                lgJfbdotgrid[i] = -10.
            else:
                lgMfbdotgrid[i] = intp_lgMfbdot(t_index)
                lgJfbdotgrid[i] = intp_lgJfbdot(t_index)

        FLAG = False

print("Mbh [Msun] = %.3f" %  (Mbh/ msun), "abh = %.3f" % abh)
file = open('./output/rel_MAD', 'wb')
import pickle
# dump information to that file
pickle.dump(temp, file)
file.close()