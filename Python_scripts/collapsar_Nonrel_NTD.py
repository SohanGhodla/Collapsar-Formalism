import pylab as pl
import numpy as np
from math import sqrt, log10, pi, log, cos, floor
from scipy.interpolate import interp1d
# from mplchange import *
import mesa_reader as mr
import sys
import pandas as pd

machine_run = False
dir_path = '../data/MESA_LOGS/0.45/'

# adjustable parameters
s_PL = 0.5    # power-law index for the radial accretion rate profile in the advective regime
alp_ss = 0.01    # viscosity parameter
themin = 30*pi/180   # the minimum polar angle [rad] below which fallback is impeded by feedback
dt_over_tvis = 0.005    # time resolution factor

# some constants
c = 2.99792e10  # speed of light
G = 6.674e-8     # Newton's constant
rsun = 6.96e10         # solar radius
msun = 1.98847e33      # solar mass
R_unit = G/c**2
J_unit = G*msun**2/c

prof = mr.MesaData(file_name= dir_path + 'profile82' + '.data')

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
Nr = 2500
rarr = np.logspace((rdat[0]), (rdat[-1]), Nr)  # right boundary of shell
# rhoarr = np.array([10**intp_lgrho(log10(r)) for r in rarr])
# Omgarr = np.array([10**intp_lgomg(log10(r)) for r in rarr])
Marr = np.array([10**intp_lgmass(log10(r)) for r in rarr])
Jarr = np.array([10**intp_lgAM(log10(r)) for r in rarr])


def risco_over_rg(a):   # ISCO radius
    z1 = 1 + (1-a*a)**(1./3) * ((1+a)**(1./3) + (1-a)**(1./3))
    z2 = (3*a*a + z1*z1)**0.5
    if a > 0:
        return 3+z2-((3-z1)*(3+z1+2*z2))**0.5
    return 3+z2+((3-z1)*(3+z1+2*z2))**0.5


def jisco_over_crg(a):      # specific AM at ISCO
    r = risco_over_rg(a)
    if a > 0:
        if a == 1:
            return (r**1.5 + r + r**0.5 - 1)/r**(3./4)/(r**0.5 + 2)**0.5
        return (r**2 - 2*a*r**0.5 + a**2)/r**(3./4)/(r**1.5 - 3*r**0.5 + 2*a)**0.5

    if a == -1:
        return (r**1.5 - r + r**0.5 + 1)/r**(3./4)/(r**0.5 - 2)**0.5
    return (r**2 + 2*a*r**0.5 + a**2)/r**(3./4)/(r**1.5 - 3*r**0.5 - 2*a)**0.5

# ----------------------------
def u_phi(M, r, a):
    numerator = sqrt(M) * (r**2 - 2 * a * sqrt(M) * r**(1/2) + a**2)
    denominator = r**(3/4) * sqrt(r**(3/2) - 3 * M * r**(1/2) + 2 * a * sqrt(M))
    u_phi_r = numerator / denominator
    return u_phi_r * c


tffarr = pi/2**1.5 * np.sqrt(rarr**3/G/Marr)
# ellarr =  Jarr / Marr #2./3 * Omgarr * rarr**2
ellarr = np.diff(Jarr)/np.diff(Marr)

# need to find the time when accretion disk forms
Mbh0, abh0, Jbh0, i_disk = 0., 0., 0., 0
for i in range(Nr-2):
    # ----- assume that the innermost 1 Msun rest mass always forms a black hole  --------
    if Marr[i] < msun:   
        continue

    Mbh = Marr[i]
    Rg = G*Mbh/c**2
    abh = c*Jarr[i]/(G*Mbh**2)
    
    if abh >= 0.9994:  # ------ Need to do this to respect cosmic censorship during the initial 1msun collapse bit  --------
        abh = 0.9994
        Jbh = abh * (G*Mbh**2)/c

    # \ell values below
    ell_in_crg = ellarr[i+1]/c/Rg
    jisco_in_crg = jisco_over_crg(abh)
    if ell_in_crg > jisco_in_crg:
        # print('disk formaiton non-rel case')
        i_disk = i+1
        Mbh0 = Marr[i]
        Jbh0 =  abh*(G*Mbh**2)/c    # = Jarr[i]   ! By SG, because forcing the bh to have spin = abh then also requires lowering  the AM
        abh0 = abh
        break
    if i == Nr-3:
        print('no disk forms for this star!')
        print('final Mbh=%.3f Msun, abh=%.3f' % (Mbh/msun, abh))
        exit()
# the i_disk-th shell starts forming a disk
tdisk = tffarr[i_disk]

#  -------- fallback rate profile ------------
Mfbdot = np.diff(Marr)/np.diff(tffarr) 
Jfbdot = np.diff(Jarr)/np.diff(tffarr) 

tmid = np.array([(tffarr[i]+tffarr[i+1])/2 for i in range(Nr-1)])
intp_lgMfbdot = interp1d(tmid, np.log10(Mfbdot), fill_value='extrapolate')
intp_lgJfbdot = interp1d(tmid, np.log10(Jfbdot), fill_value='extrapolate')

tmax = tffarr[-1]*2
Ntgrid = 2000     # interpolate lgMfbdot and lgJfbdot on a regular grid
tgrid = np.linspace(tdisk, tmax, Ntgrid)
dtgrid = tgrid[1] - tgrid[0]
lgMfbdotgrid = np.empty(Ntgrid, dtype=float)
lgJfbdotgrid = np.empty(Ntgrid, dtype=float)
for i in range(Ntgrid):
    t = tgrid[i]
    if t > tmid[-1]:
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

Mbh = Mbh0
Jbh = Jbh0
abh = c*Jbh/(G*Mbh**2)
    
Rg = G*Mbh/c**2
Risco = risco_over_rg(abh) * Rg
OmgKisco = sqrt(G*Mbh/Risco**3)
tvis_isco = 1/alp_ss/OmgKisco

# initialize the disk properties (unimportant for the total energetics)
Md0 = MJfbdot(tdisk, tgrid, lgMfbdotgrid)*tvis_isco*0.5
Rd0 = 1.01*Risco
Rd = Rd0
Jd0 = sqrt(G*Mbh*Rd0) * Md0   # initial disk amgular momentum

Md = Md0
Jd = Jd0
t = tdisk

print('Mass of star at collapse [Msun] = %.3f' % (Marr[-1]/msun))
print('disk formation time [s] = %.3f' %  tdisk)

Eacc = 0.
FLAG = True
temp = [ [] for i in range(10)]
while t < tmax:
    abh = c*Jbh/(G*Mbh**2)
    # ------ To respect cosmic censorship during the initial 1msun collapse bit  --------
       
    if abh >= 0.9994:  
        abh = 0.9994
        Jbh = abh * (G*Mbh**2)/c

    Rg = G*Mbh/c**2
    Risco = risco_over_rg(abh) * Rg
    Rd = (Jd/Md)**2/(G*Mbh)   # Newtonian Keplerian rotation
    OmgK = sqrt(G*Mbh/Rd**3)
    tvis = 1/alp_ss/OmgK
    Mdotacc = Md/tvis

    Rt = max(Risco, min(Rd, (2*Rg/Rd**s_PL * Mdotacc/msun * 10**2.5)**(1./(1-s_PL))))
    Mbhdot = sqrt(1 -2*Rg/(3*Risco)) * Mdotacc * (Rt/Rd)**s_PL  # By SG because because a part of energy is lost so reduce the total rest mass energy.
    Jwdot = 2*s_PL/(2*s_PL + 1) * sqrt(G*Mbh*Rd) * Mdotacc * (1 - (Rt/Rd)**((2*s_PL+1)/2))
  
    Mdot_wind = Mdotacc - Mdotacc * (Rt/Rd)**s_PL
    
    M_ = Mbh * R_unit
    a_ = abh * M_
    Jbhdot =  jisco_over_crg(abh) * Rg * c * Mdotacc * (Rt/Rd)**s_PL
    Jnutrino_dot =  u_phi(M_, Rt, a_) * Mdotacc * (Rt/Rd)**s_PL - Jbhdot 

    Mfbdot = MJfbdot(t, tgrid, lgMfbdotgrid)
    Jfbdot = MJfbdot(t, tgrid, lgJfbdotgrid)

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

    Mddot = Mfbdot - Mdotacc

    Jddot = Jfbdot - Jwdot - Jbhdot  #- Jnutrino_dot  #
    Lw = 0.5*s_PL/(1-s_PL)*G*Mbh/Rd * Mdotacc * ((Rd/Rt)**(1-s_PL) - 1)

    R_avg = 2.1*Risco
    gtt_fac =  np.sqrt(1 - 2*M_/R_avg)   # sqrt(R_avg**2 - 2*M_*R_avg + a_**2) / sqrt(R_avg**2 + a_**2 + 2*(a_**2)*M_/R_avg)  # --- becomes important only when Rd is close to Risco.
    eta_acc = (1 - sqrt(1 - 2*Rg/3./Risco) ) * gtt_fac
    Lacc = Mbhdot*c**2*eta_acc

    dt = dt_over_tvis * tvis
    t += dt
    Md += Mddot * dt
    Jd += Jddot * dt
    Mbh += Mbhdot * dt
    Jbh += Jbhdot * dt
    Eacc += Lacc * dt

    if FLAG == True and t > 1.0025*tdisk:
        Lj = Eacc /(t - tdisk)  # taking average
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

file = open('./output/nonrel_NTD', 'wb')
import pickle
# dump information to that file
pickle.dump(temp, file)
file.close()


