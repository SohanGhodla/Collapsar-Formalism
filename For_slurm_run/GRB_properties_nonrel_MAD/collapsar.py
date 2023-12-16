# --------- This notebook assumes a nonrelativistic physics and except some modifications its 
# ---------------  a reproduction of the work of Fuller & Lu 2022 ----------------

import pylab as pl
import numpy as np
from math import sqrt, log10, pi, log, cos, floor
from scipy.interpolate import interp1d
# from mplchange import *
import mesa_reader as mr
import sys

if len(sys.argv) > 14:
    fdir = sys.argv[1]     # directory for the files
    fname = sys.argv[2]    # file name supplied by command line
    Mexp = sys.argv[3]
    AMexp = sys.argv[4]
    Mco = sys.argv[5]
    Rexp = sys.argv[6]
    MHe_form = sys.argv[7]
    AMfinal_He_form = sys.argv[8]
    R_He_form = sys.argv[9]
    Omega_mean_He_form = sys.argv[10]
    MHe_dep = sys.argv[11]
    AMfinal_He_dep = sys.argv[12]
    R_He_dep = sys.argv[13]
    Omega_mean_He_dep = sys.argv[14]
    file = sys.argv[15]
#    print('stellar model name: ' + fname)

# import pandas as pd
# dir_path = '/nesi/nobackup/uoa03218/test/LOGS/'
# fname = 'profile42'
# Mexp = '23.43'
# file = 'fsdfdf'
# AMexp = 1.0342859544608616e+52
# Mco = 9.6
# df_profile = pd.read_csv(dir_path + fname + '.data', delim_whitespace=True, header  = 4) 

machine_run = True

# adjustable parameters
s_PL = 0.5    # power-law index for the radial accretion rate profile in the advective regime
alp_ss = 0.01   # viscosity parameter
themin = 30*pi/180   # the minimum polar angle [rad] below which fallback is impeded by feedback
dt_over_tvis = 0.005     # time resolution factor

# some constants
c = 2.99792e10  # speed of light
G = 6.674e-8     # Newton's constant
rsun = 6.96e10         # solar radius
msun = 1.98847e33      # solar mass
R_unit = G*msun/c**2
J_unit = G*msun**2/c

# prof = mr.MesaData(dir_path + fname  + '.data')
prof = mr.MesaData(file_name=fdir+fname+'.data')
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
        # Jbh = abh * (G*Mbh**2)/c

    # \ell values below
    # ell = ellarr[i+1]

    ell_in_crg = ellarr[i+1]/c/Rg
    jisco_in_crg = jisco_over_crg(abh)
    if ell_in_crg > jisco_in_crg:
        # print(ell_in_crg * c*Rg, jisco_in_crg* c * Rg)
        # print('disk formaiton non-rel case')
        i_disk = i+1
        Mbh0 = Marr[i]
        Jbh0 =  abh*(G*Mbh**2)/c    # = Jarr[i]   ! By SG, because forcing the bh to have spin = abh then also requires lowering  the AM
        abh0 = abh
        break
    if i == Nr-3:
        if machine_run:
            print('%.3e\t%.3e\t%s\t%s\t%s\t%s\t%.3e\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3e' % (Mbh/msun, abh, AMexp, Mexp, file, tffarr[-1], 0, Mco, Rexp, \
                     MHe_form,  AMfinal_He_form, R_He_form,  Omega_mean_He_form,  MHe_dep,  AMfinal_He_dep, R_He_dep, Omega_mean_He_dep, 0.))
        else:
            print('no disk forms for this star!')
            print('final Mbh=%.3f Msun, abh=%.3f' % (Mbh/msun, abh))
        exit()
# the i_disk-th shell starts forming a disk
tdisk = tffarr[i_disk]

# fallback rate profile
fb_frac = cos(themin)   # fraction of mass outside 30-deg polar cones
Mfbdot = np.abs(np.diff(Marr))/np.abs(np.diff(tffarr)) * fb_frac
Jfbdot = np.abs(np.diff(Jarr))/np.abs(np.diff(tffarr)) * (cos(themin) - 1./3 * (cos(themin))**3) * 3./2

tmid = np.array([(tffarr[i]+tffarr[i+1])/2 for i in range(Nr-1)])
intp_lgMfbdot = interp1d(tmid, np.log10(Mfbdot), fill_value='extrapolate')
intp_lgJfbdot = interp1d(tmid, np.log10(Jfbdot), fill_value='extrapolate')

tmax = tffarr[-1]*1.5
Ntgrid =  2000     # interpolate lgMfbdot and lgJfbdot on a regular grid
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


tarr = []       # time
Mdarr = []      # disk mass
Rdarr = []      # outer disk radius
Riscoarr = []    # ISCO radius
Mbharr = []    # BH mass
Mfbdotarr = []  # mass fallback rate
Jfbdotarr = []  # AM fallback rate
Mbhdotarr = []   # BH mass gaining rate
Mdotaccarr = []   # outer disk accretion rate
Liscoarr = []    # accretion power near isco
Lnuiscoarr = []     # neutrino power near isco
Lwiscoarr = []      # wind power near isco
Lwarr = []    # total wind power for the entire disk

Mbh = Mbh0
Jbh = Jbh0
abh = c*Jbh/(G*Mbh**2)

# #BY SG
if abh >= 0.9994:
    abh = 0.9994
    Jbh = abh * (G*Mbh**2)/c
    
Rg = G*Mbh/c**2
Risco = risco_over_rg(abh) * Rg
OmgKisco = sqrt(G*Mbh/Risco**3)
tvis_isco = 1/alp_ss/OmgKisco

# initialize the disk properties (unimportant for the total energetics)
Md0 = MJfbdot(tdisk, tgrid, lgMfbdotgrid)*tvis_isco*0.5
Rd0 = 1.1*Risco
Jd0 = sqrt(G*Mbh*Rd0) * Md0   # initial disk amgular momentum

Md = Md0
Jd = Jd0
t = tdisk

# print('disk formation time = ', tdisk)
# temp = [ [] for i in range(8)]
Eacc = 0.
while t < tmax:
    abh = c*Jbh/(G*Mbh**2)
    # if abh > 1: 
    #     print('oh no this was supposed to be censored')
    if abh >= 0.9994:  # ------ Need to do this to respect cosmic censorship during the initial 1msun collapse bit  --------
        abh = 0.9994
        Jbh = abh * (G*Mbh**2)/c

    Rg = G*Mbh/c**2
    Risco = risco_over_rg(abh) * Rg
    Rd = (Jd/Md)**2/(G*Mbh)   # Newtonian Keplerian rotation
    OmgK = sqrt(G*Mbh/Rd**3)
    tvis = 1/alp_ss/OmgK
    Mdotacc = Md/tvis
    Rt = max(Risco, min(Rd, (2*Rg/Rd**s_PL * Mdotacc/msun * 10**2.5)**(1./(1-s_PL))))

    # temp[0].append(Risco /  (G*Mbh/c**2) )
    # temp[1].append(Rd/  (G*Mbh/c**2) )
    # temp[2].append(Mdotacc)

    Mbhdot = sqrt(1 -2*Rg/(3*Risco)) * Mdotacc * (Rt/Rd)**s_PL  # By SG because because a part of energy is lost so reduce the total rest mass energy.
    Jwdot = 2*s_PL/(2*s_PL + 1) * sqrt(G*Mbh*Rd) * Mdotacc *\
        (1 - (Rt/Rd)**((2*s_PL+1)/2))
    # temp[3].append(Jwdot)
    # temp[4].append(t)
    # temp[5].append(Mbh)
    # temp[6].append(Rt / (G*Mbh/c**2))
    # temp[7].append(abh)
    # Jbhdot = jisco_over_crg(abh) * Rg * c * Mdotacc * (Rt/Rd)**s_PL #Mbhdot


    # ----- comment these if MAD is not needed .............
    M_ = Mbh * R_unit
    a_ = abh * M_
    k = min(0.1  + 0.5*abh, 0.35)
    r_H = risco_over_rg(1) * Rg  #critically rotating 
    # r_H = G/c**2 *(Mbh + sqrt(M**2 - abh**2))
    # Omega_H = c * sqrt(M_) / (r_H**(3/2) + a_ * sqrt(M_))
    eta = 1.063*abh**4 + 0.395*abh**2
    P_BZ = eta * Mdotacc * (Rt/Rd)**s_PL  * c**2
    Omega_H = c*a_ / (2*M_*r_H)
    Jbhdot =  0.86 * Mdotacc * (Rt/Rd)**s_PL   - P_BZ / (k*Omega_H)#

    Mfbdot = MJfbdot(t, tgrid, lgMfbdotgrid)
    Jfbdot = MJfbdot(t, tgrid, lgJfbdotgrid)
    Mddot = Mfbdot - Mdotacc
    Jddot = Jfbdot - Jwdot - Jbhdot

    # ---- comment the former bit and uncomment the latter bit if MAD is not needed ------------
    Mbhdot =  0.97 * Mdotacc * (Rt/Rd)**s_PL  -  P_BZ / c**2#(1 - sqrt(2*Rg/(3*Risco))) * Mdotacc * (Rt/Rd)**s_PL  
    eta_acc = 0.03
    Lacc = 0.01*eta_acc * Mdotacc * (Rt/Rd)**s_PL * c**2 + P_BZ 

    # --- uncomment this and comment the above bit if MAD not needed ----
    # Lacc = (1 - (1 - sqrt(2*Rg/(3*Risco))))* Mdotacc * (Rt/Rd)**s_PL * c**2

    tarr += [t]
    Mdarr += [Md/msun]
    Rdarr += [Rd]
    Riscoarr += [Risco]
    Mbharr += [Mbh/msun]
    Mfbdotarr += [Mfbdot/msun]
    Jfbdotarr += [Jfbdot/J_unit]
    Mbhdotarr += [Mbhdot/msun]
    Mdotaccarr += [Mdotacc/msun]
    dt = dt_over_tvis * tvis
    t += dt
    Md += Mddot * dt
    Jd += Jddot * dt
    Mbh += Mbhdot * dt
    Jbh += Jbhdot * dt

    Eacc += Lacc * dt
# file = open('nonrel', 'wb')
# import pickle
# # dump information to that file
# pickle.dump(temp, file)

# # close the file
# file.close()

if machine_run:
    print('%.3e\t%.3e\t%s\t%s\t%s\t%s\t%.3e\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.3e' % (Mbh/msun, abh, AMexp, Mexp, file, tffarr[-1], Md/msun, Mco, Rexp, \
                     MHe_form,  AMfinal_He_form, R_He_form,  Omega_mean_He_form,  MHe_dep,  AMfinal_He_dep, R_He_dep, Omega_mean_He_dep, Eacc))
    