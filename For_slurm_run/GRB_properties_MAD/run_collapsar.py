#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import subprocess
import tempfile


dir_path = '/nesi/nobackup/uoa03218/Remnant_mass_spectrum_MESA_1/Symbolic_link'#'/nesi/nobackup/uoa03218/Publ_CHE_models/Rot_models_for_CHE/Tui_Masses_CHE/Makemodels_He_MS/'
# dir_path = '/nesi/nobackup/uoa03218/Helium_stars_Dutch_half/Makemodels_He_MS/'

# metal = ['zem5','zem4', '3zem4', '6zem4', 'z001', 'z002','z003', 'z004']#, 'z005', 'z006']#, 'z006', 'z007', 'z008']
# # mass =  [20, 21, 22, 23, 24, 25, 35, 40, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250 , 300]
# mass = np.arange(15,300,5.)

Datafile = open('./Run_data_1.txt' , 'w')

processes = []
failing_symlink = [43, 52, 56, 91, 94, 96, 129, 131]
for symlink in range(0,342):  #len(metal)):
  if symlink not in failing_symlink:
    file_path = f'{dir_path}/{symlink}/LOGS/'
    if os.path.exists(f'{file_path}/history.data'):
      df_p_index = pd.read_csv(file_path + 'profiles.index', delim_whitespace=True)
      
      #recording the last profile number
      last_profile = df_p_index.loc[len(df_p_index)-1,'lines']
      df_profile = pd.read_csv(file_path + 'profile' + str(last_profile) + '.data', delim_whitespace=True, header  = 4) 
      
      omega_mean_exp = np.mean(df_profile.loc[:,'omega'])

      # Checking for pair instability
      df_history = pd.read_csv(file_path + 'history.data', delim_whitespace=True, header  = 4)
      # AMfinal = 10**df_history.loc[len(df_history)-1, 'log_total_angular_momentum']
      Mco = df_history.loc[len(df_history) - 1,'co_core_mass']
      Mexp = df_profile.loc[0,'mass']
      Rexp = 10**(df_history.loc[len(df_history)-1, 'log_R'])
      AM_exp = 10**(df_history.loc[len(df_history) - 1,'log_total_angular_momentum'])

      mask = df_history["center_he4"] <= 0.985
      # mask = (df_history["center_he4"] < .05) & (df_history["center_c12"] < .5)
      R_He_form = 10**np.max(df_history.loc[mask, 'log_R'])
      MHe_form = np.max(df_history.loc[mask, 'star_mass'])
      AMfinal_He_form = 10**np.max(df_history[mask]['log_total_angular_momentum'])
      # AMfinal_He_form = 10**(np.max(np.array(AMfinal)))

      # finding the profile which has core helium composition below 98%
      mod_num = np.min(df_history.loc[mask, 'model_number'])
      df_p_index = df_p_index.dropna(axis = 1)
      df_p_index.set_axis(labels=["model_num", "Priority", "Profile_num"], axis=1, inplace=True)
      mask = df_p_index['model_num'] >= mod_num
      count = -1
      for i in mask:
        count +=1
        if i == True:
          profile_num = df_p_index.loc[count,'Profile_num']
          break

      # profile_num = np.min(df_p_index.loc[mask,'Profile_num'])

      # Finding mean angular velocity when the core helium composition drops below 10% -- below is a new file
      df_profile = pd.read_csv(file_path + 'profile' + str(profile_num) + '.data', delim_whitespace=True, header  = 4) 
      omega_mean_He_form = np.mean(df_profile.loc[:,'omega'])


      mask = df_history["center_he4"] < 0.01
      # mask = (df_history["center_he4"] < .05) & (df_history["center_c12"] < .5)
      R_He_dep = 10**np.max(df_history.loc[mask, 'log_R'])
      MHe_dep = np.max(df_history.loc[mask, 'star_mass'])
      AMfinal_He_dep = 10**np.max(df_history[mask]['log_total_angular_momentum'])
      # AMfinal_He_form = 10**(np.max(np.array(AMfinal)))

      # finding the profile which has core helium composition below 1%
      mod_num = np.min(df_history.loc[mask, 'model_number'])
      mask = df_p_index['model_num'] >= mod_num
      count = -1
      for i in mask:
        count +=1
        if i == True:
          profile_num = df_p_index.loc[count,'Profile_num']
          break

      # profile_num = np.min(df_p_index.loc[mask,'Profile_num'])

      # Finding mean angular velocity when the core helium composition drops below 10% -- below is a new file
      df_profile = pd.read_csv(file_path + 'profile' + str(profile_num) + '.data', delim_whitespace=True, header  = 4) 
      omega_mean_He_dep = np.mean(df_profile.loc[:,'omega'])


    # ----------------- core carbon depletion ----------------
    mask = (df_history["center_he4"] < 0.01) & (df_history["center_c12"] < 0.01)
      # mask = (df_history["center_he4"] < .05) & (df_history["center_c12"] < .5)
    R_c12_dep = 10**np.max(df_history.loc[mask, 'log_R'])
    Mc12_dep = np.max(df_history.loc[mask, 'star_mass'])
    AMfinal_c12_dep = 10**np.max(df_history[mask]['log_total_angular_momentum'])
    
    if(sum(mask) == 0):
      R_c12_dep = Rexp
      Mc12_dep = Mexp
      AMfinal_c12_dep = AM_exp

    # finding the profile which has core helium composition below 1%
    mod_num = np.min(df_history.loc[mask, 'model_number'])
    mask = df_p_index['model_num'] >= mod_num
    count = -1
    for i in mask:
      count +=1
      if i == True:
        profile_num = df_p_index.loc[count,'Profile_num']
        break
      
      profile_num = last_profile

    # profile_num = np.min(df_p_index.loc[mask,'Profile_num'])

    # Finding mean angular velocity when the core helium composition drops below 10% -- below is a new file
    df_profile = pd.read_csv(file_path + 'profile' + str(profile_num) + '.data', delim_whitespace=True, header  = 4) 
    omega_mean_c12_dep = np.mean(df_profile.loc[:,'omega'])
    # print(file_path, str(Mexp), str('%.3e' % AM_exp), str(Mco), str(Rexp), \
    #       str(MHe_form), str('%.3e' % AMfinal_He_form), str(R_He_form), str(omega_mean_He_form), str(MHe_dep), str('%.3e' % AMfinal_He_dep), \
    #               str(R_He_dep), str(omega_mean_He_dep), str(Mc12_dep), str('%.3e' % AMfinal_c12_dep), \
    #               str(R_c12_dep), str(omega_mean_c12_dep), str(omega_mean_exp),'\n')

    # if df_history.loc[len(df_history)-1,'He_core'] >= 65. and  df_history.loc[len(df_history)-1,'He_core'] < 135.:
      #   continue  #then ignore the following

    fname = 'profile' + str(last_profile) 
    f = tempfile.TemporaryFile()

    p = subprocess.Popen(['python', 'collapsar.py', file_path, fname, str(Mexp), str('%.3e' % AM_exp), str(Mco), str(Rexp), \
          str(MHe_form), str('%.3e' % AMfinal_He_form), str(R_He_form), str(omega_mean_He_form), str(MHe_dep), str('%.3e' % AMfinal_He_dep), \
                  str(R_He_dep), str(omega_mean_He_dep), str(Mc12_dep), str('%.3e' % AMfinal_c12_dep), \
                  str(R_c12_dep), str(omega_mean_c12_dep), str(omega_mean_exp), f'{symlink}/LOGS/profile{last_profile}'], stdout=f)
    processes.append((p,f))
    # Mbh, abh, Ewind, Eacc, Eacc_ADAF = [float(x) for x in output]
    # print('%.3e\t%.3e\t%.3e\t%.3e\t%.3e' % (Mbh, abh, Ewind, Eacc, Eacc_ADAF))
    # Datafile.write('%.3e\t%.3e\t%.3e\t%.3e\t%.3e' % (Mbh, abh, Ewind, Eacc, Eacc_ADAF))

Datafile.write('Mbh\t\t abh\t\t AMexp\t\t  Mexp\t\t filename\t\t tff\t\t\t Md\t\t\t Mco\t\t\t Rexp\t\t\t  MHe_form\t\t\t  AMfinal_He_form\t\t\t R_He_form\t\t  Omega_mean_He_form\t\t  MHe_dep\t\t  AMfinal_He_dep\t\t R_He_dep\t\t  Omega_mean_He_dep\t Omega_mean_exp\t Mc12_dep\t\t AM_c12_dep\t\t R_c12_dep\t\t Omega_c12_dep\t\t Eacc\n')
for p,f in processes:
  p.wait()
  f.seek(0)
  Datafile.write((f.read().decode('utf-8')))
  f.close()
Datafile.close()

