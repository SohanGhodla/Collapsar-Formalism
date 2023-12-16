import os
import numpy as np

symlink = [0, 150, 300, 450, 600,750, 911]# 900, 1100, 1250, 1439]
for Z in range(len(symlink) - 1):
  os.system('rm ./run_collapsar_' + str(symlink[Z+1]) + '.py')
  print('removed old ./run_collapsar_' + str(symlink[Z+1]) + '.py')
  os.system('cp ./run_collapsar.py run_collapsar_' + str(symlink[Z+1]) + '.py')
  print('--> made new ./run_collapsar_' + str(symlink[Z+1]) + '.py')

  os.system('rm ./GRB_sbatch_' + str(symlink[Z+1]) + '.sh')
  # print('removed old ./run_collapsar_' + str(Z+1))
  os.system('cp ./GRB_sbatch.sh GRB_sbatch_' + str(symlink[Z+1]) + '.sh')
  # print('--> made new ./run_collapsar_' + str(Z+1))
 

  INLIST = open('./run_collapsar.py', 'r')
  TFILE = open('./run_collapsar_' + str(symlink[Z+1]) + '.py', 'w')
  datalines = INLIST.read()
  for line in datalines.split('\n'):
    check = 0

    if './Run_data_1.txt' in line:
            TFILE.write(f'Datafile = open(\'./Run_data_{symlink[Z+1]}.txt\' , \'w\')' + '\n')
            check = 1
            
    if 'for symlink in range(0,342):' in line:
      TFILE.write(f'for symlink in range({symlink[Z]},{symlink[Z+1]}):' + '\n')
      check = 1
    if check < 0.5:
      TFILE.write(line + '\n')
  INLIST.close()
  TFILE.close()

  INLIST = open('./GRB_sbatch.sh', 'r')
  TFILE = open('./GRB_sbatch_' + str(symlink[Z+1]) + '.sh', 'w')
  datalines = INLIST.read()
  for line in datalines.split('\n'):
    check = 0
    if 'run_collapsar.py' in line:
      TFILE.write(f'python run_collapsar_{symlink[Z+1]}.py' + '\n')
      check = 1
    if check < 0.5:
      TFILE.write(line + '\n')
  INLIST.close()
  TFILE.close()
      
  os.system('sbatch ./GRB_sbatch_' + str(symlink[Z+1]) + '.sh')
  # os.chdir('../')