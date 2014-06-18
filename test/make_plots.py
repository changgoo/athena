#!/usr/bin/python

# Runs and generates plots for predefined problems.
# Compares to old Athena in some cases

# Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc
import argparse
import glob
import os
import subprocess

# Main function
def main(**kwargs):

  # Plot settings
  rc('text', usetex=True)
  rc('text.latex', preamble='\usepackage{color}')

  # Extract inputs
  plots_needed = not kwargs['computation_only']
  computation_needed = not kwargs['plot_only']

  # Case out on problem
  problem = kwargs['problem']
  if problem == 'hydro_sr_shockset':  # relativistic hydro shocks
    if computation_needed:
      settings = [['1', '400', '0.4', '0.4', 'prim'],
                  ['2', '400', '0.4', '0.4', 'prim'],
                  ['3', '400', '0.4', '0.4', 'prim'],
                  ['4', '400', '0.4', '0.4', 'prim']]
      run_old('mb', 'hydro_sr_old_', settings)
      run_new('mb_', 'hydro_sr_new_', settings)
    if plots_needed:
      plot_shockset('plots/hydro_sr_shockset')
  else:
    print('ERROR: problem not recognized')

# Function for plotting shock set
def plot_shockset(filename):

  # Read data
  print('Reading data...')
  data_old_1 = read_athena('data/hydro_sr_old_1.0001.tab',
      ['x', 'rho', 'vx', 'vy', 'vz', 'pgas'])
  data_old_2 = read_athena('data/hydro_sr_old_2.0001.tab',
      ['x', 'rho', 'vx', 'vy', 'vz', 'pgas'])
  data_old_3 = read_athena('data/hydro_sr_old_3.0001.tab',
      ['x', 'rho', 'vx', 'vy', 'vz', 'pgas'])
  data_old_4 = read_athena('data/hydro_sr_old_4.0001.tab',
      ['x', 'rho', 'vx', 'vy', 'vz', 'pgas'])
  data_new_1 = read_athena('data/hydro_sr_new_1.0001.tab',
      ['x', 'rho', 'pgas', 'vx', 'vy', 'vz'])
  data_new_2 = read_athena('data/hydro_sr_new_2.0001.tab',
      ['x', 'rho', 'pgas', 'vx', 'vy', 'vz'])
  data_new_3 = read_athena('data/hydro_sr_new_3.0001.tab',
      ['x', 'rho', 'pgas', 'vx', 'vy', 'vz'])
  data_new_4 = read_athena('data/hydro_sr_new_4.0001.tab',
      ['x', 'rho', 'pgas', 'vx', 'vy', 'vz'])

  # Plot data
  print('Plotting data...')
  plot_shockset_aux(2, 2, 1, data_old_1, data_new_1, [10, 25, 1],
      [-0.1, 1.1], [0.0, 1.0, 6], 4)
  plot_shockset_aux(2, 2, 2, data_old_2, data_new_2, [12, 25, 1],
      [-0.7, 1.0], [-0.5, 1.0, 4], 5)
  plot_shockset_aux(2, 2, 3, data_old_3, data_new_3, [10, 20, 1],
      [-0.1, 1.1], [0.0, 1.0, 6], 4)
  plot_shockset_aux(2, 2, 4, data_old_4, data_new_4, [10, 1000, 1],
      [-0.1, 1.1], [0.0, 1.0, 6], 4)

  # Save figure
  plt.tight_layout()
  plt.savefig(filename)

# Auxiliary function for plotting shock set
def plot_shockset_aux(rows, cols, position, data_old, data_new, divisors, y_limits,
    y_ticks, y_subdivisions):

  # Plot old data
  plt.subplot(rows, cols, position)
  plt.plot(data_old['x'], data_old['rho']/float(divisors[0]), 'r', alpha=0.25, lw=2.0)
  plt.plot(data_old['x'], data_old['pgas']/float(divisors[1]), 'g', alpha=0.25, lw=2.0)
  plt.plot(data_old['x'], data_old['vx']/float(divisors[2]), 'b', alpha=0.25, lw=2.0)

  # Plot new data
  plt.plot(data_new['x'], data_new['rho']/divisors[0],
      'ro', markeredgecolor='r', ms=0.8)
  plt.plot(data_new['x'], data_new['pgas']/divisors[1],
      'go', markeredgecolor='g', ms=0.8)
  plt.plot(data_new['x'], data_new['vx']/divisors[2],
      'bo', markeredgecolor='b', ms=0.8)

  # Set ticks
  plt.xlim([-0.5, 0.5])
  plt.xticks(np.linspace(-0.5, 0.5, 6))
  plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
  plt.ylim(y_limits)
  plt.yticks(np.linspace(y_ticks[0], y_ticks[1], y_ticks[2]))
  plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(y_subdivisions))

  # Set labels
  plt.xlabel(r'$x$')
  divisor_strings = []
  for divisor in divisors:
    if divisor != 1.0:
      divisor_strings.append('/' + repr(divisor))
    else:
      divisor_strings.append('')
  plt.ylabel((r'$\rho{0}\ \mathrm{{(r)}},\ ' +
      r'p_\mathrm{{g}}{1}\ \mathrm{{(g)}},\ ' +
      r'v_x{2}\ \mathrm{{(b)}}$').\
      format(divisor_strings[0], divisor_strings[1], divisor_strings[2]))

# Function for running old Athena
def run_old(input_prefix, output_prefix, settings):

  # Prepare strings
  old_athena_env = 'ATHENA_ROOT'
  try:
    old_directory = os.environ['ATHENA_ROOT']
  except KeyError:
    print('ERROR: {0} must be set to the directory containing old Athena'.\
        format(old_athena_env))
    exit()
  old_configure_string = './configure --with-problem=shkset1d --with-gas=hydro \
      --enable-special-relativity --with-integrator=vl --with-order=2p --with-flux=hllc'
  old_run_string = 'bin/athena -i tst/1D-sr-hydro/athinput.' + input_prefix + '{1} \
      -d {0}/data job/problem_id=' + output_prefix + '{1} job/maxout=1 \
      output1/dat_fmt=%24.16e output1/dt={3} output1/out={5} time/cour_no={4} \
      time/nlim=100000 time/tlim={3} domain1/Nx1={2} domain1/x1min=-0.5 \
      domain1/x1max=0.5'

  # Generate data
  print('deleting old Athena data...')
  try:
    data_files = glob.glob('data/{0}*.tab'.format(output_prefix))
    rm_command = 'rm -f'.split()
    rm_command.extend(data_files)
    subprocess.call(rm_command)
  except OSError as err:
    print('OS Error ({0}): {1}'.format(err.errno, err.strerror))
    exit()
  print('running old Athena...')
  try:
    current_directory = os.getcwd()
    os.chdir(old_directory)
    os.system('make clean &> /dev/null')
    os.system(old_configure_string+' &> /dev/null')
    os.system('make all &> /dev/null')
    for case in settings:
      os.system(old_run_string.format(current_directory, case[0], case[1], case[2],
          case[3], case[4]) + ' &> /dev/null')
    os.chdir(current_directory)
  except OSError as err:
    print('OS Error ({0}): {1}'.format(err.errno, err.strerror))
    exit()

# Function for running new Athena
def run_new(input_prefix, output_prefix, settings):

  # Prepare strings
  new_make_string = 'make all COORDINATES_FILE=cartesian.cpp \
      CONVERT_VAR_FILE=adiabatic_hydro_sr.cpp PROBLEM_FILE=shock_tube_sr.cpp \
      RSOLVER_FILE=hllc_sr.cpp RECONSTRUCT_FILE=plm.cpp'
  # TODO: change when -d option works
  #new_run_string = './athena -i inputs/hydro_sr/athinput.' + input_prefix + '{1} \
  #    -d {0}/data job/problem_id=' + output_prefix + '{1} output1/variable={5} \
  #    output1/data_format=%24.16e output1/dt={3} time/cfl_number={4} time/nlim=-1 \
  #    time/tlim={3} mesh/nx1={2}'
  new_run_string = './athena \
      -i ../inputs/hydro_sr/athinput.' + input_prefix + '{1} \
      job/problem_id=' + output_prefix + '{1} output1/variable={5} \
      output1/data_format=%24.16e output1/dt={3} time/cfl_number={4} time/nlim=-1 \
      time/tlim={3} mesh/nx1={2}'

  # Generate data
  print('deleting new Athena data...')
  try:
    data_files = glob.glob('data/{0}*.tab'.format(output_prefix))
    rm_command = 'rm -f'.split()
    rm_command.extend(data_files)
    subprocess.call(rm_command)
    # TODO: remove when -d option works
    data_files = glob.glob('../bin/{0}*.tab'.format(output_prefix))
    rm_command = 'rm -f'.split()
    rm_command.extend(data_files)
    subprocess.call(rm_command)
  except OSError as err:
    print('OS Error ({0}): {1}'.format(err.errno, err.strerror))
    exit()
  print('running new Athena...')
  try:
    current_directory = os.getcwd()
    os.chdir('..')
    os.system('make clean &> /dev/null')
    os.system(new_make_string + ' &> /dev/null')
    os.chdir('bin')
    for case in settings:
      os.system(new_run_string.format(current_directory, case[0], case[1], case[2],
          case[3], case[4]) + ' &> /dev/null')
      # TODO: remove when -d option works
      os.system('mv {0}*.tab {1}/data/.'.format(output_prefix, current_directory))
    os.chdir(current_directory)
  except OSError as err:
    print('OS Error ({0}): {1}'.format(err.errno, err.strerror))
    exit()

# Function for reading Athena data
def read_athena(filename, headings=None):

  # Read raw data
  with open(filename, 'r') as data_file:
    raw_data = data_file.readlines()

  # Create array of data
  data = []
  for line in raw_data:
    if line.split()[0][0] == '#':
      continue
    row = []
    for val in line.split()[1:]:
      row.append(float(val))
    data.append(row)
  data = np.array(data)

  # Create dict if desired
  if headings is not None:
    data_dict = {}
    for i in range(len(headings)):
      data_dict[headings[i]] = data[:,i]
    return data_dict
  else:
    return data

# Execute main function
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('problem', type=str, help='name of problem')
  parser.add_argument('-p', '--plot_only', action='store_true', default=False,
      help='flag indicating computations are not to be redone')
  parser.add_argument('-c', '--computation_only', action='store_true', default=False,
      help='flag indicating no plots are to be made')  
  args = parser.parse_args()
  main(**vars(args))
