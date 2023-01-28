import numpy as np
from astropy.io import fits
import yaml

import sys
sys.path.append('/home/dhayaa/Desktop/DECADE/measure_shear/')
sys.path.append('/home/dhayaa/Desktop/DECADE/measure_shear/metacal')
from _step import _run_metacal as run_metacal
from legacy import _run_metacal_old as run_metacal_old
import fitsio

tile = 'DES1156-3706'
seed = 100

dir_meds = '/project2/chihway/data/decade/decade.ncsa.illinois.edu/deca_archive/DEC/multiepoch/shear/r5765/'+tile+'/p03/meds/'
filename = [dir_meds+tile+'_r5765p03_r_meds-shear.fits.fz',
            dir_meds+tile+'_r5765p03_i_meds-shear.fits.fz',
            dir_meds+tile+'_r5765p03_z_meds-shear.fits.fz']

with open('/home/dhayaa/Desktop/DECADE/mcal_sim_test/runs/run_template/metacal.yaml', 'r') as fp:
     mcal_config = yaml.load(fp, Loader=yaml.Loader)
        
output = run_metacal(filename, seed, mcal_config)
output_old = run_metacal_old(filename, seed, mcal_config)

fitsio.write('/scratch/midway2/dhayaa/metacal_output_'+tile+'.fits', output, clobber=True)
fitsio.write('/scratch/midway2/dhayaa/metacal_output_'+tile+'_old.fits', output_old, clobber=True)