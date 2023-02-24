import logging
import numpy as np

import joblib
import esutil as eu
import fitsio
from ngmix import ObsList, MultiBandObsList
from ngmix.gexceptions import GMixRangeError
from ngmix.medsreaders import NGMixMEDS, MultiBandNGMixMEDS
NGMIX_V1=False

from metacal.metacal_fitter import MetacalFitter
import meds_preprocess.preprocess as preprocess
import galsim

# always and forever
MAGZP_REF = 30.0

logger = logging.getLogger(__name__)

def _run_mcal_one_chunk_old(meds_files, start, end, seed, mcal_config):
    """Run metcal for `meds_files` only for objects from `start` to `end`.

    Note that `start` and `end` follow normal python indexing conventions so
    that the list of indices processed is `list(range(start, end))`.

    Parameters
    ----------
    meds_files : list of str
        A list of paths to the MEDS files.
    start : int
        The starting index of objects in the file on which to run metacal.
    end : int
        One plus the last index to process.
    seed : int
        The seed for the RNG.
    mcal_config : yaml
        The config file for the metacal run

    Returns
    -------
    output : np.ndarray
        The metacal outputs.
    """
    rng = np.random.RandomState(seed=seed)

    # seed the global RNG to try to make things reproducible
    np.random.seed(seed=rng.randint(low=1, high=2**30))

    output = None
    mfiles = []
    data = []
    try:
        # get the MEDS interface
        for m in meds_files:
            mfiles.append(NGMixMEDS(m))
        mbmeds = MultiBandNGMixMEDS(mfiles)
        cat = mfiles[0].get_cat()

        for ind in range(start, end):
            
#             print("IN GAL", ind)
            
            #Load mbobs for fiducial run
            o = mbmeds.get_mbobs(ind, weight_type='weight')
            
            #Use wcs of coadd to make gaussian pixel weights for bad mask fraction
            if len(o[0]) == 0: 
                continue
            else:
                coadd_wcs_rband = o[0][0].jacobian.get_galsim_wcs()
             
            o = preprocess._strip_coadd(o, mcal_config) #Remove coadd since it isnt used in fitting
            o = preprocess._strip_zero_flux(o, mcal_config) #Remove any obs with zero flux
            
            #Check if missing any bands. Happens due to observation
            #Do this once at start just to prevent computation later on
            if preprocess._check_band_coverage(o): continue
                
            Ncutouts_per_band = [len(i) for i in o]
            
            #Keep only Nmax exps per band
            if mcal_config['custom']['Nexp_max'] > 0: 
                o = preprocess._strip_Nexposures(o, np.random.RandomState(seed=seed), mcal_config)
             
            #Zero-out weights for bmask != 0
            #from now on weights contains the bmask in it
            o = preprocess._set_zero_weights(o, mcal_config)
            
            #Add 180deg Symmetry of weights (NOT UBERSEG)
            mcal_config['custom']['symmetrize_weights'] = mcal_config['custom']['symmetrize_mask'] #Monkeypatch for now. FIX LATER
            if mcal_config['custom']['symmetrize_weights']: 
                o = preprocess._symmetrize_weights(o, mcal_config)             
            
            #Check again if missing any bands. 
            #Happens due to weights symmetry, uberseg, or adding bmask
            if preprocess._check_band_coverage(o): continue
            
            #gauss-weighted fraction of bad pixels
            o = preprocess._get_masked_frac(o, mcal_config, coadd_wcs_rband)
            
            badfrac = o.meta['badfrac']
            
            
            #Fill empty pixels of the cutout using interpolation + a noise image
            if mcal_config['custom']['interp_bad_pixels']: 
                o = preprocess._fill_empty_pix(o, rng, mcal_config) 
                
            
            ##########################################################################
            
            o.meta['id'] = ind
            o[0].meta['Tsky'] = 1
            o[0].meta['magzp_ref']   = MAGZP_REF
#             o[0][0].meta['orig_col'] = cat['orig_col'][ind, 0]
#             o[0][0].meta['orig_row'] = cat['orig_row'][ind, 0]
                    
            nband = len(o)
            mcal = MetacalFitter(mcal_config, nband, rng)

            try:
                mcal.go([o])
                
                #Read metacal output, and get the dtype of the field.
                #This step is done so we can add custom quantities, not computed in mcal
                #to the final output of the metacal files
                tmp = mcal.result
                
                if tmp is not None:
                    dt = tmp.dtype.fields
                    dt = [(i, dt[i][0]) for i in dt]
                    
                    #Add custom quantities for badfrac
                    dt += [('badfrac', '>f8')]
                    
                    dt += [('Ncutouts_raw', '>i8', (len(o),))]
                    
                    #Add custom quantities for focal plane coords.
                    #Has shape (N_bands, N_cutouts)
                    dt += [('expnum', '>i8', (len(o), mcal_config['custom']['Nexp_max']))]
                    dt += [('ccdnum', '>i8', (len(o), mcal_config['custom']['Nexp_max']))]
                    dt += [('x_exp',  '>f8', (len(o), mcal_config['custom']['Nexp_max']))]
                    dt += [('y_exp',  '>f8', (len(o), mcal_config['custom']['Nexp_max']))]

                    res = np.zeros(len(tmp), dtype = dt)

                    for i in tmp.dtype.names:
                        res[i] = tmp[i]
                        
                    #Store number of cutouts in each band BEFORE we subsample
                    for i in range(len(o)):
                        res['Ncutouts_raw'][0, i] = Ncutouts_per_band[i]
                    
                    #Store mean bad fraction of this image to use in making cuts
                    res['badfrac'] = badfrac
                        
                    #Hacking to get coadd position in results
                    #(Mcal would just write first cutout position here by default,
                    #but we stripped coadd so it isn't first cutout)
                    res['x'] = cat['orig_col'][ind, 0]
                    res['y'] = cat['orig_row'][ind, 0]

                    #Set default values for these. Entries take these values
                    #for an object that doesn't have all the cutouts we need
                    res['expnum'] = -9999
                    res['ccdnum'] = -9999
                    res['x_exp']  = -9999
                    res['y_exp']  = -9999

                    for i_band in range(len(o)):
                        for i_cutout in range(len(o[i_band])):
                            file_path = o[i_band][i_cutout].meta['file_path']
                            res['expnum'][0, i_band, i_cutout] = file_path.split('_')[0][1:] #Store expnum as "D00605764" -> 605764
                            res['ccdnum'][0, i_band, i_cutout] = file_path.split('_')[2][1:] #Store ccdnum as "c42" -> 42
                            res['x_exp'][0, i_band, i_cutout]  = o[i_band][i_cutout].meta['orig_col']
                            res['y_exp'][0, i_band, i_cutout]  = o[i_band][i_cutout].meta['orig_row']                  
                
                elif tmp is None:
                    res = tmp
                               
            except GMixRangeError as e:
                logger.debug(" metacal error: %s", str(e))
                res = None

            if res is not None:
                data.append(res)

        if len(data) > 0:
            output = eu.numpy_util.combine_arrlist(data)
    finally:
        for m in mfiles:
            m.close()

    return output


import os

import numpy as np
import joblib
import esutil as eu
import fitsio
from ngmix import ObsList, MultiBandObsList
from ngmix.gexceptions import GMixRangeError
import logging

from ngmix.medsreaders import MultiBandNGMixMEDS, NGMixMEDS
from metacal import MetacalFitter
from ngmix_compat_metacal import NGMIX_V2
import yaml
import argparse

logger = logging.getLogger(__name__)

my_parser = argparse.ArgumentParser()

def _run_metacal_old(meds_files, seed, mcal_config):
    """Run metacal on a tile.

    Parameters
    ----------
    meds_files : list of str
        A list of the meds files to run metacal on.
    seed : int
        The seed for the global RNG.
    metacal_config:
        The metacal config file for the runs
    """
    with NGMixMEDS(meds_files[0]) as m:
        cat = m.get_cat()
    logger.info(' meds files %s', meds_files)

    n_cpus = joblib.externals.loky.cpu_count()
    n_chunks = max(n_cpus, 60)
    n_obj_per_chunk = int(cat.size / n_chunks)
    if n_obj_per_chunk * n_chunks < cat.size:
        n_obj_per_chunk += 1
    assert n_obj_per_chunk * n_chunks >= cat.size
    logger.info(
        ' running metacal for %d objects in %d chunks', cat.size, n_chunks)

    seeds = np.random.RandomState(seed=seed).randint(1, 2**30, size=n_chunks)

    jobs = []
    for chunk in range(n_chunks):
        start = chunk * n_obj_per_chunk
        end = min(start + n_obj_per_chunk, cat.size)
        jobs.append(joblib.delayed(_run_mcal_one_chunk_old)(
            meds_files, start, end, seeds[chunk], mcal_config))

    with joblib.Parallel(
        n_jobs=n_cpus, backend='multiprocessing',
        verbose=100, max_nbytes=None
    ) as p:
        outputs = p(jobs)

    assert not all([o is None for o in outputs]), (
        "All metacal fits failed!")

    output = eu.numpy_util.combine_arrlist(
        [o for o in outputs if o is not None])
    logger.info(' %d of %d metacal fits worked!', output.size, cat.size)

    return output