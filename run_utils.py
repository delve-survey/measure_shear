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

def _run_mcal_one_chunk(meds_files, start, end, seed, mcal_config):
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
            o = mbmeds.get_mbobs(ind)
            
            o = preprocess._strip_coadd(o, mcal_config) #Remove coadd since it isnt used in fitting
            o = preprocess._strip_zero_flux(o, mcal_config) #Remove any obs with zero flux
            o = preprocess._add_zeroweights_mask(o, mcal_config) #Add mask from weights==0 condition
            
            #Keep only Nmax exps per band
            if mcal_config['custom']['Nexp_max'] > 0: 
                o = preprocess._strip_Nexposures(o, rng, mcal_config) 
                
                
            #Remove obs with too many bad pixs
            if mcal_config['custom']['maxbadfrac'] > 0: 
                o = preprocess._strip_10percent_masked(o, mcal_config)
                
                
            #gauss-weighted fraction of good pixs
            if mcal_config['custom']['goodfrac']: 
                o = preprocess._get_masked_frac(o, mcal_config) 
                
                
            #Add 180deg Symmetry of bmask
            if mcal_config['custom']['symmetrize_mask']: 
                o = preprocess._symmetrize_mask(o, mcal_config) 
            
            
            #Interpolate empty pixels
            if mcal_config['custom']['interp_bad_pixels']: 
                o = preprocess._fill_empty_pix(o, rng, mcal_config)
            
            
            o = preprocess._set_zero_weights(o,  mcal_config) #Set all masked pix to have wgt=0, include mask symmetry

            skip_me = False
            for ol in o:
                if len(ol) == 0:
                    logger.debug(' not all bands have images - skipping!')
                    skip_me = True
            if skip_me:
                continue

            o.meta['id'] = ind
            o[0].meta['Tsky'] = 1
            o[0].meta['magzp_ref']   = MAGZP_REF
            o[0][0].meta['orig_col'] = cat['orig_col'][ind, 0]
            o[0][0].meta['orig_row'] = cat['orig_row'][ind, 0]

            if mcal_config['custom']['goodfrac']:
                #put all the good_fraction numbers into one list
                #one entry per cutout (so all bands are combined here)
                good_frac = []
                weight    = []
                for _one in o:
                    for _two in _one:
                        good_frac.append(_two.meta['good_frac'])
                        weight.append(_two.meta['weight'])
                    
            nband = len(o)
            mcal = MetacalFitter(mcal_config, nband, rng)

            try:
                mcal.go([o])
                res = mcal.result
            except GMixRangeError as e:
                logger.debug(" metacal error: %s", str(e))
                res = None

            if res is not None:
                if mcal_config['custom']['goodfrac']:
                    res['good_frac'] = np.average(good_frac, weights = weight) #store mean good_fraction per object
                else:
                    res['good_frac'] = 1 #Else, we're assuming image is "perfect" (completely unmasked) == 1
                data.append(res)

        if len(data) > 0:
            output = eu.numpy_util.combine_arrlist(data)
    finally:
        for m in mfiles:
            m.close()

    return output
