import numpy as np

import esutil as eu
from ngmix import ObsList, MultiBandObsList
from ngmix.gexceptions import GMixRangeError
from ngmix.medsreaders import NGMixMEDS, MultiBandNGMixMEDS
from meds_preprocess.interpolate import interpolate_image_at_mask
NGMIX_V1=False

import galsim

def _strip_coadd(mbobs, mcal_config):
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        for i in range(1, len(ol)):
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs


def _strip_zero_flux(mbobs, mcal_config):
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        for i in range(len(ol)):
            if np.sum(ol[i].image) > 0:
                _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs


def _apply_pixel_scale(mbobs, mcal_config):
    for ol in mbobs:
        for o in ol:
            scale    = o.jacobian.get_scale()
            scale2   = scale * scale
            scale4   = scale2 * scale2
            o.image  = o.image / scale2
            o.weight = o.weight * scale4
            
            if mcal_config['custom']['interp_bad_pixels']:
                o.noise  = o.noise / scale2
            
    return mbobs

def _strip_10percent_masked(mbobs, mcal_config):
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
            
            msk = ol[i].bmask.astype(bool) #Mask where TRUE means bad pixel
            
            if np.average(msk) >= mcal_config['custom']['maxbadfrac']:
                continue
            
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs

def _strip_Nexposures(mbobs, rng, mcal_config):
    
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        Nexposures_current = len(ol) #How many exposures are available
        Nexposures_max     = mcal_config['custom']['Nexp_max'] #Maximum exp count we use for mcal
        
        
        if Nexposures_current <= Nexposures_max:
            list_of_exposures = np.arange(Nexposures_current)
        else:
            list_of_exposures = rng.choice(Nexposures_current, Nexposures_max, replace = False) #Indices of Subsampled list

        for i in list_of_exposures:
            _ol.append(ol[i])
            
        _mbobs.append(_ol)
    return _mbobs
    
    
def _get_masked_frac(mbobs, mcal_config):
    
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    
    gauss = galsim.Gaussian(fwhm = 1.2) #Fixed aperture gauss weights for image
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
            
            msk = ol[i].bmask.astype(bool) #Mask where TRUE means bad pixel
            wgt = np.median(ol[i].weight[ol[i].weight != 0]) #Median weight used to populate noise in empty pix
            
            #get wcs of this observations
            wcs = ol[i].jacobian.get_galsim_wcs()

            #Create gaussian weights image (as array)
            gauss_wgt = gauss.drawImage(nx = msk.shape[0], ny = msk.shape[1], wcs = wcs, method = 'real_space').array 

            #msk is nonzero for bad pixs. Invert it, and convert to int
            good_frac = np.average(np.invert(msk).astype(int), weights = gauss_wgt) #Fraction of missing values

            #Save fraction of good pix. Will use later to remove
            #problematic objects directly from metacal catalog
            ol[i].meta['good_frac'] = good_frac
            ol[i].meta['weight']    = wgt
            
#             print("goodfrac", good_frac, wgt)

            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs

def _symmetrize_mask(mbobs, mcal_config):
    
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
            
            msk = ol[i].bmask.astype(bool) #Mask where TRUE means bad pixel
                
#             print("symm1", msk.sum(), np.sum(ol[i].bmask))
            #Rotate because Metacal needs this
            msk |= np.rot90(msk, k = 1)
            
            #Write rotated mask back to observation
            ol[i].bmask = msk.astype(np.int32)
            
#             print("symm2", msk.sum(), np.sum(ol[i].bmask))
            
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs
    
def _fill_empty_pix(mbobs, rng, mcal_config):
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
            
            msk = ol[i].bmask.astype(bool) #Mask where TRUE means bad pixel
            wgt = np.median(ol[i].weight[ol[i].weight != 0]) #Median weight used to populate noise in empty pix
            
            #Observation doesn't have noise image, and so add noise image in.
            #Just random gaussian noise image using weights
            #Need to do this for interpolation step    
            ol[i].noise = rng.normal(loc = 0, scale = 1/np.sqrt(wgt), size = ol[i].image.shape)

            
            #If there are any bad mask pixels, then do interpolation
            if np.any(msk):
                
                #Interpolate image to fill in gaps. Setting maxfrac=1 since maxfrac is checked beforehand
                im    = interpolate_image_at_mask(image=ol[i].image, weight=wgt, bad_msk=msk, 
                                                  rng=rng, maxfrac=1, buff=4,
                                                  fill_isolated_with_noise=True)

                #Interpolate over noise image
                noise = interpolate_image_at_mask(image=ol[i].noise, weight=wgt, bad_msk=msk, 
                                                  rng=rng, maxfrac=1, buff=4,
                                                  fill_isolated_with_noise=True)

                #If we can't interpolate image or noise due to lack of data
                #then we skip this observation (it is stripped from MultiBandObs list)
                if (im is None) | (noise is None):
                    continue
                    
                #Set all masked pixel weights to 0.0
                ol[i].image  = im
                ol[i].weight = np.where(msk, 0, ol[i].weight)
                ol[i].noise  = noise

            
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs
