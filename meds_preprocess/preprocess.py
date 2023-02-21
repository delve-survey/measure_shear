import numpy as np

import esutil as eu
from ngmix import ObsList, MultiBandObsList
from ngmix.gexceptions import GMixRangeError, GMixFatalError
from ngmix.medsreaders import NGMixMEDS, MultiBandNGMixMEDS
from meds_preprocess.interpolate import interpolate_image_at_mask
NGMIX_V1=False

import galsim

BMASK_WGT = 2**20 #value used to set weight==0 mask bit

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
    
def _add_zeroweights_mask(mbobs, mcal_config):
    
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
            
            weight_mask = np.where(ol[i].weight <= 0, BMASK_WGT, 0)
            ol[i].bmask = np.bitwise_or(ol[i].bmask, weight_mask)
            
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs

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

def _get_masked_frac(mbobs, mcal_config, coadd_wcs_rband):
    
    gauss = galsim.Gaussian(fwhm = 2) #Fixed aperture gauss weights for image. 2arcsec FWMH suggested by Matt
   
    #Quantities to make mask coadd
    mask_coadd_sum  = 0
    mask_weight_sum = 0
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
            
            msk = ol[i].weight == 0 #ol[i].bmask.astype(bool) #Mask where TRUE means bad pixel
            
            if np.sum(np.invert(msk)) == 0: continue #If no good pixel, then skip this loop
            
            wgt = np.median(ol[i].weight[np.invert(msk)]) #Median weight used to do coadd of mask
            
            mask_coadd_sum  += msk*wgt
            mask_weight_sum += wgt
    
    #If all image is masked completely
    if (mask_weight_sum == 0) & np.all(mask_coadd_sum == 0):

        mbobs.meta['badfrac'] = 1 #All images are masked
    
    else:
        mask_coadd = mask_coadd_sum/mask_weight_sum

        #Create gaussian weights image (as array)
        #We use the r-band coadd wcs to make the gaussian weight image. DHAYAA: FEELS LIKE THIS COULD BE WRONG THING TO DO :P
        gauss_wgt = gauss.drawImage(nx = mask_coadd.shape[0], ny = mask_coadd.shape[1], wcs = coadd_wcs_rband, method = 'real_space').array 

        #msk is nonzero for bad pixs.
        badfrac   = np.average(mask_coadd, weights = gauss_wgt) #Fraction of missing values

        #Save fraction of bad pix. Will use later to remove
        #problematic objects directly from metacal catalog
        mbobs.meta['badfrac'] = badfrac
    
    return mbobs

def _symmetrize_mask(mbobs, mcal_config):
    
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
            
            #Rotate, merge mask for 180deg symmetry and write back to observation
            ol[i].bmask = np.bitwise_or(ol[i].bmask, np.rot90(ol[i].bmask))
            
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs

def _symmetrize_weights(mbobs, mcal_config):
    
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
            
            #Rotate, merge mask for 180deg symmetry and write back to observation
            ol[i].weight = np.where((ol[i].weight <= 0) | (np.rot90(ol[i].weight) <= 0), 0, ol[i].weight)

            ol[i].bmask = ol[i].weight == 0
            
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs

def _set_zero_weights(mbobs, mcal_config):
    
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
                        
            #Set weights=0 whenever bmask is set
            wgt = ol[i].weight.copy()
            wgt[ol[i].bmask != 0] = 0.
            ol[i].weight = wgt
            
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs

def _add_uberseg(mbobs, mbobs_with_uberseg, mcal_config):
    
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    
    #First build a dictionary that tells you image name
    #to the index in the multiband obs list for the corresponding cutout
    
    locator = {}
    for i, ol in enumerate(mbobs_with_uberseg):
        for j, o in enumerate(ol):
#             print(j, ol, ol.meta)
            locator[o.meta['file_path']] = (i, j)
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
            
            #If uberseg version doesn't exist then skip this observation
            #this means uberseg weight mask is so large that the image has no usable pixels
            if ol[i].meta['file_path'] not in locator: continue
                
            inds = locator[ol[i].meta['file_path']]
            
            #add uberseg weights image to observation
            ol[i].uberseg = mbobs_with_uberseg[inds[0]][inds[1]].weight

            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs

def _apply_uberseg(mbobs, mcal_config):
    
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    
    #Loop over different band observations (r, i, z)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        
        #Loop over different exposures/cutouts in each band
        for i in range(len(ol)):
            
            #apply uberseg weights to fiducial weights
            #Ensures that all ubserseg pixels are masked out here
            new_weights  = np.where(ol[i].uberseg == 0, 0, ol[i].weight)
            
            if np.sum(new_weights != 0) == 0: continue #Check if entire image is masked. Skip if yes.
            
            ol[i].weight = new_weights

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
            
            msk = ol[i].weight == 0 #ol[i].bmask.astype(bool) #Mask where TRUE means bad pixel
            
            if np.sum(np.invert(msk)) == 0: continue
            wgt = np.median(ol[i].weight[np.invert(msk)]) #Median weight (of only good pix) used to populate noise in empty pix
            
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
                ol[i].noise  = noise

            
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs
