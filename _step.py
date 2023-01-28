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
from run_utils import _run_mcal_one_chunk

logger = logging.getLogger(__name__)

my_parser = argparse.ArgumentParser()
        
def _run_metacal(meds_files, seed, mcal_config):
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
        jobs.append(joblib.delayed(_run_mcal_one_chunk)(
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

'''

#Dhayaa: Commenting out parts of code that we don't really use
         Keeping here for posterity (maybe just remove though)

from eastlake.step import Step
from eastlake.utils import safe_mkdir


class NewishMetcalRunner(Step):
    """Run a newer metacal.

    Config Params
    -------------
    bands : list of str, optional
        A list of bands to use. Defaults to `["r", "i", "z"]`.
    """
    def __init__(self, config, base_dir, name="newish-metacal", logger=None,
                 verbosity=0, log_file=None):
        super(NewishMetcalRunner, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        # bands to use
        self.bands = self.config.get("bands", ["r", "i", "z"])

    def clear_stash(self, stash):
        pass

    def execute(self, stash, new_params=None):
        self.clear_stash(stash)

        base_output_dir = os.path.join(self.base_dir, "newish-metacal")
        safe_mkdir(base_output_dir)

        for tilename in stash["tilenames"]:

            # meds files
            built_meds_files = stash.get_filepaths("meds_files", tilename)

            meds_files_to_use = []
            for band in self.bands:
                sstr = "%s_%s" % (tilename, band)
                if not any(sstr in f for f in built_meds_files):
                    raise RuntimeError(
                        "could not find meds file for tilename %s band %s" % (
                            tilename, band))
                else:
                    for f in built_meds_files:
                        if sstr in f:
                            meds_files_to_use.append(f)
                            break
            assert len(meds_files_to_use) == len(self.bands), (
                "We did not find the right number of meds files!")

            # record what bands we're running in the stash
            stash["newish_metacal_bands"] = self.bands

            try:
                staged_meds_files = []
                tmpdir = os.environ['TMPDIR']
                for fname in meds_files_to_use:
                    os.system("cp %s %s/." % (fname, tmpdir))
                    staged_meds_files.append(
                        os.path.join(tmpdir, os.path.basename(fname))
                    )
                # FIXME hard coded seed
                output = _run_metacal(meds_files = staged_meds_files, seed = args['mcal_seed'])
            finally:
                for fname in staged_meds_files:
                    os.system("rm -f %s" % fname)

            output_dir = os.path.join(base_output_dir, tilename)
            safe_mkdir(output_dir)

            fname = os.path.join(output_dir, "%s_newish_metacal.fits" % tilename)
            fitsio.write(fname, output, clobber=True)

            stash.set_filepaths("newish_metacal_output", fname, tilename)

        return 0, stash
'''
