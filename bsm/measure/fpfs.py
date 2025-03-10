import os
import gc
import time
import logging
from configparser import ConfigParser, ExtendedInterpolation

import numpy as np
import anacal
import fitsio

from .utils import get_gridpsf_obj, get_psf_array
from ..simulation.base import SimulationCore
from ..simulation.loader import SimulationLoader

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)

class MeasureFPFS(SimulationCore):
    def __init__(self, config):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config)
        self.config = config
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Cannot find image directory {self.img_dir}")
        if not os.path.isdir(self.cat_dir):
            os.makedirs(self.cat_dir, exist_ok=True)
        
        # FPFS task
        self.sigma_arcsec = cparser.getfloat("FPFS", "sigma_arcsec", fallback=0.52)
        self.rcut = cparser.getint("FPFS", "rcut", fallback=32)
        self.ngrid = 2 * self.rcut
        psf_rcut = cparser.getint("FPFS", "psf_rcut", fallback=22)
        self.psf_rcut = min(psf_rcut, self.rcut)
        self.klim_thres = cparser.getfloat("FPFS", "klim_thres", fallback=1e-12)
        self.fpfs_config = anacal.fpfs.FpfsConfig(
            sigma_arcsec=self.sigma_arcsec
        )
        return

    def process_image(
        self,
        gal_array,
        psf_array,
        mag_zero,
        pixel_scale,
        noise_std,
        noise_array,
        psf_obj,
        mask_array,
        star_cat,
    ):
        out = anacal.fpfs.process_image(
            fpfs_config=self.fpfs_config,
            mag_zero=mag_zero,
            gal_array=gal_array,
            psf_array=psf_array,
            mask_array=mask_array,
            pixel_scale=pixel_scale,
            noise_variance=noise_std**2,
            noise_array=noise_array,
            detection=None,
            star_catalog=star_cat,
        )
        return out


    def prepare_data(self, fname):
        loader = SimulationLoader(self.config)
        seed = loader.get_seed_from_fname(fname, "r") + 1
        exposure = loader.generate_exposure(fname)
        pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
        variance = np.average(exposure.getMaskedImage().variance.array)
        mag_zero = (
            np.log10(exposure.getPhotoCalib().getInstFluxAtZeroMagnitude())
            / 0.4
        )
        psf_obj = get_gridpsf_obj(
            exposure,
            ngrid=self.ngrid,
            psf_rcut=self.psf_rcut,
            dg=250
        )
        psf_array = get_psf_array(
            exposure,
            ngrid=self.ngrid,
            psf_rcut=self.psf_rcut,
            dg=250,
        ).astype(np.float64)
        gal_array = np.asarray(
            exposure.getMaskedImage().image.array,
            dtype=np.float64
        )
        mask_array = np.asarray(
            exposure.getMaskedImage().mask.array,
            dtype=np.int16
        )
        del exposure, loader
        ny = self.coadd_dim + 10
        nx = self.coadd_dim + 10
        noise_std = np.sqrt(variance)
        if variance > 1e-8:
            noise_array = (
                np.random.RandomState(seed)
                .normal(
                    scale=noise_std,
                    size=(ny, nx),
                ).astype(np.float64)
            )
        else:
            noise_array = None
        gc.collect()
        star_cat = None
        return {
            "gal_array": gal_array,
            "psf_array": psf_array,
            "mag_zero": mag_zero,
            "pixel_scale": pixel_scale,
            "noise_std": noise_std,
            "noise_array": noise_array,
            "psf_obj": psf_obj,
            "mask_array": mask_array,
            "star_cat": star_cat
        }
    
    def run(self, fname):
        data = self.prepare_data(fname)
        start_time = time.time()
        out = self.process_image(**data)
        sname = fname.replace(
            "image", "src").replace(
                self.sim_name, self.cat_dir
            )
        fitsio.write(
            sname,
            out,
            clobber=True,
        )
        del data
        elapsed_time = time.time() - start_time
        logging.info(f"Elapsed time: {elapsed_time:.2f}")
        return

        

