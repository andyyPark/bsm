import os
from copy import deepcopy
import logging
from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import fitsio
import galsim

from .base import SimulationCore

import lsst.afw.image as afw_image
from descwl_shear_sims.psfs import make_dm_psf
from descwl_shear_sims.sim import get_se_dim, get_coadd_center_gs_pos
from descwl_shear_sims.surveys import DEFAULT_SURVEY_BANDS, get_survey
from descwl_shear_sims.wcs import make_dm_wcs, make_se_wcs, make_coadd_dm_wcs

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)

class SimulationLoader(SimulationCore):
    def __init__(self, config):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config)
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("Cannot find image directory")
        
        self.nstd_map = {
            "g": 0.315,
            "r": 0.371,
            "i": 0.595,
            "z": 1.155,
        }
        
    def generate_exposure(self, fname):
        field_id = int(fname.split("image-")[-1].split("_")[0])
        rng = np.random.RandomState(field_id)
        pixel_scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.survey_name],
            survey_name=self.survey_name,
        ).pixel_scale
        psf_obj = self.get_psf_obj(rng, pixel_scale)
        variance = 0.0
        weight_sum = 0.0
        ny = self.coadd_dim + 10
        nx = self.coadd_dim + 10
        gal_array = np.zeros((ny, nx))
        msk_array = np.zeros((ny, nx), dtype=int)
        for i, band in enumerate(self.sim_band_list):
            logging.info(f"reading {band} band")
            # Add noise
            nstd_f = self.nstd_map[band]
            weight = 1.0 / (self.nstd_map[band]) ** 2.0
            variance += (nstd_f * weight) ** 2.0
            logging.info(f"Using noisy setup with std: {nstd_f:.2f}")
            if nstd_f > 1e-4:
                seed = self.get_seed_from_fname(fname, band)
                noise_array = np.random.RandomState(seed).normal(
                    scale=nstd_f,
                    size=(ny, nx),)
            
                logging.info(f"Simulated noise STD is: {np.std(noise_array):.2f}")
            else:
                noise_array = 0.0
            gal_array = (
                gal_array
                + (
                    fitsio.read(fname.replace("_xxx", f"_{band}"))
                    + noise_array
                )
                * weight
            )
            weight_sum += weight
            del noise_array
        masked_image = afw_image.MaskedImageF(ny, nx)
        masked_image.image.array[:, :] = gal_array / weight_sum
        masked_image.variance.array[:, :] = variance / (weight_sum) ** 2.0
        std_final = np.sqrt(variance / (weight_sum) ** 2.0)
        logging.info(f"The final noise std is {std_final:.2f}")
        masked_image.mask.array[:, :] = msk_array
        exp = afw_image.ExposureF(masked_image)
        zero_flux = 10.0 ** (0.4 * self.calib_mag_zero)
        photo_calib = afw_image.makePhotoCalibFromCalibZeroPoint(zero_flux)
        exp.setPhotoCalib(photo_calib)
        psf_dim = 51
        coadd_wcs, coadd_bbox = make_coadd_dm_wcs(
            coadd_dim=self.coadd_dim,
            pixel_scale=pixel_scale,
        )
        coadd_bbox_cen_gs_skypos = get_coadd_center_gs_pos(
            coadd_wcs=coadd_wcs,
            coadd_bbox=coadd_bbox,
        )
        coadd_scale = coadd_wcs.getPixelScale().asArcseconds()
        se_dim = get_se_dim(
            coadd_scale=coadd_scale,
            coadd_dim=self.coadd_dim,
            se_scale=pixel_scale,
            rotate=self.rotate,
        )
        dims = [int(se_dim)] * 2
        cen = (np.array(dims) + 1) / 2
        se_origin = galsim.PositionD(x=cen[1], y=cen[0])
        se_wcs = make_se_wcs(
            pixel_scale=pixel_scale,
            image_origin=se_origin,
            world_origin=coadd_bbox_cen_gs_skypos,
            dither=self.dither,
            dither_size=0.5,
            rotate=self.rotate,
            rng=rng,
        )
        dm_psf = make_dm_psf(
            psf=psf_obj,
            psf_dim=psf_dim,
            wcs=deepcopy(se_wcs),
        )
        exp.setPsf(dm_psf)
        dm_wcs = make_dm_wcs(se_wcs)
        exp.setWcs(dm_wcs)
        return exp

    def run(self, fname):
        return self.generate_exposure(fname)