import gc
import os
import sys
from copy import deepcopy
import json
import logging
from configparser import ConfigParser, ExtendedInterpolation
import fitsio
import numpy as np

from .shear import ShearTXConstant

from descwl_shear_sims.galaxies import WLDeblendGalaxyCatalog
from descwl_shear_sims.psfs import make_fixed_psf
from descwl_shear_sims.sim import make_sim
from descwl_shear_sims.surveys import DEFAULT_SURVEY_BANDS, get_survey


logger = logging.getLogger(__name__)
fmt = logging.Formatter("%(name)s: %(asctime)s | %(levelname)s | %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(handler)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

default_config = {
    "cosmic_rays": False,
    "bad_columns": False,
    "star_bleeds": False,
    "draw_method": "auto",
    "noise_factor": 0.0,
    "draw_gals": True,
    "draw_stars": False,
    "draw_bright": False,
    "star_catalog": None,
}
DEFAULT_BAND_LIST_JSON = '["r"]'
DEFAULT_ZERO_POINT = 30.0


class SimulationCore(object):
    def __init__(self, cparser):
        # Utility functions
        def get_config(key, fallback=None, parse_type=None):
            """Fetch a configuration value with optional parsing."""
            if parse_type == "int":
                return cparser.getint("simulation", key, fallback=fallback)
            elif parse_type == "float":
                return cparser.getfloat("simulation", key, fallback=fallback)
            elif parse_type == "boolean":
                return cparser.getboolean("simulation", key, fallback=fallback)
            return cparser.get("simulation", key, fallback=fallback)

        def construct_dir(sub_dir):
            """Construct a directory path relative to the root directory."""
            return os.path.join(self.root_dir, sub_dir) if sub_dir else None

        # Basic directories
        self.root_dir = get_config("root_dir")
        self.sim_name = get_config("sim_name")
        self.img_dir = construct_dir(self.sim_name)

        # Catalog directories
        cat_dir = get_config("cat_dir")
        self.cat_dir = construct_dir(cat_dir)
        cat_dm_dir = get_config(
            "cat_dm_dir", fallback=cat_dir.replace("cat", "cat_dm") if cat_dir else None
        )
        self.cat_dm_dir = construct_dir(cat_dm_dir)
        self.input_cat_dir = construct_dir(get_config("input_cat_dir"))

        # Summary directory
        self.sum_dir = construct_dir(get_config("sum_dir"))

        # Rotations
        self.nrot = get_config("nrot", parse_type="int", fallback=2)
        self.rot_list = [np.pi / self.nrot * i for i in range(self.nrot)]

        # Measurement and calibration
        self.bands = get_config("band")
        self.calib_mag_zero = DEFAULT_ZERO_POINT

        # Simulation layout and settings
        self.layout = get_config("layout")
        self.rotate = get_config("rotate", parse_type="boolean")
        self.dither = get_config("dither", parse_type="boolean")
        self.coadd_dim = get_config("coadd_dim", parse_type="int")
        self.buff = get_config("buff", parse_type="int")
        self.survey_name = get_config("survey_name", fallback="LSST")
        self.psf_fwhm = get_config("psf_fwhm", parse_type="float")
        self.psf_e1 = get_config("psf_e1", fallback=0.0, parse_type="float")
        self.psf_e2 = get_config("psf_e2", fallback=0.0, parse_type="float")
        self.psf_variation = get_config(
            "psf_variation", fallback=0.0, parse_type="float"
        )
        assert self.psf_variation >= 0.0

        # Shear settings
        self.shear_comp_sim = get_config("shear_component", fallback="gt")
        if self.shear_comp_sim not in ["gt", "gx"]:
            raise ValueError(f"Only supports 'shear_comp' of 'gt' or 'gx'")
        self.shear_mode_list = json.loads(get_config("shear_mode_list"))
        self.nshear = len(self.shear_mode_list)

        # Systematics
        self.cosmic_rays = get_config(
            "cosmic_rays", fallback=False, parse_type="boolean"
        )
        self.bad_columns = get_config(
            "bad_columns", fallback=False, parse_type="boolean"
        )

        # Stars
        self.stellar_density = get_config(
            "stellar_density", fallback=-1.0, parse_type="float"
        )
        self.stellar_density = (
            None if self.stellar_density < -0.01 else self.stellar_density
        )
        self.draw_bright = get_config(
            "draw_bright", fallback=False, parse_type="boolean"
        )
        self.star_bleeds = get_config(
            "star_bleeds", fallback=False, parse_type="boolean"
        )

        # Bands
        self.sim_band_list = json.loads(
            get_config("sim_band_list", fallback=DEFAULT_BAND_LIST_JSON)
        )
        self.nband = len(self.sim_band_list)
        return

    def get_psf_obj(self, rng, scale):
        psf_obj = make_fixed_psf(
            psf_type="moffat",
            psf_fwhm=self.psf_fwhm,
        ).shear(e1=self.psf_e1, e2=self.psf_e2)
        return psf_obj

    def get_sim_fnames(self, min_id, max_id, field_only=False):
        """Generate filename for simulations
        Args:
            min_id (int):   minimum id
            max_id (int):   maximum id
            field_only (bool): only include filed number
        Returns:
            out (list):     a list of file name
        """
        if field_only:
            out = [
                os.path.join(
                    self.img_dir,
                    "image-%05d_xxx.fits" % (fid),
                )
                for fid in range(min_id, max_id)
            ]
        else:
            out = [
                os.path.join(
                    self.img_dir,
                    "image-%05d_g1-%d_rot%d_xxx.fits" % (fid, gid, rid),
                )
                for fid in range(min_id, max_id)
                for gid in self.shear_mode_list
                for rid in range(self.nrot)
            ]
        return out

    def get_seed_from_fname(self, fname):
        """This function returns the random seed for image noise simulation.
        It makes sure that different sheared versions have the same seed.
        But different rotated version, different bands have different seeds.
        """
        # field id
        fid = int(fname.split("image-")[-1].split("_")[0]) + 212
        # rotation id
        rid = int(fname.split("rot")[1][0])
        return ((fid * self.nrot + rid)) * 3


class Simulation(SimulationCore):
    def __init__(self, config_name):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config_name)
        super().__init__(cparser)
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir, exist_ok=True)
        self.shear_value = cparser.getfloat("simulation", "shear_value")
        return

    def run(self, ifield):
        logger.info(f"Simulating for field: {ifield}")
        rng = np.random.RandomState(ifield)
        scale = get_survey(
            gal_type="wldeblend",
            band=deepcopy(DEFAULT_SURVEY_BANDS)[self.survey_name],
            survey_name=self.survey_name,
        ).pixel_scale
        psf_obj = self.get_psf_obj(rng, scale)

        kargs = deepcopy(default_config)

        # galaxy catalog; you can make your own
        galaxy_catalog = WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=self.coadd_dim,
            buff=self.buff,
            pixel_scale=scale,
            layout=self.layout,
        )
        logger.info(f"Simulation has galaxies: {len(galaxy_catalog)}")
        for shear_mode in self.shear_mode_list:
            shear_obj = ShearTXConstant(
                mode=shear_mode,
                g_dist=self.shear_comp_sim,
                shear_value=self.shear_value,
            )
            for irot in range(self.nrot):
                sim_data = make_sim(
                    rng=rng,
                    galaxy_catalog=galaxy_catalog,
                    coadd_dim=self.coadd_dim,
                    shear_obj=shear_obj,
                    psf=psf_obj,
                    dither=self.dither,
                    rotate=self.rotate,
                    bands=self.sim_band_list,
                    theta0=self.rot_list[irot],
                    calib_mag_zero=self.calib_mag_zero,
                    survey_name=self.survey_name,
                    **kargs,
                )
                # write galaxy images
                for band_name in self.sim_band_list:
                    gal_fname = "%s/image-%05d_%s-%d_rot%d_%s.fits" % (
                        self.img_dir,
                        ifield,
                        self.shear_comp_sim,
                        shear_mode,
                        irot,
                        band_name,
                    )
                    mi = sim_data["band_data"][band_name][0].getMaskedImage()
                    gdata = mi.getImage().getArray()
                    fitsio.write(gal_fname, gdata)
                    del mi, gdata, gal_fname
                del sim_data
                gc.collect()
        del galaxy_catalog
        return
