from configparser import ConfigParser, ExtendedInterpolation
import numpy as np
import fitsio

from ..simulation.base import SimulationCore
from bsm.meas_rotate import *

class MCBias(SimulationCore):
    def __init__(self, config):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config)
        self.config = config
        super().__init__(cparser)

    def run(self, i):
        fnames = self.get_sim_fnames(
            min_id=i, max_id=i+1
        )
        image_dim = fitsio.read(fnames[0]).shape[0]
        snames = [
            fname.replace(
                "image", "src"
            ).replace(self.sim_name, self.cat_dir)
            for fname in fnames
        ]
        
        s00, s01, s10, s11 = snames
        s00 = fitsio.read(s00)
        s01 = fitsio.read(s01)
        s10 = fitsio.read(s10)
        s11 = fitsio.read(s11)

        xp = np.concatenate([s00["x"], s01["x"]])
        yp = np.concatenate([s00["y"], s01["y"]])
        anglep = get_angle_from_pixel(xp, yp, image_dim/2, image_dim/2)
        e1p = np.concatenate([s00["fpfs_e1"], s01["fpfs_e1"]])
        e2p = np.concatenate([s00["fpfs_e2"], s01["fpfs_e2"]])
        e1_g1p = np.concatenate([s00["fpfs_de1_dg1"], s01["fpfs_de1_dg1"]])
        e2_g2p = np.concatenate([s00["fpfs_de2_dg2"], s01["fpfs_de2_dg2"]])
        wp = np.concatenate([s00["fpfs_w"], s01["fpfs_w"]])
        w_g1p = np.concatenate([s00["fpfs_dw_dg1"], s01["fpfs_dw_dg1"]])
        w_g2p = np.concatenate([s00["fpfs_dw_dg2"], s01["fpfs_dw_dg2"]])

        eTp, eXp = rotate_spin_2_ellip(e1p, e2p, anglep)
        R1p, R2p = get_response_from_w_and_der(
                e1p, e2p, wp, e1_g1p, e2_g2p, w_g1p, w_g2p
                )
        RTp, RXp = rotate_spin_2_response(R1p, R2p, anglep)

        eTp = np.sum(eTp * wp)
        eXp = np.sum(eXp * wp)
        RTp = np.sum(RTp)
        RXp = np.sum(RXp)

        xm = np.concatenate([s10["x"], s11["x"]])
        ym = np.concatenate([s10["y"], s11["y"]])
        anglem = get_angle_from_pixel(xm, ym, image_dim/2, image_dim/2)
        e1m = np.concatenate([s10["fpfs_e1"], s11["fpfs_e1"]])
        e2m = np.concatenate([s10["fpfs_e2"], s11["fpfs_e2"]])
        e1_g1m = np.concatenate([s10["fpfs_de1_dg1"], s11["fpfs_de1_dg1"]])
        e2_g2m = np.concatenate([s10["fpfs_de2_dg2"], s11["fpfs_de2_dg2"]])
        wm = np.concatenate([s10["fpfs_w"], s11["fpfs_w"]])
        w_g1m = np.concatenate([s10["fpfs_dw_dg1"], s11["fpfs_dw_dg1"]])
        w_g2m = np.concatenate([s10["fpfs_dw_dg2"], s11["fpfs_dw_dg2"]])

        eTm, eXm = rotate_spin_2_ellip(e1m, e2m, anglem)
        R1m, R2m = get_response_from_w_and_der(
                e1m, e2m, wm, e1_g1m, e2_g2m, w_g1m, w_g2m
                )
        RTm, RXm = rotate_spin_2_response(R1m, R2m, anglem)

        eTm = np.sum(eTm * wm)
        eXm = np.sum(eXm * wm)
        RTm = np.sum(RTm)
        RXm = np.sum(RXm)

        return [eTp, eTm, RTp, RTm, eXp, eXm, RXp, RXm]


