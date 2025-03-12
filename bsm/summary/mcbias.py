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

class MCBiasBinned(SimulationCore):
    def __init__(self, config):
        cparser = ConfigParser(interpolation=ExtendedInterpolation())
        cparser.read(config)
        self.config = config
        super().__init__(cparser)
        self.scale = 0.02

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

        max_pixel = (image_dim - 110) / 2

        n_bins = 15
        pixel_bin_edges = np.linspace(0, max_pixel, n_bins + 1)
        pixel_bin_edges = np.logspace(np.log10(200), np.log10(max_pixel), n_bins + 1)
        angular_bin_edges = pixel_bin_edges * self.scale
        angular_bin_mid = (angular_bin_edges[1:] + angular_bin_edges[:-1]) / 2
        eTp_list = []
        eXp_list = []
        RTp_list = []
        RXp_list = []
        eTm_list = []
        eXm_list = []
        RTm_list = []
        RXm_list = []
        ngal_p = []
        ngal_m = []

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
        radial_dist_p = np.sqrt(
            (xp - image_dim/2) ** 2 + (yp - image_dim/2) ** 2
        )

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
        radial_dist_m = np.sqrt(
            (xm - image_dim/2) ** 2 + (ym - image_dim/2) ** 2
        )

        for bin_i in range(n_bins):
            mask = (radial_dist_p >= pixel_bin_edges[bin_i]) & (radial_dist_p < pixel_bin_edges[bin_i + 1])
            ngal_p.append(np.sum(mask))
            eTp, eXp = rotate_spin_2_ellip(e1p[mask], e2p[mask], anglep[mask])
            R1p, R2p = get_response_from_w_and_der(
                    e1p[mask], e2p[mask], wp[mask], e1_g1p[mask], e2_g2p[mask], w_g1p[mask], w_g2p[mask]
                    )
            RTp, RXp = rotate_spin_2_response(R1p, R2p, anglep[mask])

            eTp_list.append(np.sum(eTp * wp[mask]))
            eXp_list.append(np.sum(eXp * wp[mask]))
            RTp_list.append(np.sum(RTp))
            RXp_list.append(np.sum(RXp))

            mask = (radial_dist_m >= pixel_bin_edges[bin_i]) & (radial_dist_m < pixel_bin_edges[bin_i + 1])
            ngal_m.append(np.sum(mask))
            eTm, eXm = rotate_spin_2_ellip(e1m[mask], e2m[mask], anglem[mask])
            R1m, R2m = get_response_from_w_and_der(
                    e1m[mask], e2m[mask], wm[mask], e1_g1m[mask], e2_g2m[mask], w_g1m[mask], w_g2m[mask]
                    )
            RTm, RXm = rotate_spin_2_response(R1m, R2m, anglem[mask])

            eTm_list.append(np.sum(eTm * wm[mask]))
            eXm_list.append(np.sum(eXm * wm[mask]))
            RTm_list.append(np.sum(RTm))
            RXm_list.append(np.sum(RXm))

        return [eTp_list, eTm_list, RTp_list, RTm_list, eXp_list, eXm_list, RXp_list, RXm_list, angular_bin_mid, ngal_p, ngal_m]





