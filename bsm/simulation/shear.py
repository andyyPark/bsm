import galsim
import numpy as np


class ShearTXConstant(object):
    """
    Constant shear in the full exposure
    Parameters
    ----------
    g1, g2:    Constant shear distortion
    """

    def __init__(self, mode, g_dist="gt", shear_value=0.02):
        if mode == 0:
            self.gv = shear_value * -1.0
        elif mode == 1:
            self.gv = shear_value
        else:
            raise ValueError("mode not supported")
        self.g_dist = g_dist
        return

    def distort_galaxy(self, gso, shift, redshift):
        """This function distorts the galaxy's shape and position
        Parameters
        ---------
        gso (galsim object):        galsim galaxy
        shift (galsim.PositionD):   position of the galaxy
        redshift (float):           redshift of galaxy

        Returns
        ---------
        gso, shift:
            distorted galaxy object and shift
        """
        theta = np.arctan2(shift.y, shift.x)
        if self.g_dist == "gt":
            g1 = self.gv * np.cos(2.0 * theta)
            g2 = self.gv * np.sin(2.0 * theta)
        else:
            g1 = self.gv * np.sin(2.0 * theta)
            g2 = -self.gv * np.cos(2.0 * theta)

        shear = galsim.Shear(g1=g1, g2=g2)
        gso = gso.shear(shear)
        shift = shift.shear(shear)
        return _get_shear_res_dict(gso, shift, gamma1=g1, gamma2=g2, kappa=0)


def _get_shear_res_dict(gso, lensed_shift, gamma1=-1, gamma2=-1, kappa=-1):
    shear_res_dict = {
        "gso": gso,
        "lensed_shift": lensed_shift,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "kappa": kappa,
    }
    return shear_res_dict
