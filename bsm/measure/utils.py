import anacal
import lsst.geom as lsst_geom
import numpy as np


def get_psf_array(exposure, ngrid, psf_rcut=26, dg=250, gcent=None):
    """This function returns the PSF model at the center of the exposure."""
    if gcent is None:
        gcent = dg // 2
    bbox = exposure.getBBox()
    width, height = bbox.getWidth(), bbox.getHeight()

    width = (width // dg) * dg - 1
    height = (height // dg) * dg - 1
    # Calculate the central point
    x_array = np.arange(0, width, dg, dtype=int) + gcent
    y_array = np.arange(0, height, dg, dtype=int) + gcent
    nx, ny = len(x_array), len(y_array)
    out = np.zeros((ngrid, ngrid))
    ncount = 0.0
    for j in range(ny):
        yc = int(y_array[j])
        for i in range(nx):
            xc = int(x_array[i])
            centroid = lsst_geom.Point2I(xc, yc)
            data = exposure.getPsf().computeImage(centroid).getArray()
            dx = data.shape[0]
            assert dx == data.shape[1]
            if ngrid > dx:
                shift = (ngrid - dx + 1) // 2
                out[shift : shift + dx, shift : shift + dx] = (
                    out[shift : shift + dx, shift : shift + dx] + data
                )
            else:
                shift = -(ngrid - dx) // 2
                out = out + data[shift : shift + ngrid, shift : shift + ngrid]
            ncount += 1
    out = out / ncount
    anacal.fpfs.base.truncate_square(out, psf_rcut)
    return out


def get_gridpsf_obj(exposure, ngrid, psf_rcut=26, dg=250, gcent=None):
    """This function returns the PSF model object at the center of the
    exposure."""
    if gcent is None:
        gcent = dg // 2
    bbox = exposure.getBBox()
    width, height = bbox.getWidth(), bbox.getHeight()

    width = (width // dg) * dg - 1
    height = (height // dg) * dg - 1
    # Calculate the central point
    x_array = np.arange(0, width, dg, dtype=int) + gcent
    y_array = np.arange(0, height, dg, dtype=int) + gcent
    nx, ny = len(x_array), len(y_array)
    out = np.zeros((ny, nx, ngrid, ngrid))
    for j in range(ny):
        yc = int(y_array[j])
        for i in range(nx):
            xc = int(x_array[i])
            centroid = lsst_geom.Point2I(xc, yc)
            data = exposure.getPsf().computeImage(centroid).getArray()
            dx = data.shape[0]
            assert dx == data.shape[1]
            if ngrid > dx:
                shift = (ngrid - dx + 1) // 2
                out[j, i, shift : shift + dx, shift : shift + dx] = data
            else:
                shift = -(ngrid - dx) // 2
                out[j, i] = data[shift : shift + ngrid, shift : shift + ngrid]
            anacal.fpfs.base.truncate_square(out[j, i], psf_rcut)
    return anacal.psf.GridPsf(x0=0, y0=0, dx=dg, dy=dg, model_array=out)