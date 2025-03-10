import numpy as np

def rotate_spin_2_ellip(e1, e2, angle):
    '''
    e_t = -Re[e * exp(-2i\theta)]
        = -Re[(e1 + i e2) * exp(-2i\theta)]
        = -[e1 * cos(2\theta) + e2 * sin(2\theta)]          
    e_x = -Im[e * exp(-2i\theta)]
        = -Im[(e1 + i e2) * exp(-2i\theta)]
        = e1 * sin(2\theta) - e2 * cos(2\theta)
    '''
    angle2 = 2.0 * angle
    cos2a = np.cos(angle2)
    sin2a = np.sin(angle2)
    e_t = -(e1 * cos2a + e2 * sin2a)
    e_x = e1 * sin2a - e2 * cos2a
    return e_t, e_x

def get_response_from_w_and_der(e1, e2, w, de1_dg1, de2_dg2, dw_dg1, dw_dg2):
    R11 = de1_dg1 * w + e1 * dw_dg1
    R22 = de2_dg2 * w + e2 * dw_dg2
    return R11, R22

def rotate_spin_2_response(R11, R22, angle):
    angle2 = 2.0 * angle
    cos2a = np.cos(angle2)
    sin2a = np.sin(angle2)
    Rt = cos2a ** 2 * R11 + sin2a ** 2 * R22
    Rx = sin2a ** 2 * R11 + cos2a ** 2 * R22
    return Rt, Rx

def get_angle_from_pixel(x, y, x_cen, y_cen):
    return np.arctan2(y - y_cen, x - x_cen)