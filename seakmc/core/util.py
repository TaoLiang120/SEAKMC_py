import numpy as np
from numpy import pi
import math

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"


def loop_cap(val, loopmax):
    if val > loopmax: val -= loopmax
    return val


def loop_cap4array(val, loopmax):
    np.select([val <= loopmax, val > loopmax], [val, val - loopmax])
    return val


def abs_cap(val, max_abs_val=1):
    return max(min(val, max_abs_val), -max_abs_val)


def mats_sum_mul(x, y):
    return np.sum(np.multiply(x, y))


def mat_mag(x):
    return np.sqrt(mats_sum_mul(x, x))


def mat_mag_by_vec(x, vec_axis=0):
    vec_mag = np.sqrt(np.sum(x * x, axis=vec_axis))
    return np.sum(vec_mag)


def mat_unit(x):
    return x / mat_mag(x)


def mat_lengths(x, axis=1):
    if len(x.shape) > 1:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    elif len(x.shape) == 1:
        return np.sqrt(np.sum(x ** 2))


def mat_angles(x, axis=1):
    if len(x.shape) != 2:
        print("Can only do 3x3 matrix!")
    else:
        if x.shape[0] != 3 or x.shape[1] != 3:
            print("Can only do 3x3 matrix")

    lengths = mat_lengths(x, axis=axis)
    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(x[j], x[k]) / (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / pi
    return angles


def mats_angle(x, y, Flatten=False):
    xmag = mat_mag(x)
    ymag = mat_mag(y)
    if Flatten:
        angle = abs_cap(np.sum(np.dot(x.flatten(), (y.flatten()).T)) / (xmag * ymag))
    else:
        angle = abs_cap(np.sum(np.dot(x, y.T)) / (xmag * ymag))
    return np.arccos(angle) * 180 / pi


def mats_angles(x, y):
    angles = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        angles[i] = abs_cap(np.dot(x[i], y[i].T) / (mat_mag(x[i]) * mat_mag(y[i])))
    angles = np.arccos(angles) * 180.0 / pi
    return angles


def generate_rotation_matrix(angles, Ang_Format="Radian", Ang_Style="Euler"):
    ##Ang_Style is Euler or Tait-Bryan (yaw, pitch and roll)
    if Ang_Format[0:3].upper() == "DEG":
        alpha = angles[0] * pi / 180.0
        beta = angles[1] * pi / 180.0
        gamma = angles[2] * pi / 180.0
    else:
        alpha = angles[0]
        beta = angles[1]
        gamma = angles[2]

    if Ang_Style[0:2].upper() == "EU":
        rotmat = np.array([[math.cos(beta) * math.cos(gamma),
                            math.sin(alpha) * math.sin(beta) * math.cos(gamma) - math.cos(alpha) * math.sin(gamma),
                            math.cos(alpha) * math.sin(beta) * math.cos(gamma) + math.sin(alpha) * math.sin(gamma)],
                           [math.cos(beta) * math.sin(gamma),
                            math.sin(alpha) * math.sin(beta) * math.sin(gamma) + math.cos(alpha) * math.cos(gamma),
                            math.cos(alpha) * math.sin(beta) * math.sin(gamma) - math.sin(alpha) * math.cos(gamma)],
                           [-math.sin(beta),
                            math.sin(alpha) * math.cos(beta),
                            math.cos(alpha) * math.cos(beta)]])
    else:
        rotmat = np.array([[math.cos(alpha) * math.cos(beta),
                            math.cos(alpha) * math.sin(beta) * math.sin(gamma) - math.sin(alpha) * math.cos(gamma),
                            math.cos(alpha) * math.sin(beta) * math.cos(gamma) + math.sin(alpha) * math.sin(gamma)],
                           [math.sin(alpha) * math.cos(beta),
                            math.sin(alpha) * math.sin(beta) * math.sin(gamma) + math.cos(alpha) * math.cos(gamma),
                            math.sin(alpha) * math.sin(beta) * math.cos(gamma) - math.cos(alpha) * math.sin(gamma)],
                           [-math.sin(beta),
                            math.cos(beta) * math.sin(gamma),
                            math.cos(beta) * math.cos(gamma)]])
    return rotmat


def sigmoid_function(x, scaling=1.0):
    return 1.0 / (1.0 + np.exp(-x * scaling))


def to_half_matrix(a):
    a1 = a[0][0]
    a2 = a[1][0]
    a3 = a[2][0]
    b1 = a[0][1]
    b2 = a[1][1]
    b3 = a[2][1]
    c1 = a[0][2]
    c2 = a[1][2]
    c3 = a[2][2]
    aaa = np.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)
    bbb = np.sqrt(b1 ** 2 + b2 ** 2 + b3 ** 2)
    ccc = np.sqrt(c1 ** 2 + c2 ** 2 + c3 ** 2)
    alpha = (b1 * c1 + b2 * c2 + b3 * c3) / bbb / ccc
    beta = (a1 * c1 + a2 * c2 + a3 * c3) / ccc / aaa
    gamma = (b1 * a1 + b2 * a2 + b3 * a3) / bbb / aaa
    lxx = aaa
    lxy = bbb * gamma
    lxz = ccc * beta
    lyy = np.sqrt(bbb * bbb - lxy * lxy)
    lyz = (bbb * ccc * alpha - lxy * lxz) / lyy
    lzz = np.sqrt(ccc * ccc - lxz * lxz - lyz * lyz)
    b = np.array([3, 3], dtype=float)
    b[0][0] = lxx
    b[1][0] = 0.0
    b[2][0] = 0.0
    b[0][1] = lxy
    b[1][1] = lyy
    b[2][1] = 0.0
    b[0][2] = lxz
    b[1][2] = lyz
    b[2][2] = lzz
    return b
