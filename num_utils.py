"""Numerical utilities."""

import ctypes
import ctypes.util
from ctypes import c_float, c_int32
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import numpy.ctypeslib as npct
from PIL import Image
from PIL.Image import NEAREST, BILINEAR, BICUBIC, LANCZOS  # pylint: disable=unused-import
import pywt
from scipy.linalg import blas

# Machine epsilon for float32
EPS = np.finfo(np.float32).eps


# pylint: disable=no-member
def dot(x, y):
    """Returns the dot product of two float32 arrays with the same shape."""
    return blas.sdot(x.ravel(), y.ravel())


# pylint: disable=no-member
def axpy(a, x, y):
    """Sets y = a*x + y for float a and float32 arrays x, y and returns y."""
    y_ = blas.saxpy(x.ravel(), y.ravel(), a=a).reshape(y.shape)
    if y is not y_:
        y[:] = y_
    return y


ROW_MAJOR, COL_MAJOR = 101, 102
UPPER, LOWER = 121, 122
LEFT, RIGHT = 141, 142

c_float_arr2d = npct.ndpointer(np.float32, ndim=2, flags='C')

try:
    mkl = ctypes.cdll.LoadLibrary(ctypes.util.find_library('mkl_rt'))
    mkl.cblas_ssymm.restype = None
    mkl.cblas_ssymm.argtypes = [
        c_int32, c_int32, c_int32, c_int32, c_int32, c_float, c_float_arr2d, c_int32,
        c_float_arr2d, c_int32, c_float, c_float_arr2d, c_int32]

    def symm(a, b, c=None, alpha=1, beta=0, side=LEFT, uplo=UPPER):
        """Wraps MKL's cblas_ssymm() scalar-symmetric matrix-matrix product. If side is LEFT, sets
        C = alpha * A @ B + beta * C; if RIGHT, sets C = alpha * B @ A + beta * C. See
        https://software.intel.com/en-us/node/520779.
        """
        if c is None:
            c = np.zeros_like(b)
        mkl.cblas_ssymm(ROW_MAJOR, side, uplo, c.shape[0], c.shape[1],
                        alpha, a, a.shape[0], b, c.shape[1], beta, c, c.shape[1])
        return c
except (AttributeError, OSError):
    pass


def norm2(arr):
    """Returns 1/2 the L2 norm squared."""
    return np.sum(arr**2) / 2


def p_norm(arr, p=2):
    """Returns the pth power of the p-norm and its gradient."""
    loss = np.sum(abs(arr)**p)
    grad = p * np.sign(arr) * abs(arr)**(p-1)
    return loss, grad


def normalize(arr):
    """Normalizes an array in-place to have an L1 norm equal to its size."""
    arr /= np.mean(abs(arr)) + EPS
    return arr


def resize(a, hw, method=Image.LANCZOS):
    """Resamples an image array in CxHxW format to a new HxW size. The interpolation is performed
    in floating point and the result dtype is numpy.float32."""
    def _resize(a, b):
        b[:] = Image.fromarray(a).resize((hw[1], hw[0]), method)

    a = np.float32(a)
    if a.ndim == 2:
        b = np.zeros(hw, np.float32)
        _resize(a, b)
        return b
    ch = a.shape[0]
    b = np.zeros((ch, hw[0], hw[1]), np.float32)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futs = [ex.submit(_resize, a[i], b[i]) for i in range(ch)]
        _ = [fut.result() for fut in futs]

    return b


def roll_by_1(arr, shift, axis):
    """Rolls a 3D array in-place by a shift of one element. Axes 1 and 2 only."""
    if axis == 1:
        if shift == -1:
            line = arr[:, 0, :].copy()
            arr[:, :-1, :] = arr[:, 1:, :]
            arr[:, -1, :] = line
        elif shift == 1:
            line = arr[:, -1, :].copy()
            arr[:, 1:, :] = arr[:, :-1, :]
            arr[:, 0, :] = line
    elif axis == 2:
        if shift == -1:
            line = arr[:, :, 0].copy()
            arr[:, :, :-1] = arr[:, :, 1:]
            arr[:, :, -1] = line
        elif shift == 1:
            line = arr[:, :, -1].copy()
            arr[:, :, 1:] = arr[:, :, :-1]
            arr[:, :, 0] = line
    else:
        raise ValueError('Unsupported shift or axis')
    return arr


def roll2(arr, xy):
    """Translates an array by the shift xy, wrapping at the edges."""
    if (xy == 0).all():
        return arr
    # return np.roll(arr, xy, (2, 1))
    return np.roll(np.roll(arr, xy[0], -1), xy[1], -2)


def gram_matrix(feat):
    """Computes the Gram matrix corresponding to a feature map."""
    n, mh, mw = feat.shape
    feat = feat.reshape((n, mh * mw))
    return np.dot(feat, feat.T) / np.float32(feat.size)
    # return blas.ssyrk(1 / feat.size, feat)


def tv_norm(x, beta=2):
    """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis and [3]."""
    x_diff = x - roll_by_1(x.copy(), -1, axis=2)
    y_diff = x - roll_by_1(x.copy(), -1, axis=1)
    grad_norm2 = x_diff**2 + y_diff**2 + EPS
    loss = np.sum(grad_norm2**(beta/2))
    dgrad_norm = (beta/2) * grad_norm2**(beta/2 - 1)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad -= roll_by_1(dx_diff, 1, axis=2)
    grad -= roll_by_1(dy_diff, 1, axis=1)
    return loss, grad


def wt_norm(x, p=1, wavelet='haar'):
    """Computes the wavelet denoising p-norm and its gradient."""
    coeffs = pywt.wavedec2(x, wavelet, mode='periodic')
    coeffs[0][:] = 0
    inv = pywt.waverec2(coeffs, wavelet, mode='periodic')
    if inv.shape != x.shape:
        inv = inv[:, :x.shape[1], :x.shape[2]]
    loss, grad = p_norm(inv, p)
    return loss, grad
