import numpy as np
from util import timing
from scipy.ndimage import correlate


def shape_compute(n_h: int, n_w: int, k_h: int, k_w: int, stride: int,
                  padding: int):
    '''计算二维卷积下的输出特征数.

    Parameters
    ----------
    n_h : int
        输入数据的height特征数;
    n_w : int
        输入数据的width特征数;
    k_h : int
        卷积核在height方向的大小;
    k_w : int
        卷积核在width方向的大小.
    stride : int
        卷积步长;
    padding : int
        填充长度
    '''
    return (
        (n_h + 2 * padding - k_h) // stride + 1,
        (n_w + 2 * padding - k_w) // stride + 1,
    )


@timing
def baseline_conv2d(x: np.ndarray,
                    kernel: np.ndarray,
                    stride: int = 1,
                    padding: int = 0):
    '''用最基础的for循环实现二维卷积.

    Parameters
    ----------
    x : numpy.ndarray
        输入数据, 形状为(N, in_channels, n_h, n_w).
    kernel : numpy.ndarray
        卷积核，形状为(out_channels, in_channels, k_h, k_w)
    stride : int
        卷积步长;
    padding : int
        填充长度.
    '''
    assert x.ndim == 4, "输入数据的形状必须为(N, in_channels, n_h, n_w)."
    assert kernel.ndim == 4, "卷积核形状必须为(out_channels, in_channels, k_h, k_w)."
    N, in_channels, n_h, n_w = x.shape
    out_channels, _, k_h, k_w = kernel.shape
    assert _ == in_channels, "输入数据和卷积核的in_channels不同, {}!={}".format(
        in_channels, _)
    out_h, out_w = shape_compute(n_h, n_w, k_h, k_w, stride, padding)
    x = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)],
               'constant')
    output = np.zeros((N, out_channels, out_h, out_w))
    for i in range(N):
        for j in range(out_channels):
            for k in range(out_h):
                for l in range(out_w):
                    for m in range(in_channels):
                        for row in range(k_h):
                            for col in range(k_w):
                                output[i, j, k,
                                       l] += x[i, m, k * stride + row,
                                               l * stride +
                                               col] * kernel[j, m, row, col]
    return output


@timing
def broadcast_conv2d(x: np.ndarray,
                     kernel: np.ndarray,
                     stride: int = 1,
                     padding: int = 0):
    '''用广播机制优化二维卷积.

    Parameters
    ----------
    x : numpy.ndarray
        输入数据, 形状为(N, in_channels, n_h, n_w).
    kernel : numpy.ndarray
        卷积核，形状为(out_channels, in_channels, k_h, k_w)
    stride : int
        卷积步长;
    padding : int
        填充长度.
    '''
    assert x.ndim == 4, "输入数据的形状必须为(N, in_channels, n_h, n_w)."
    assert kernel.ndim == 4, "卷积核形状必须为(out_channels, in_channels, k_h, k_w)."
    N, in_channels, n_h, n_w = x.shape
    out_channels, _, k_h, k_w = kernel.shape
    assert _ == in_channels, "输入数据和卷积核的in_channels不同, {}!={}".format(
        in_channels, _)
    out_h, out_w = shape_compute(n_h, n_w, k_h, k_w, stride, padding)
    x = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)],
               'constant')
    output = np.zeros((N, out_channels, out_h, out_w))
    for i in range(out_channels):
        for j in range(out_h):
            for k in range(out_w):
                output[:, i, j, k] = np.sum(
                    x[:, :, j * stride:j * stride + k_h,
                      k * stride:k * stride + k_w] * kernel[i],
                    axis=(-1, -2, -3),
                )
    return output


@timing
def scipy_conv2d(x: np.ndarray,
                     kernel: np.ndarray,
                     stride: int = 1,
                     padding: int = 0):
    '''用`scipy.ndimage.correlate`优化二维卷积.

    Parameters
    ----------
    x : numpy.ndarray
        输入数据, 形状为(N, in_channels, n_h, n_w).
    kernel : numpy.ndarray
        卷积核，形状为(out_channels, in_channels, k_h, k_w)
    stride : int
        卷积步长;
    padding : int
        填充长度.
    '''
    assert x.ndim == 4, "输入数据的形状必须为(N, in_channels, n_h, n_w)."
    assert kernel.ndim == 4, "卷积核形状必须为(out_channels, in_channels, k_h, k_w)."
    N, in_channels, n_h, n_w = x.shape
    out_channels, _, k_h, k_w = kernel.shape
    assert _ == in_channels, "输入数据和卷积核的in_channels不同, {}!={}".format(
        in_channels, _)
    out_h, out_w = shape_compute(n_h, n_w, k_h, k_w, stride, padding)
    x = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)],
               'constant')
    output = np.zeros((N, out_channels, out_h, out_w))
    for i in range(N):
        for j in range(out_channels):
            for k in range(in_channels):
                output[i, j, :] += correlate(
                    x[i, k],
                    kernel[j, k],
                    origin=-1,
                )[:out_h * stride:stride, :out_w * stride:stride]
    return output


@timing
def im2col_conv2d(x: np.ndarray,
                  kernel: np.ndarray,
                  stride: int = 1,
                  padding: int = 0):
    '''用im2col策略优化二维卷积.

    Parameters
    ----------
    x : numpy.ndarray
        输入数据, 形状为(N, in_channels, n_h, n_w).
    kernel : numpy.ndarray
        卷积核，形状为(out_channels, in_channels, k_h, k_w)
    stride : int
        卷积步长;
    padding : int
        填充长度.
    '''
    assert x.ndim == 4, "输入数据的形状必须为(N, in_channels, n_h, n_w)."
    assert kernel.ndim == 4, "卷积核形状必须为(out_channels, in_channels, k_h, k_w)."
    N, in_channels, n_h, n_w = x.shape
    out_channels, _, k_h, k_w = kernel.shape
    assert _ == in_channels, "输入数据和卷积核的in_channels不同, {}!={}".format(
        in_channels, _)
    out_h, out_w = shape_compute(n_h, n_w, k_h, k_w, stride, padding)
    x = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)],
               'constant')
    col = np.zeros((N, in_channels, k_h, k_w, out_h, out_w))

    for y in range(k_h):
        y_max = y + stride * out_h
        for x in range(k_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = x[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    col_filter = kernel.reshape(out_channels, -1).T
    out = col @ col_filter
    return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)