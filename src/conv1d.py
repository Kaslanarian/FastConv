import numpy as np
from scipy.ndimage import correlate1d
from util import timing


def shape_compute(n_features: int, kernel_size: int, stride: int,
                  padding: int):
    '''计算一维卷积下的输出特征数.

    Parameters
    ----------
    n_features : int
        输入特征数;
    kernel_size : int
        卷积核大小;
    stride : int
        卷积步长;
    padding : int
        填充长度.
    '''
    return (n_features + 2 * padding - kernel_size) // stride + 1


@timing
def baseline_conv1d(x: np.ndarray,
                    kernel: np.ndarray,
                    stride: int = 1,
                    padding: int = 0):
    '''用最基础的for循环实现一维卷积.

    Parameters
    ----------
    x : numpy.ndarray
        输入数据, 形状为(N, in_channels, n_features).
    kernel : numpy.ndarray
        卷积核，形状为(out_channels, in_channels, kernel_size)
    stride : int
        卷积步长;
    padding : int
        填充长度.
    '''
    assert x.ndim == 3, "输入数据的形状必须为(N, in_channels, n_features)."
    assert kernel.ndim == 3, "卷积核形状必须为(out_channels, in_channels, kernel_size)."
    N, in_channels, n_features = x.shape
    out_channels, _, kernel_size = kernel.shape
    assert _ == in_channels, "输入数据和卷积核的in_channels不同, {}!={}".format(
        in_channels, _)
    n_output = shape_compute(n_features, kernel_size, stride, padding)
    x = np.pad(x, [(0, 0), (0, 0), (padding, padding)], 'constant')
    output = np.zeros((N, out_channels, n_output))
    for i in range(N):
        for j in range(out_channels):
            for k in range(n_output):
                start = k * stride
                for l in range(in_channels):
                    for m in range(kernel_size):
                        output[i, j, k] += x[i, l, start + m] * kernel[j, l, m]
    return output


@timing
def broadcast_conv1d(x: np.ndarray,
                     kernel: np.ndarray,
                     stride: int = 1,
                     padding: int = 0):
    '''用广播机制优化baseline，循环只剩下输出特征循环和输出通道循环.

    Parameters
    ----------
    x : numpy.ndarray
        输入数据, 形状为(N, in_channels, n_features).
    kernel : numpy.ndarray
        卷积核，形状为(out_channels, kernel_size)
    '''
    assert x.ndim == 3, "输入数据的形状必须为(N, in_channels, n_features)."
    assert kernel.ndim == 3, "卷积核形状必须为(out_channels, kernel_size)."
    N, in_channels, n_features = x.shape
    out_channels, _, kernel_size = kernel.shape
    assert _ == in_channels, "输入数据和卷积核的in_channels不同, {}!={}".format(
        in_channels, _)
    n_output = shape_compute(n_features, kernel_size, stride, padding)
    x = np.pad(x, [(0, 0), (0, 0), (padding, padding)], 'constant')
    output = np.zeros((N, out_channels, n_output))
    for i in range(out_channels):
        for j in range(n_output):
            output[:, i, j] = np.sum(
                x[..., j * stride:j * stride + kernel_size] * kernel[i],
                axis=(1, 2),
            )
    return output


@timing
def scipy_conv1d(x: np.ndarray,
                 kernel: np.ndarray,
                 stride: int = 1,
                 padding: int = 0):
    '''使用`scipy.ndimage.correlate1d`来加速一维卷积.

    Parameters
    ----------
    x : numpy.ndarray
        输入数据, 形状为(N, in_channels, n_features).
    kernel : numpy.ndarray
        卷积核，形状为(out_channels, kernel_size)
    '''
    assert x.ndim == 3, "输入数据的形状必须为(N, in_channels, n_features)."
    assert kernel.ndim == 3, "卷积核形状必须为(out_channels, kernel_size)."
    N, in_channels, n_features = x.shape
    out_channels, _, kernel_size = kernel.shape
    assert _ == in_channels, "输入数据和卷积核的in_channels不同, {}!={}".format(
        in_channels, _)
    n_output = shape_compute(n_features, kernel_size, stride, padding)
    x = np.pad(x, [(0, 0), (0, 0), (padding, padding)], 'constant')
    output = np.zeros((N, out_channels, n_output))
    for i in range(in_channels):
        for j in range(out_channels):
            output[:, j, :] += correlate1d(
                x[:, i, :],
                kernel[j, i],
                origin=-1,
            )[..., :stride * n_output:stride]
    return output


@timing
def im2col_conv1d(x: np.ndarray,
                  kernel: np.ndarray,
                  stride: int = 1,
                  padding: int = 0):
    '''使用im2col来加速一维卷积.

    Parameters
    ----------
    x : numpy.ndarray
        输入数据, 形状为(N, in_channels, n_features).
    kernel : numpy.ndarray
        卷积核，形状为(out_channels, kernel_size)
    '''
    assert x.ndim == 3, "输入数据的形状必须为(N, in_channels, n_features)."
    assert kernel.ndim == 3, "卷积核形状必须为(out_channels, in_channels, kernel_size)."
    N, in_channels, n_features = x.shape
    out_channels, _, kernel_size = kernel.shape
    assert _ == in_channels, "输入数据和卷积核的in_channels不同, {}!={}".format(
        in_channels, _)
    n_output = shape_compute(n_features, kernel_size, stride, padding)
    x = np.pad(x, [(0, 0), (0, 0), (padding, padding)], 'constant')
    col = np.zeros((N, in_channels, n_output, kernel_size))

    for i in range(kernel_size):
        i_max = i + n_output * stride
        col[..., i] = x[..., i:i_max:stride]

    col_filter = kernel.transpose(1, 2, 0)
    out = col @ col_filter
    return out.sum(1).transpose(0, 2, 1)


@timing
def strided_conv1d(x: np.ndarray,
                   kernel: np.ndarray,
                   stride: int = 1,
                   padding: int = 0):
    '''使用im2col+strided来加速一维卷积.

    Parameters
    ----------
    x : numpy.ndarray
        输入数据, 形状为(N, in_channels, n_features).
    kernel : numpy.ndarray
        卷积核，形状为(out_channels, kernel_size)
    '''
    assert x.ndim == 3, "输入数据的形状必须为(N, in_channels, n_features)."
    assert kernel.ndim == 3, "卷积核形状必须为(out_channels, in_channels, kernel_size)."
    N, in_channels, n_features = x.shape
    out_channels, _, kernel_size = kernel.shape
    assert _ == in_channels, "输入数据和卷积核的in_channels不同, {}!={}".format(
        in_channels, _)
    n_output = shape_compute(n_features, kernel_size, stride, padding)
    x = np.pad(x, [(0, 0), (0, 0), (padding, padding)], 'constant')

    size = x.itemsize
    col = np.lib.stride_tricks.as_strided(
        x,
        shape=(N, in_channels, n_output, kernel_size),
        strides=(size * in_channels * n_features, size * n_features,
                 size * stride, size),
    )

    col_filter = kernel.transpose(1, 2, 0)
    out = col @ col_filter
    return out.sum(1).transpose(0, 2, 1)
