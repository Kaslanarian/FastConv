from conv1d import *
from util import avg_timing
import util

import matplotlib.pyplot as plt
import torch

try:
    import seaborn as sns
    sns.set()
except:
    pass

util.count = False


def conv1d_test():
    '''一维卷积优化实验
    
    - baseline:朴素for循环实现;
    - broadcast:基于广播的优化;
    - scipy:基于scipy.ndimage.correlate1d进行加速;
    - im2col:用im2col将卷积转换成矩阵乘积;
    - strided:用im2col+strided进行优化.

    实验方法:

    控制N=100, in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=0不变
    然后针对不同输入特征数(3-100)的数据进行卷积，独立重复50次实验取平均。
    '''
    baseline_list = []
    broadcast_list = []
    scipy_list = []
    im2col_list = []
    strided_list = []

    T = 50
    N = 100
    kernel_size = in_channels = 3
    out_channels = 2
    kernel = np.random.rand(out_channels, in_channels, kernel_size)
    for n_features in range(3, 101):
        X = np.random.randn(N, in_channels, n_features)

        baseline_list.append(avg_timing(T, baseline_conv1d, X, kernel))
        broadcast_list.append(avg_timing(T, broadcast_conv1d, X, kernel))
        scipy_list.append(avg_timing(T, scipy_conv1d, X, kernel))
        im2col_list.append(avg_timing(T, im2col_conv1d, X, kernel))
        strided_list.append(avg_timing(T, strided_conv1d, X, kernel))

        print(n_features)

    plt.plot(np.log10(baseline_list), label="baseline")
    plt.plot(np.log10(broadcast_list), label="broadcast")
    plt.plot(np.log10(scipy_list), label="scipy")
    plt.plot(np.log10(im2col_list), label="im2col")
    plt.plot(np.log10(strided_list), label="strided")

    plt.legend()
    plt.savefig("../img/test.png")


def im2col1d_test():
    '''针对im2col和im2col+strided的实验
    
    - im2col:用im2col将卷积转换成矩阵乘积;
    - strided:用im2col+strided进行优化;
    - torch:pytorch实现的卷积

    实验方法:

    控制N=100, in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=0不变
    然后针对不同输入特征数(3-300)的数据进行卷积，独立重复10次实验取平均。
    '''
    im2col_list = []
    strided_list = []
    torch_list = []

    T = 50
    N = 100
    kernel_size = in_channels = 3
    out_channels = 2
    kernel = np.random.rand(out_channels, in_channels, kernel_size)
    for n_features in range(3, 301):
        X = np.random.randn(N, in_channels, n_features)

        im2col_list.append(avg_timing(T, im2col_conv1d, X, kernel))
        strided_list.append(avg_timing(T, strided_conv1d, X, kernel))
        torch_list.append(
            avg_timing(T, torch.conv1d, torch.tensor(X), torch.tensor(kernel)))

        print(n_features)

    plt.plot(im2col_list, label="im2col")
    plt.plot(strided_list, label="strided+im2col")
    plt.plot(torch_list, label="pytorch")

    plt.legend()
    plt.savefig("../img/test_im2col.png")
