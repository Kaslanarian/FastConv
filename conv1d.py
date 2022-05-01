import numpy as np
from scipy.ndimage import correlate1d
from util import timing


class Conv1d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = np.random.randn(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
        )

    def calculate_shape(self, n_features):
        return (n_features + 2 * self.padding -
                self.kernel_size) // self.stride + 1

    @timing
    def baseline(self, x: np.array) -> np.array:
        '''
        x.shape should be (
            N, in_channels, n_features
        )
        output.shape is (
            N, out_channels, n_output
        )
        '''
        kernel = self.kernel
        N, in_channels, n_features = x.shape
        assert in_channels == self.in_channels
        n_output = self.calculate_shape(n_features)
        output = np.zeros((N, self.out_channels, n_output))
        padding_x = np.pad(
            x,
            [(0, 0), (0, 0), (self.padding, self.padding)],
            'constant',
        )
        for i in range(N):
            for j in range(self.out_channels):
                for k in range(n_output):
                    for l in range(in_channels):
                        for m in range(self.kernel_size):
                            output[i, j,
                                   k] += padding_x[i, l, k * self.stride +
                                                   m] * kernel[j, l, m]
        return output

    @timing
    def scipy_corr_opt(self, x: np.array) -> np.array:
        N, in_channels, n_features = x.shape
        assert in_channels == self.in_channels
        assert self.stride == 1
        n_output = self.calculate_shape(n_features)
        padding_x = np.pad(
            x,
            [(0, 0), (0, 0), (self.padding, self.padding)],
            'constant',
        )
        output = np.zeros((N, self.out_channels, n_output))
        for i in range(in_channels):
            for j in range(self.out_channels):
                output[:, j, :] += correlate1d(
                    padding_x[:, i, :],
                    self.kernel[j, i],
                    origin=-1,
                )[..., :n_output]
        return output

    @timing
    def broadcast_opt(self, x: np.array) -> np.array:
        N, in_channels, n_features = x.shape
        assert in_channels == self.in_channels
        n_output = self.calculate_shape(n_features)
        padding_x = np.pad(
            x,
            [(0, 0), (0, 0), (self.padding, self.padding)],
            'constant',
        )
        output = np.empty((N, self.out_channels, n_output))
        for i in range(self.out_channels):
            for j in range(n_output):
                output[:, i, j] = np.sum(
                    padding_x[:, :, j * self.stride:j * self.stride +
                              self.kernel_size] * self.kernel[i, :],
                    axis=(-1, -2),
                )
        return output

    @timing
    def matmul_opt(self, x: np.array) -> np.array:
        N, in_channels, n_features = x.shape
        assert in_channels == self.in_channels
        n_output = self.calculate_shape(n_features)
        padding_x = np.pad(
            x,
            [(0, 0), (0, 0), (self.padding, self.padding)],
            'constant',
        )
        col = np.zeros((N, in_channels, self.kernel_size, n_output))

        for i in range(self.kernel_size):
            i_max = i + n_output * self.stride
            col[:, :, i, :] = padding_x[:, :, i:i_max:self.stride]

        col = col.transpose(0, 3, 1, 2).reshape(N * n_output, -1)
        col_filter = self.kernel.reshape(self.out_channels, -1).T
        out = col @ col_filter
        return out.reshape(N, n_output, -1).transpose(0, 2, 1)
