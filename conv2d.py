import numpy as np
from scipy.ndimage import correlate
from util import timing


class Conv2d:
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
            self.kernel_size,
        )

    def calculate_shape(self, n_h, n_w):
        return (
            (n_h + 2 * self.padding - self.kernel_size) // self.stride + 1,
            (n_w + 2 * self.padding - self.kernel_size) // self.stride + 1,
        )

    @timing
    def baseline(self, x: np.array) -> np.array:
        N, in_channels, n_h, n_w = x.shape
        assert in_channels == self.in_channels
        h_out, w_out = self.calculate_shape(n_h, n_w)
        data = np.pad(
            x,
            [(0, 0), (0, 0), (self.padding, self.padding),
             (self.padding, self.padding)],
            'constant',
        )
        output = np.zeros((N, self.out_channels, h_out, w_out))
        stride = self.stride
        kernel = self.kernel
        for i in range(N):
            for j in range(self.out_channels):
                for k in range(h_out):
                    for l in range(w_out):
                        for m in range(in_channels):
                            for row in range(self.kernel_size):
                                for col in range(self.kernel_size):
                                    output[i, j, k,
                                           l] += data[i, m, k * stride + row,
                                                      l * stride +
                                                      col] * kernel[j, m, row,
                                                                    col]
        return output

    @timing
    def scipy_corr_opt(self, x: np.array) -> np.array:
        N, in_channels, n_h, n_w = x.shape
        assert in_channels == self.in_channels
        h_out, w_out = self.calculate_shape(n_h, n_w)
        data = np.pad(
            x,
            [(0, 0), (0, 0), (self.padding, self.padding),
             (self.padding, self.padding)],
            'constant',
        )
        output = np.zeros((N, self.out_channels, h_out, w_out))
        for i in range(N):
            for j in range(self.out_channels):
                for k in range(self.in_channels):
                    output[i, j, :] += correlate(
                        data[i, k],
                        self.kernel[j, k],
                        origin=-1,
                    )[:h_out * self.stride:self.stride, :w_out *
                      self.stride:self.stride]
        return output

    @timing
    def broadcast_opt(self, x: np.array) -> np.array:
        N, in_channels, n_h, n_w = x.shape
        assert in_channels == self.in_channels
        h_out, w_out = self.calculate_shape(n_h, n_w)
        data = np.pad(
            x,
            [(0, 0), (0, 0), (self.padding, self.padding),
             (self.padding, self.padding)],
            'constant',
        )
        output = np.empty((N, self.out_channels, h_out, w_out))
        for i in range(self.out_channels):
            for j in range(h_out):
                for k in range(w_out):
                    output[:, i, j, k] = np.sum(
                        data[:, :, j * self.stride:j * self.stride +
                             self.kernel_size, k *
                             self.stride:k * self.stride + self.kernel_size] *
                        self.kernel[i],
                        axis=(-1, -2, -3),
                    )
        return output

    @timing
    def im2col_opt(self, x: np.array) -> np.array:
        N, in_channels, n_h, n_w = x.shape
        assert in_channels == self.in_channels
        h_out, w_out = self.calculate_shape(n_h, n_w)
        data = np.pad(
            x,
            [(0, 0), (0, 0), (self.padding, self.padding),
             (self.padding, self.padding)],
            'constant',
        )
        col = np.zeros((
            N,
            in_channels,
            self.kernel_size,
            self.kernel_size,
            h_out,
            w_out,
        ))

        for y in range(self.kernel_size):
            y_max = y + self.stride * h_out
            for x in range(self.kernel_size):
                x_max = x + self.stride * w_out
                col[:, :, y, x, :, :] = data[:, :, y:y_max:self.stride,
                                             x:x_max:self.stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * h_out * w_out, -1)
        col_filter = self.kernel.reshape(self.out_channels, -1).T
        out = col @ col_filter
        return out.reshape(N, h_out, w_out, -1).transpose(0, 3, 1, 2)
