import conv2d
import numpy as np

my_conv2d = conv2d.Conv2d(
    in_channels=3,
    out_channels=1,
    kernel_size=3,
    stride=2,
    padding=3,
)

x = np.random.randn(100, 3, 256, 256)

my_conv2d.baseline(x)
my_conv2d.broadcast_opt(x)
my_conv2d.scipy_corr_opt(x)
my_conv2d.im2col_opt(x)
