import numpy as np
from conv1d import Conv1d

conv = Conv1d(3, 2, 3)
x = np.random.randn(1000, 3, 256)
y1 = conv.baseline(x)
y2 = conv.scipy_corr_opt(x)
y4 = conv.broadcast_opt(x)
y3 = conv.matmul_opt(x)
