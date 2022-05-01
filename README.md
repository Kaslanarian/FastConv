# FastConv

尝试各种方法，避免NumPy基础卷积操作中的多重循环：样本循环、卷积循环、通道循环，从而加速卷积过程。

## Conv1d

1. 利用`scipy.ndimage.correlate1d`消除样本循环和卷积循环；
2. 利用NumPy的广播机制消除样本循环和一部分通道循环；
3. 利用类似im2col策略消除所有循环。
