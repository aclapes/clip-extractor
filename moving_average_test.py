import numpy as np

def moving_average_filter_in_1d_vec(x, kernel_size=5):
    v = np.zeros(len(x), dtype=np.float32)

    for i in range(0, kernel_size/2):
        v[i] = np.mean(x[max(0,i-kernel_size/2):i+kernel_size/2+1])

    acc = np.sum(x[:kernel_size],dtype=np.float32)
    for i in range(kernel_size/2, len(x) - kernel_size/2 - 1):
        v[i] = acc / kernel_size
        acc += x[i+kernel_size/2+1]
        acc -= x[i-kernel_size/2]
    v[i+1] = acc / kernel_size

    for i in range(len(x)-kernel_size/2, len(x)):
        v[i] = np.mean(x[max(0,i-kernel_size/2):i+kernel_size/2+1])

    return v


a = moving_average_filter_in_1d_vec(np.array([1,0,0,0,1,1,1,1,1,0]),3)
print a