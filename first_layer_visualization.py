import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

if __name__ == '__main__' :
    layer1_weight = np.loadtxt('layer1_weight_manual_40.txt')
    weight_fft = fft(layer1_weight, n=128, axis=1)
    abs_fft = np.abs(weight_fft)
    abs_fft_ranged = abs_fft[:, 0:64]
    for i in range(abs_fft.shape[0]):
        graph = plt.subplot(4, 10, i+1)
        plt.plot(abs_fft_ranged[i])
        # plt.plot(abs_fft[1][0:32])
    plt.show()
    fre_index = np.argmax(abs_fft_ranged, axis=1)
    sort_index = np.argsort(fre_index)
    sorted_fft = abs_fft_ranged[sort_index]
    # logrith_fft = 20*np.log10(sorted_fft)
    # plt.imshow(np.transpose(sorted_fft))
    plt.imshow(sorted_fft)
    plt.show()