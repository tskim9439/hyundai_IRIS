import numpy as np
import scipy
from scipy.signal import welch, hann, windows
import matplotlib.pyplot as plt

# FILTERA Generates an A-weighting filter.
#    FILTERA Uses a closed-form expression to generate
#    an A-weighting filter for arbitary frequencies.
#
# Author: Douglas R. Lanman, 11/21/05

# Define filter coefficients.
# See: http://www.beis.de/Elektronik/AudioMeasure/
# WeightingFilters.html#A-Weighting
def filter_A(F):
    c1 = 3.5041384e16
    c2 = 20.598997 ** 2
    c3 = 107.65265 ** 2
    c4 = 737.86223 ** 2
    c5 = 12194.217 ** 2

    f = F
    arr = np.array(list(map(lambda x : 1e-17 if x == 0 else x, f)))

    num = np.sqrt(c1 * (arr ** 8))
    den = (f ** 2 + c2) * np.sqrt((f ** 2 + c3) * (f**2 + c4)) * (f ** 2 + c5)
    A = num / den
    return A

def dBA_metric(y, gt, plot=True):
    """
    |args|
    :y: generated sound data, it's shape should be (time, 8)
    :gt: ground truth data, it's shape should be (time, 8)
    :plot: if True, plot graph of each channels
    """
    d = gt
    e = gt - y
    Tn = y.shape[0]
    K = 8
    M = 8
    """Post processing : performance metric and plots"""
    p_ref = 20e-6
    fs = 2000
    Nfft = fs
    noverlap = Nfft / 2
    t = (np.arange(Tn) / fs)[np.newaxis, :]
    #win = hann(fs, False)
    win = windows.hamming(fs)
    #autopower calculation
    D = np.zeros((int(Nfft/2 + 1), M))
    E = np.zeros((int(Nfft/2 + 1), M))
    for m in range(M):
        F, D[:,m] = welch(d[:, m], fs=fs, window=win, noverlap=noverlap, nfft=Nfft, return_onesided=True, detrend=False)
        F, E[:,m] = welch(e[:, m], fs=fs, window=win, noverlap=noverlap, nfft=Nfft, return_onesided=True, detrend=False)
    
    A = filter_A(F)
    AA = np.concatenate([[A]] * M, axis=0).transpose(1,0)
    D_A = D * AA ** 2 / p_ref ** 2
    E_A = E * AA ** 2 / p_ref ** 2
    
    # perfomance metric calculation
    D_A_dBA_Sum = np.zeros((1,M))
    E_A_dBA_Sum = np.zeros((1,M))
    freq_range = np.arange(500)
    result = []
    E_result = []
    for m in range(M):
        D_A_dBA_Sum[0, m] = 10 * np.log10(np.sum(D_A[freq_range, m]))
        E_A_dBA_Sum[0, m] = 10 * np.log10(np.sum(E_A[freq_range, m]))
        result.append(D_A_dBA_Sum[0,m] - E_A_dBA_Sum[0,m])
        E_result.append(E_A_dBA_Sum[0, m])
    result.append(np.array(np.mean(result)))
    avg_result = np.mean(result)
    #E_result = np.array(np.mean(E_result))
    E_re = np.mean(np.array(E_result))
    
    if plot:
        for m in range(M):
            plt.subplot(2, 4, m+1)
            da = D_A[:,m]
            ea = E_A[:,m]
            plt.plot(F, 10 * np.log10(da), color="red", label=f'D{D_A_dBA_Sum[0, m]:.2f}dBA', linewidth=0.1)
            plt.plot(F, 10 * np.log10(ea), color="blue", label=f'E{E_A_dBA_Sum[0, m]:.2f}dBA', linewidth=0.1)
            plt.legend()
            plt.ylim((10, 60))
            plt.xlim((20,1000))
            plt.xlabel('Frequency(Hz)')
            plt.ylabel('Level(dBA)')
        plt.tight_layout()
        plt.show()
    
    return avg_result, E_re