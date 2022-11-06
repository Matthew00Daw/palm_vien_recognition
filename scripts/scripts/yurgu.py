# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:13:33 2021

@author: workstation2
"""

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import cwt, morlet2, find_peaks
import sys
sys.path.append('../')
from scipy.ndimage import gaussian_filter1d


def get_main_peak(peak0,num=5):
    peak= [ (p,peak0[1]['peak_heights'][i])for i,p in enumerate(peak0[0])]
    peak.sort(key=lambda x:-x[1])
    peak = [p[0] for p in peak]
    max_5 = peak[:num]
    max_5.sort()
    return max_5



def diff(x,y):
    z = np.zeros_like(x)
    for i in range(1,len(x)-1):
        z[i] = (y[i-1]-y[i+1])/(x[i-1]-x[i+1])
    return z

class GetPeaks:
    height = -100
    distance =100
    kernel_size =100
    def __init__(self, n_peaks=5):
        self.__n_peaks = n_peaks
    def set_params(self,kernel_size):
        self.kernel_size=kernel_size
    def execute2(self,x, signal):
        dy = -np.diff(signal, 1)
        dy = gaussian_filter1d(dy,self.kernel_size)
        dy2=dy[self.kernel_size//2:-self.kernel_size//2]
        x2 = x[self.kernel_size//2:-self.kernel_size//2]
        peaks = find_peaks(dy2,height=self.height,distance=self.distance)
        peaks = get_main_peak(peaks,self.__n_peaks)
        return np.array(x2[peaks]),np.array(dy2[peaks])


    def execute(self,x, signal,index=False):
        dy = -diff(x,signal )
        dy = gaussian_filter1d(dy,self.kernel_size)

        #idy2=dy[self.kernel_size//2:-self.kernel_size//2]
        #x2 = x[self.kernel_size//2:-self.kernel_size//2]
        peaks = find_peaks(dy,height=self.height,distance=self.distance)
        peaks = get_main_peak(peaks,self.__n_peaks)
        return peaks



if __name__ == "__main__":
    filepath = "spectrum_2021-08-03_14-59-58.049 .csv"
    data = pd.read_csv(filepath,
                       delimiter=';',
                       encoding="ISO-8859-1",
                       )
    signal = data.iloc[:, -1]
    x = data.iloc[:, 0]

    command = GetPeaks(5)
    retval = command.execute(x,signal)

    plt.figure(figsize=(12, 12))
    plt.plot(signal)
    plt.plot(retval, signal[retval.astype(int)], 'o', markersize=14)
    plt.grid()
    plt.show()
    print(retval)
