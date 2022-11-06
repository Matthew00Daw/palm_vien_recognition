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
from scipy.ndimage import gaussian_filter1d,uniform_filter
from scipy import signal
from .base_method import BaseMethod

def sign_chage_postiv(a):
    return  np.where((a[1:]*a[:-1]<0 )& (a[1:]<0))[0]+1

def resize(arr,new_N):
    from     scipy.interpolate import interp1d
    xrange = lambda x: np.linspace(0, 1, x)

    f = interp1d(xrange(len(arr)), arr, kind="linear")
    new_arr = f(xrange(new_N))
    return new_arr

def get_main_peak(peak0,num=5):
    peak= [ (p,peak0[1]['peak_heights'][i])for i,p in enumerate(peak0[0])]
    peak.sort(key=lambda x:-x[1])
    peak = [p[0] for p in peak]
    max_5 = peak[:num]
    max_5.sort()
    return max_5

def multi_kenrel_filter(sig_in,kernel_list,min_size,number,step):
    sig = sig_in
    filtered = np.zeros(len(sig))+sig.min()
    for kernel in kernel_list:
        for x in range(number):
            win = resize(kernel,min_size+step*x)
            #win /= len(win)
            filtered0 = signal.correlate(sig, win,mode="same")
            filtered = np.maximum.reduce([filtered0,filtered])
    return filtered

def correlate(sig_in,kernel_list,min_size,number,step,height):
    filtred = multi_kenrel_filter(sig_in,kernel_list,min_size,number,step)
    peak = find_peaks(filtred,height=height,distance=300)
    return peak

class GetPeaks(BaseMethod):
    height = -100
    distance =100
    kernel_size =200
    threshoold = -200
    uniform_kernel=1000
    min_size=100
    number =100
    step=10
    small_kernel=3
    def __init__(self, n_peaks=5):
        self.__n_peaks = n_peaks
        self.kernel_list = np.genfromtxt('methodsapp/scripts/kernel_list.txt')
        #self.small_kernel=small_kernel
    def filter(self,x,signal):
        return multi_kenrel_filter(signal,self.kernel_list,self.min_size,self.number,self.step)
    def execute(self,x, signal):
        sig=gaussian_filter1d(signal,self.small_kernel)# -0.1*uniform_filter(signal,self.uniform_kernel)
        #sig = signal - uniform_filter(signal,500)
        filtred = self.filter(x,sig)

        dy2=filtred[self.kernel_size//2:-self.kernel_size//2]
        x2 = x[self.kernel_size//2:-self.kernel_size//2]
        peaks = find_peaks(dy2,height=self.height,distance=self.distance)
        peaks = get_main_peak(peaks,self.__n_peaks)
        return peaks


if __name__ == "__main__":
    filepath = "spectrum_2021-07-13_15-04-27.832.csv"
    data = pd.read_csv(filepath,
                       delimiter=';',
                       encoding="ISO-8859-1",
                       )
    signal = data.iloc[:, -1]
    x = data.iloc[:, 0]

    command = GetPeaks(5)
    retval = command.execute(x,signal)

    plt.figure(figsize=(12, 12))
    sig=signal-0*uniform_filter(signal,1000)
    plt.plot(sig)
    plt.plot(retval, sig[retval.astype(int)], 'o', markersize=14)
    plt.grid()
    plt.show()
    print(retval)
