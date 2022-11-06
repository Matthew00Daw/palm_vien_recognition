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
from .che_signal import Signal
def distance(x,y):
    s  = (x-y)**2
    return s.sum()



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

from methodsapp.scripts.tools import Candidat

class GetPeaks():
    height = 0
    distance =100
    kernel_size = 100
    kernel_size2 = 1000
    peaks= None
    def __init__(self, n_peaks=5,peack=[]):
        self.__n_peaks = n_peaks
        self.peaks=peack
    def set_params(self,kernel_size):
        self.kernel_size=kernel_size

    def execute(self,x, signal):
        dy = gaussian_filter1d(signal,3)
        dy = gaussian_filter1d(dy,self.kernel_size)
        dy = -diff(x,dy)
        dy = gaussian_filter1d(dy,self.kernel_size)
        peaks = find_peaks(dy,height=self.height,distance=self.distance)
        self._candidat_list = self.get_candidat(x,dy)
        peaks = get_main_peak(peaks,self.__n_peaks)
        peaks.sort()
        return peaks
    def get_candidat(self,x,y):
        dy = -diff(x,y)
        dy = gaussian_filter1d(dy,self.kernel_size)
        peack = find_peaks(dy,height=self.height,distance=self.distance)
        candidat_list = []
        l_last = 0
        r_last=0
        for i in peack[0]:
            if dy[i]>0:
                c =Candidat(i,x,dy)
                l,r = c.get_width_index()
                if l==l_last and r==r_last:
                    if c.get_heigth()>candidat_list[-1].get_heigth():
                        candidat_list[-1]=c
                else:
                    candidat_list.append(c)
                    l_last,r_last = l,r
        candidat_list.sort(key=lambda x: -x.get_weight())
        return candidat_list 




    def execute0(self,x, signal,index=False):
        s = Signal([x,signal])
        candidat_list = self.candidat_generator(s)
        peaks = np.zeros(self.__n_peaks)
        if self.peaks:
            for x in range(self.__n_peaks):
                peaks[x] = candidat_list[find_beter(self.peaks[x],candidat_list)].lam
        else:
            candidat_list.sort(key=lambda x:-x.get_length())
            for x in range(self.__n_peaks):
                peaks[x] =candidat_list[x].lam
            peaks.sort()


        return peaks



if __name__ == "__main__":
    filepath = "spectrum_2021-07-13_15-04-25.832.csv"
    data = pd.read_csv(filepath,
                       delimiter=';',
                       encoding="ISO-8859-1",
                       )
    signal = data.iloc[:, -1]
    x = data.iloc[:, 0]

    command = GetPeaks(5)
    retval = command.execute(x,signal)

    plt.figure(figsize=(12, 12))
    plt.plot(x,signal)
    plt.plot(retval, signal[retval.astype(int)], 'o', markersize=14)
    plt.grid()
    plt.show()
    print(retval)
