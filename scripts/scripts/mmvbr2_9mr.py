# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:13:33 2021

@author: workstation2
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from .base_method import BaseMethod2

from methodsapp.scripts.handler import mean_redus,diff,window_mask,select_uniq,detect_bad_candidat
from methodsapp.scripts.candidat import Candidat

def distance(x,y):
    s  = (x-y)**2
    return s.sum()


def get_kernel(size,lam=1,sig=8):
    y = np.zeros(size)
    for i in range(size):
        x=(i/size-0.5)
        y[i]=np.sin(lam*x*np.pi)*np.exp(-x**2*sig)
    y/=size
    return y

class GetPeaks(BaseMethod2):
    kernel_size = 17
    kernel_size2 = 50

    shift = (17)//2

    def __init__(self, n_peaks=5,peack=[]):
        self.__n_peaks = n_peaks
        self.peaks=peack

    def signal_filter(self,x,y):
        x1,y1= mean_redus(x,y)
        k = get_kernel(self.kernel_size,lam=2)
        dy = np.convolve(y1,k,'same')
        shift =self.shift
        return x1[shift:-shift],dy[shift:-shift]

    def execute(self,xx, signal):
        x,dy = self.signal_filter(xx,signal)
        candidat_list = self.get_candidat(x,dy)
        self._candidat_list = candidat_list
        #candidat_list = self.select_uniq(candidat_list)
        candidat_list = detect_bad_candidat(self.__n_peaks,candidat_list)

        if len(self.peaks)>0 and self.windows_on:
            candidat_list= window_mask(self.peaks,candidat_list,self.window_size)
        peaks=[0 for x in range(self.__n_peaks)]
        for i in range(self.__n_peaks):
            p=candidat_list[i]
            if p:
                peaks[i] = (p.get_lambda())
        if len(self.peaks)==0 or not self.windows_on :
            peaks.sort()
        self.peaks=peaks
        return peaks


    def plot(self,x,y,fig=None,ax=None):
        if not fig:
            fig,ax = plt.subplots(2,sharex=True,figsize=(14,8))
        x2, y2 = self.signal_filter(x,y)
        ax[0].plot(x,y,markersize=1,marker='o',ls='')
        ax[1].plot(x2,y2)
        ax[0].set_title('0. Исходный сигнал')
        candidata_list =self.get_candidat(x2,y2)
        ax[1].set_title('1. После фильра')
        for c  in candidata_list:
            px = x2[c.idx]
            py = y2[c.idx]
            ax[1].plot([px,px],[py,py],ls='',marker='o',color="k")
            ax[1].text(px,py*1.01,'{:1.2f}'.format(c.get_weight()))

        peack = self.execute(x,y)
        good_candidata  = self.detect_bad_candidat(candidata_list)
        for c in peack:
            px = x[int(c)]
            ax[1].axvline(px,ls='--')
            ax[0].axvline(px,ls='--')
        ax[1].set_xlabel('$\lambda$,нм')

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
