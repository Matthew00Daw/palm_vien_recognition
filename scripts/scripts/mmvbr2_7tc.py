# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:13:33 2021

@author: workstation2
"""

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sys
sys.path.append('../')
from scipy.ndimage import gaussian_filter1d,uniform_filter
from methodsapp.scripts.handler import mean_redus,diff,window_mask,select_uniq,detect_bad_candidat,trend_correct
from methodsapp.scripts.candidat import Candidat

from collections import defaultdict 
from .base_method import BaseMethod2
def distance(x,y):
    s  = (x-y)**2
    return s.sum()




def diff(x,y):
    z = np.zeros_like(x)
    for i in range(0,len(x)-1):
        z[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
    return z


class GetPeaks(BaseMethod2):

    kernel_size = 50
    kernel_size2 = 50
    trend_c = 0.0001


    def __init__(self, n_peaks=5,peack=[]):
        self.__n_peaks = n_peaks
        self.peaks=peack

    def signal_filter(self,x,y):
        dy = gaussian_filter1d(y,3)
        x1,dy = trend_correct(x,dy,self.trend_c)
        if self.kernel_size>0:
            dy = gaussian_filter1d(dy,self.kernel_size)
        return x,dy
    def execute(self,xx, signal):
        x,dy = self.signal_filter(xx,signal)
        _,dy = self.diff_smooth(x,dy)
        candidat_list = self.get_candidat(x,dy)
        self._candidat_list = candidat_list
        candidat_list = select_uniq(candidat_list)
        candidat_list = detect_bad_candidat(self.__n_peaks,candidat_list)

        if len(self.peaks)>0 and self.windows_on:
            candidat_list= window_mask(self.peaks,candidat_list,self.window_size)
        peaks=[0 for x in range(self.__n_peaks)]
        for i in range(min(self.__n_peaks,len(candidat_list))):
            p=candidat_list[i]
            if p:
                peaks[i] = (p.get_lambda())
        if len(self.peaks)==0 or not self.windows_on :
            peaks.sort()
        self.peaks=peaks
        return peaks
    def diff_smooth(self,x,y):
        dy = -diff(x,y)
        if self.kernel_size2>0:
            dy= gaussian_filter1d(dy,self.kernel_size2)
        return x,  dy

    def plot(self,x,y,fig=None,ax=None):
        if not fig:
            fig,ax = plt.subplots(4,sharex=True,figsize=(14,8))
        x2, y2 = self.signal_filter(x,y)
        ax[0].plot(x,y,markersize=1,marker='o',ls='')
        ax[1].plot(x2,y2)
        ax[0].set_title('0. Исходный сигнал')
        #ax[2].plot(x1,gaussian_filter1d(y2,5))
        dy0 = -diff(x2,y2)
        dx,dy = self.diff_smooth(x2,y2)
        candidata_list =self.get_candidat(dx,dy)
        ax[2].plot(dx,dy0)
        ax[3].plot(dx,dy)
        peack = self.execute(x,y)
        good_candidata  = self.detect_bad_candidat(candidata_list)
        ax[1].set_title('1. Сглаженный сигнал')
        for c  in candidata_list:
            px = c.get_lambda()
            py = dy[c.idx]
            ax[3].plot([px,px],[py,py],ls='',marker='o',color="k")
            ax[3].text(px,py*1.01,'{:1.2f}'.format(c.get_weight()))
        for c in peack:
            px = x[int(c)]
            ax[3].plot([px,px],[min(dy),max(dy)],ls='--')
            ax[2].plot([px,px],[min(dy),max(dy)],ls='--')
            ax[1].plot([px,px],[min(y),max(y)],ls='--')
            ax[0].plot([px,px],[min(y),max(y)],ls='--')
        ax[2].set_title('2. Производная')
        ax[3].set_title('3. Производная со сглаживанием')
        ax[3].set_xlabel('$\lambda$,нм')

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
