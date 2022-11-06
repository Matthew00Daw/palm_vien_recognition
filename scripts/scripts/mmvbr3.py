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

from .base_processing import BaseProcessor
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

class Candidat:
    def __init__(self,idx,signal):
        data =signal.get_data()
        x = data[0]
        y =  signal.get_gaus_data()
        dy =  signal.get_d_gaus_data()
        self.lam=x[idx]
        self.h=y[idx]
        self.dh=dy[idx]
        self.dx=(x[idx+1]-x[idx-1])/2
        self.idx = idx
        self.signal=signal
    def get_m(self):
        peak = self.signal.get_peack()
        for i in range(len(peak)-1):
            if peak[i+1]>self.idx:
                break
        return peak[i]
    def half_height(self):
        m = self.get_m()
        y =  self.signal.get_gaus_data()
        hh = (y[m]+y[self.idx])/2
        return hh
    def get_p(self):
        hh = self.half_height()
        y = self.signal.get_gaus_data()
        r=self.idx
        for i in range(self.idx,0,-1):
            if y[i]> hh:
                r=i
                break
        l=r
        for i in range(r-1,0,-1):
            if y[i]< hh:
                l=i
                break
        return l,r
    def get_heigth(self):
        #if hasattr(self,'assimetry'):
        #    return self.height
        #else:
        y =  self.signal.get_d2_gaus_data()
        y0 =  self.signal.get_gaus_data()
        p= find_peaks(y**2)[0]
        for i,x in enumerate(p):
            if x>self.idx:
                break
        self.height =y0[p[i-1]]-y0[p[i]]
        return self.height

    def get_assimetry(self):
        if hasattr(self,'assimetry'):
            return self.assimetry
        else:
            x=self.signal.get_data()[0]
            l,r= self.get_p()
            m=self.get_m()
            self.assimetry= -(x[r]-x[m])/(x[l]-x[m])
            return self.assimetry
    def get_vector(self):
        return np.array([
            self.lam/1500/2,
            self.h,
            self.dh,
            self.get_heigth(),
            self.get_assimetry()

        ])
    def get_length(self):
        x =self.get_vector()
        return (x[2]).sum()
    def calc_rating(self,candidat_list):
        v = self.get_vector()
        self.rating = [{'idx':i,'distance':distance(v,x)} for i,x in enumerate(candidat_list)]
        return np.argmin(dist_list)
def find_beter(v,candidat_list):
    dist_list = [distance(v,x.get_vector()) for x in candidat_list]
    return np.argmin(dist_list)

def resemple(data,N):
    res = np.zeros((len(data),len(data[0])//N))
    for x in range(0,(len(data[0])-1)//N):
        res[0][x]=np.mean(data[0][x*N:x*N+N])
        res[1][x]=np.mean(data[1][x*N:x*N+N])
    return res

class GetPeaks():
    height = 0
    distance =1
    kernel_size = 100
    kernel_size2 = 1000
    peaks= None
    def __init__(self, n_peaks=5,peack=[]):
        self.__n_peaks = n_peaks
        self.peaks=peack
    def set_params(self,kernel_size):
        self.kernel_size=kernel_size

    def execute(self,x, signal,index=False):
        N_sempl = 10
        res = resemple([x,signal],N_sempl)
        dy = gaussian_filter1d(res[1],10)
        dy  = -diff(res[0],dy)
        candidat_list= find_peaks(dy,height=self.height,distance=self.distance)
        peaks = get_main_peak(candidat_list,self.__n_peaks)

        peaks.sort()
        return [x*N_sempl for x in peaks]
        return np.array(res[0][peaks])



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
