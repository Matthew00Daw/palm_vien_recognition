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


class Candidat:
    def __init__(self,idx,x,y):
        self.idx = idx
        self.x = x
        self.y = y
    def get_width_index(self):
        i=self.idx
        for i in range(self.idx,len(self.x)-2,1):
            if (self.y[i]<self.y[i+1] ):
                break
        r=i
        for i in range(self.idx,1,-1):
            if (self.y[i-1]>self.y[i]):
                break
        l=i
        return l,r
    def get_width(self):
        l,r= self.get_width_index()
        return self.x[r]-self.x[l]
    def half_height(self):
        m = self.get_m()
        y =  self.signal.get_gaus_data()
        hh = (y[m]+y[self.idx])/2
        return hh
    def get_assimetry(self):
        if hasattr(self,'assimetry'):
            return self.assimetry
        else:
            x=self.x
            l,r= self.get_width_index()
            m= self.idx
            self.assimetry= -(x[r]-x[m])/(x[l]-x[m])
            return self.assimetry
    def get_vector(self):
        return np.array([
            self.x[self.idx],
            self.get_width(),
            self.y[y],

        ])
    def get_heigth(self):
        return self.y[self.idx]
    def get_weight(self):
        #l,r= self.get_width_index()
        return self.get_heigth()
    def get_lambda(self):
        return self.x[self.idx]

        return np.argmin(dist_list)
def find_beter(v,candidat_list):
    dist_list = [distance(v,x.get_vector()) for x in candidat_list]
    return np.argmin(dist_list)


class GetPeaks():
    height = 0
    distance =100
    kernel_size = 100
    kernel_size2 = 1000
    peaks= None
    windows_on = True
    window_size = 2

    def __init__(self, n_peaks=5,peack=[]):
        self.__n_peaks = n_peaks
        self.peaks=peack
    
    def set_params(self,kernel_size):
        self.kernel_size=kernel_size
    
    def signal_filter(self,x,y):
        dy = gaussian_filter1d(y,3)
        dy = gaussian_filter1d(dy,self.kernel_size)
        ks =self.kernel_size*2
        return x,dy

    def window_mask(self,candidat_list):
        ret = [0]*len(self.peaks)
        for i,pk in enumerate(self.peaks):
            tmp = []
            for c in candidat_list:
                if abs(c.get_lambda()-pk)<self.window_size:
                    tmp.append(c)
            if len(tmp)>0:
                tmp.sort(key=lambda x:-x.get_weight())
                self.peaks[i]= tmp[0].get_lambda()
                ret[i]=tmp[0]
        return ret
 

    
    def detect_bad_candidat(self,cl):
        if self.__n_peaks<len(cl):
            ln = (min(len(cl)-self.__n_peaks,3))
            nois_w = cl[self.__n_peaks].get_weight()

            ret = []
            cl2 = cl[:self.__n_peaks][::-1]

            for x in cl2:
                if x.get_weight()>nois_w*1.8:
                    ret.append(x)
                else:
                    nois_w=x.get_weight()
            return ret[::-1]
        else:
            return cl

    def execute(self,xx, signal):
        x,dy = self.signal_filter(xx.tolist(),signal.tolist())
        candidat_list = self.get_candidat(x,dy)
       
        candidat_list.sort(key=lambda x: -x.get_weight())
        #candidat_list = self.detect_bad_candidat(candidat_list)

        self._candidat_list = candidat_list

        if len(self.peaks)>0:
            candidat_list= self.window_mask(candidat_list)
        peaks=[]
        for  p in candidat_list[:self.__n_peaks]:
            if p:
                peaks.append(p.get_lambda())
            else:
                peaks.append(0)
        if self.windows_on and len(self.peaks)==0:
            peaks.sort()
            self.peaks=peaks[:self.__n_peaks]
 
        peack = [ ]
        for p in candidat_list[:self.__n_peaks]:
            if p:
                peack.append(p.idx)
            else:
                peack.append(0)

   #     peack.sort()
   #     peack += [0 for x in range(self.__n_peaks-len(peack))]
        return peack
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
