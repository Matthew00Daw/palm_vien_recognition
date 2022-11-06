# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:13:33 2021

@author: workstation2
"""

import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import cwt, morlet2, find_peaks,detrend,savgol_filter
import sys
sys.path.append('../')
from scipy.ndimage import gaussian_filter1d,uniform_filter
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
def diff2(x,y):
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
        for i in range(self.idx,len(self.y)-3,1):
            if (self.y[i]<self.y[i+1] or self.y[i]<0):
                break
        r=i

        for i in range(self.idx,1,-1):
            if (self.y[i+1]<0):
                break
        m=i
        for i in range(m,1,-1):
            if (self.y[i-1]*self.y[i]<0):
                break
        l=i
        self.M = m
        return l,m,r

    def get_width(self):
        l,m,r= self.get_width_index()
        return self.x[r]-self.x[l]
    def line_approx(self,idx,y):
        l,m,r= self.get_width_index()
        return (y[r]-y[l])/(self.x[r]-self.x[l])*(self.x[idx]-self.x[l])+y[l]
    def patch_signal(self,y):
        l,m,r= self.get_width_index()

        #dy  =  diff2(self.x[l:r],y[l:r])
        A = (y[r]-y[l])/(self.x[r]-self.x[l])
        for i in range(l,r):
            y[i] = A*(self.x[i]-self.x[l])+y[l]
        #A = y[l:r].mean()
        #for i in range(l,r):
        #   y[i] = A

        return y

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
        return self.get_heigth()*self.get_width()
    def get_lambda(self):
        return self.x[self.idx]

        return np.argmin(dist_list)
def find_beter(v,candidat_list):
    dist_list = [distance(v,x.get_vector()) for x in candidat_list]
    return np.argmin(dist_list)

def diffusion_smooth(x,y,N=20,D=0.40):
    for i in range(N): 
        y[1:-1]+=D*(y[2:]+y[:-2]-2*y[1:-1])
    return y
    
class GetPeaks():
    height = 0
    distance =1
    peaks= None
    windows_on = False
    window_size=2
    patch_iteration_number=1
    decimate=20
    smooth_iteration = 20
    kernel_size = 3
    kernel_size2 = 2

    hight = False

    def __init__(self, n_peaks=5,peack=[]):
        self.__n_peaks = n_peaks
        self.peaks=peack

    def set_params(self,kernel_size):
        self.kernel_size=kernel_size

    def execute(self,xx, signal0):
        x,dy = self.signal_filter(xx,signal0)
        x,dy = self.diff_smooth(x,dy)
        candidat_list = self.get_candidat(x,dy)
        candidat_list = self.detect_bad_candidat(candidat_list)

        if self.hight:
            candidat_list = self.hight_accurance(candidat_list,xx,signal0)
        else:
            candidat_list =[Candidat(c.idx*self.decimate,xx,signal0) for c in candidat_list ]
        if len(self.peaks)>0 and self.windows_on:
            candidat_list= self.window_mask(candidat_list)
        peaks=np.zeros(self.__n_peaks)

        for  i,p in enumerate(candidat_list[:self.__n_peaks]):
            if p:
                peaks[i]=p.idx
        if self.windows_on and len(self.peaks)==0:
            peaks.sort()
            self.peaks=peaks[:self.__n_peaks]
        peaks.sort()
        return peaks



    def hight_accurance(self,peaks,x,y):
        new_list = []
        for c in peaks:
            new_list.append(self.hight_accurance_candidat(c,x,y))
        return new_list
    def hight_accurance_candidat(self,c,x0,y0):
        l_idx = (c.idx-5)*self.decimate
        r_idx = (c.idx+6)*self.decimate
        y = y0[l_idx:r_idx]
        x = x0[l_idx:r_idx]
        yg=y
        if self.patch_iteration_number >0:
            yg = gaussian_filter1d(y,10)
            yg = self.patch(x,yg)
            yg = gaussian_filter1d(y,50)
        dy = -np.diff(yg)
        newi = find_peaks(dy)[0]
        if len(newi)>0:
            new_c =Candidat(newi[0]+l_idx,x0,y0)
        else:
            new_c= Candidat(c.idx*self.decimate,x0,y0)
        return new_c


    def diff_smooth(self,x,y):
        dy = -diff(x,y)
        dy = gaussian_filter1d(dy,self.kernel_size2)
        return x,dy

    def signal_filter(self,x,y):
        #dy = gaussian_filter1d(y,3)
#        dy = gaussian_filter1d(y,self.kernel_size)
        #ks =self.kernel_size*2
    
        #dy = savgol_filter(y[::self.decimate],self.kernel_size,2)
        xx = x[::self.decimate]
        dy  = y[::self.decimate].copy()
        #dy = gaussian_filter1d(dy,self.kernel_size)
        dy = diffusion_smooth(xx,dy,N=self.smooth_iteration)
        #ks =self.kernel_size*2
        return xx,dy

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

    def get_candidat(self,x,dy,squared=False):
        peack = find_peaks(dy,height=self.height,distance=self.distance)
        candidat_list = []
        l_last = 0
        r_last=0
        for i in peack[0]:
            if True:
                c =Candidat(i,x,dy)
 #               l,r = c.get_width_index()
 ##               if l==l_last and r==r_last:
 #                   if c.get_heigth()>candidat_list[-1].get_heigth():
##                       candidat_list[-1]=c
# #               else:
                candidat_list.append(c)
 #                   l_last,r_last = l,r
        candidat_list.sort(key=lambda x: -x.get_weight())
        return candidat_list





    def detect_bad_candidat(self,cl):
        if self.__n_peaks<len(cl):
            ln = (min(len(cl)-self.__n_peaks,3))
            nois_w = cl[self.__n_peaks].get_weight()

            ret = []
            cl2 = cl[:self.__n_peaks][::-1]

            for x in cl2:
                if x.get_weight()>nois_w*1.2:
                    ret.append(x)
                else:
                    nois_w=x.get_weight()
            return ret[::-1]
        else:
            return cl
    def patch(self,x,y):
        pathced_y = y
        x0,dy = self.diff_smooth(x,y)
        candidat_list = self.get_candidat(x0,dy)
        for c in filter(lambda x:x.get_width()<2, candidat_list):
            pathced_y=c.patch_signal(pathced_y)
        return pathced_y



    def execute0(self,xx, signal0):
        x,dy = self.signal_filter(xx,signal0)
        #dy  = detrend(dy)
        for i in range(self.patch_iteration_number):
            dy = self.patch(x,dy)
            x,dy = self.signal_filter(x,dy)
        candidat_list = self.get_candidat(x,dy)

        candidat_list = self.detect_bad_candidat(candidat_list)

        if len(self.peaks)>0 and self.windows_on==True:
            candidat_list= self.window_mask(candidat_list)
        peaks=np.zeros(self.__n_peaks)
        for i,p in enumerate(candidat_list[:self.__n_peaks]):
            if p:
                peaks[i]=p.idx
        if self.windows_on and len(self.peaks)==0:
            peaks.sort()
            self.peaks=peaks[:self.__n_peaks]
        else:
            peaks.sort()


   #     peack = [p.idx for p in candidat_list[:self.__n_peaks]]
   #     peack.sort()
   #     peack += [0 for x in range(self.__n_peaks-len(peack))]
        return peaks
    def plot(self,x,y):
        fig,ax = plt.subplots(3,sharex=True,figsize=(14,6))
        x2, y2 = self.signal_filter(x,y)
        ax[0].plot(x,y,markersize=1,marker='o',ls='')
        ax[1].plot(x2,y2)
        ax[0].set_title('Исходный сигнал')
        #ax[2].plot(x1,gaussian_filter1d(y2,5))
        dx,dy = self.diff_smooth(x2,y2)
        candidata_list =self.get_candidat(dx,dy)
        ax[2].plot(dx,dy)
        peack = self.execute(x,y)
        good_candidata  = self.detect_bad_candidat(candidata_list)
        ax[1].set_title('Сглаженный сигнал')
        for c  in candidata_list:
            px = c.get_lambda()
            py = dy[c.idx]
            ax[2].plot([px,px],[py,py],ls='',marker='o',color="k")
            ax[2].text(px,py*1.01,'{:1.2f}'.format(c.get_weight()))
        for c in peack:
            px = x[int(c)]
            ax[2].plot([px,px],[min(dy),max(dy)],ls='--')
            ax[1].plot([px,px],[min(y),max(y)],ls='--')
            ax[0].plot([px,px],[min(y),max(y)],ls='--')
        ax[2].set_title('Производная со сглаживанием')
        ax[2].set_xlabel('$\lambda$,нм')
        return fig,ax


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
