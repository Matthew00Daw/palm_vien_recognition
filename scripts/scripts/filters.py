import matplotlib
import pandas as pd
import glob, os
import csv
import numpy as np
import math
def mixer_shmooth(x,wl, aaa=0.5*10**8):

    length=len(wl) #определение длины массива

# constants
    aaaa=np.zeros(length, dtype=float)
    bbbb=np.zeros(length, dtype=float)
    gggg=np.zeros(length, dtype=float)
    delta=np.zeros(length, dtype=float)
    TT=np.zeros(length, dtype=float)
# алгоритм
#    kk=1 # канал со спектром
    aa=1+aaa
    i=0
# i=3, length-2
# расчет для i=1
    aaaa[0] = 2*aaa/aa
    bbbb[0]=-aaa/aa
    gggg[0]=wl[0]/aa
#i=2
    delta[1]=1+aaa*(5-2*aaaa[0])
    aaaa[1]=2*aaa*(2+bbbb[0])/delta[1]
    bbbb[1]=-aaa/delta[1]
    gggg[1]=(wl[1]+2*aaa*gggg[0])/delta[1]

    for i in range(2,length):
        if i>1 and i<length-2:
            delta[i]=1.0+aaa*(6.0-4*aaaa[i-1]+aaaa[i-1]*aaaa[i-2]+bbbb[i-2])
            aaaa[i]=aaa*(4+4*bbbb[i-1]-aaaa[i-2]*bbbb[i-1])/delta[i]
            bbbb[i]=-aaa/delta[i]
            gggg[i]=(wl[i]-aaa*(gggg[i-2]-4*gggg[i-1]+gggg[i-1]*aaaa[i-2]))/delta[i]
        elif i==length-2:
            delta[length-2]=1+aaa*(5-4*aaaa[length-3]+aaaa[length-3]*aaaa[length-4]+bbbb[length-4])
            aaaa[length-2]=aaa*(2+4*bbbb[length-3]-aaaa[length-4]*bbbb[length-3])/delta[length-2]
            bbbb[length-2]=0
            gggg[length-2]=(wl[length-2]-aaa*(gggg[length-4]-4*gggg[length-3]+gggg[length-3]*aaaa[length-4]))/delta[length-2]
        elif i==length-1:
            delta[length-1]=1+aaa*(1-2*aaaa[length-2]+aaaa[length-2]*aaaa[length-3]+bbbb[length-3])
            aaaa[length-1]=0
            bbbb[length-1]=0
            gggg[length-1]=(wl[length-1]-aaa*(gggg[length-3]-2*gggg[length-2]+gggg[length-2]*aaaa[length-3]))/delta[length-1]
    TT[length-1]=gggg[length-1]
    TT[length-2]=aaaa[length-2]*TT[length-1]+gggg[length-2]
    for i in range(0,length-1):
        ii=length-i
        if ii>1 and ii<length-2:
            TT[ii]=aaaa[ii]*TT[ii+1]+bbbb[ii]*TT[ii+2]+gggg[ii]
    TT[1]=aaaa[1]*TT[1+1]+bbbb[1]*TT[1+2]+gggg[1]
    TT[0]=aaaa[0]*TT[1]+bbbb[0]*TT[2]+gggg[0]
    return x,TT
