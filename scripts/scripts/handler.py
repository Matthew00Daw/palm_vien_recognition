import numpy as np
from scipy.signal import detrend
from collections import defaultdict 
def mean_redus(x,y,n=20):
    N = len(x)//n
    x1 = np.zeros(N)
    y1 = np.zeros(N)
    for i in range(N):
        x1[i]=x[i*n:(i+1)*n].mean()
        y1[i]=y[i*n:(i+1)*n].mean()
    return x1,y1

def diff(x,y):
    z = np.zeros_like(x)
    for i in range(0,len(x)-1):
        z[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
    return z

def window_mask(peaks,candidat_list,window_size):
    ret = [0]*len(peaks)
    for i,pk in enumerate(peaks):
        tmp = []
        for c in candidat_list:
            if abs(c.get_lambda()-pk)<window_size:
                tmp.append(c)
        if len(tmp)>0:
            tmp.sort(key=lambda x:-x.get_weight())
            peaks[i]= tmp[0].get_lambda()
            ret[i]=tmp[0]
    return ret

def select_uniq(candidat_list):
    candidat_categoty =defaultdict(list)
    cl=[]
    for c in  candidat_list:
        l,m,r = c.get_width_index()
        candidat_categoty[m].append(c)
    for k,v in candidat_categoty.items():
        v=sorted(v,key=lambda x: -x.get_heigth())
        cl.append(v[0])
    return cl

def detect_bad_candidat(n,cl):
    if n<len(cl):
        ln = (min(len(cl)-n,3))
        nois_w = cl[n].get_weight()

        ret = []
        cl2 = cl[:n][::-1]

        for x in cl2:
            if x.get_weight()>nois_w*1.1:
                ret.append(x)
            else:
                nois_w=x.get_weight()
        return ret[::-1]
    else:
        return cl


def trend_correct(x,y,a=0.00005):
    y0=detrend(y)
    x1=x-x.min()
    ymin = y.min()
    return x,(y0-ymin)*(1+x1*a)+ymin



