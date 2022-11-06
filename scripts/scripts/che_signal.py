import os
import numpy as np
from scipy.signal import cwt, morlet2, find_peaks
from scipy.ndimage import gaussian_filter1d,uniform_filter

def diff(x,y):
    z = np.zeros_like(x)
    for i in range(1,len(x)-1):
        z[i] = (y[i-1]-y[i+1])/(x[i-1]-x[i+1])
    return z

class Signal:
    def __init__(self,data,dir_name='./',col=1,mag_thres=1e-4):
        self.pre_filter=False
        self.col=col
        self.dir_name=dir_name
        if type(data) is str:
            self.name=data
            self.full_name=os.path.join(dir_name,data)
        else:
            self.data=data
    def plot(self,p=None):
        for pp in p:
            plt.plot([pp,pp],[-60,0])
        x,y = self.get_data()
        plt.plot(x,y)
    def get_data(self,save=False):
        if hasattr(self,'data'): #and  isinstance(self.data, np.ndarray):
            d = self.data
        else:
            d =  np.genfromtxt(self.full_name,delimiter=";",skip_header=1,encoding = "ISO-8859-1").T
            if self.pre_filter:
                for x in range(len(d)-1):
                    d[x+1]=gaussian_filter1d(d[x+1],3)
                    z = np.polyfit(d[0], d[x+1], 2)
                    p = np.poly1d(z)

                    #d[x+1]-=p(d[0])
                    #d[x+1]-=0.1*uniform_filter(d[x+1],1000)
                    d[x+1]-=np.min(d[x+1])
                    d[x+1]/=np.max(d[x+1])
            if save:
                self.data=d
        return d
    def get_time(self):
        return d
    def get_time(self):
        st = re.search("spectrum_([\d\-\._]+).csv",self.name)
        if st:
            s= st.group(1)
            return datetime.strptime(s,'%Y-%m-%d_%H-%M-%S.%f')
    def get_gaus_data(self):
        if hasattr(self,'gaus'):
            return self.gaus
        else:
            d = self.get_data()
            self.gaus = gaussian_filter1d(d[self.col],100)
            return self.gaus

    def get_d_gaus_data(self):
        if hasattr(self,'d_gaus'):
            return self.d_gaus
        else:
            d = self.get_data()
            x = d[0]
            y = d[1]
            gaus = self.get_gaus_data()
            dy = -diff(x,gaus)
            
            m_dy = gaussian_filter1d(dy,100)
            self.d_gaus = m_dy
            return  m_dy
    def get_d2_gaus_data(self):
        if hasattr(self,'d2_gaus'):
            return self.d2_gaus
        else:
            d = self.get_data()
            x = d[0]
            y = d[1]
            gaus = self.get_d_gaus_data()
            dy = diff(x,gaus)
            m_dy = gaussian_filter1d(dy,100)
            self.d2_gaus = m_dy
            return  m_dy



    def get_peack(self):
        d = self.get_data()
        gaus = self.get_gaus_data()
        return  find_peaks(gaus )[0]
    def get_d_peack(self):
        d = self.get_data()
        x = d[0]
        y = d[1]
        gaus = gaussian_filter1d(y,70)
        dy = -diff(x,gaus)
        m_dy = uniform_filter(dy,50)
        return  find_peaks(m_dy,height=0 )[0]
