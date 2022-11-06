
import numpy as np
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
            l,m,r= self.get_width_index()
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


