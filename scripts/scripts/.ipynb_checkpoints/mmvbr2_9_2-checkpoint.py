{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cd7174ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "# -*- coding: utf-8 -*-\n",
    "\"\"\"\n",
    "Created on Thu Jul 29 14:13:33 2021\n",
    "\n",
    "@author: workstation2\n",
    "\"\"\"\n",
    "\n",
    "import sys\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from matplotlib import pyplot as plt\n",
    "from scipy.signal import cwt, morlet2, find_peaks\n",
    "import sys\n",
    "sys.path.append('../')\n",
    "from scipy.ndimage import gaussian_filter1d,uniform_filter\n",
    "from .che_signal import Signal\n",
    "\n",
    "from scipy.signal import savgol_filter\n",
    "\n",
    "from collections import defaultdict\n",
    "\n",
    "\n",
    "from .base_method import BaseMethod\n",
    "def distance(x,y):\n",
    "    s  = (x-y)**2\n",
    "    return s.sum()\n",
    "\n",
    "\n",
    "def get_kernel(size,lam=1,sig=8):\n",
    "    y = np.zeros(size)\n",
    "    for i in range(size):\n",
    "        x=(i/size-0.5)\n",
    "        y[i]=np.sin(lam*x*np.pi)*np.exp(-x**2*sig)\n",
    "    y/=size\n",
    "    return y\n",
    "def process(signal, filters):\n",
    "    accum = np.zeros_like(signal)\n",
    "    for k in filters:\n",
    "        signal2 = np.convolve(signal,k,'same')\n",
    "        np.maximum(accum, signal2, accum)\n",
    "    return accum\n",
    "\n",
    "def diff(x,y):\n",
    "    z = np.zeros_like(x)\n",
    "    for i in range(0,len(x)-1):\n",
    "        z[i] = (y[i+1]-y[i])/(x[i+1]-x[i])\n",
    "    return z\n",
    "\n",
    "\n",
    "class Candidat:\n",
    "    def __init__(self,idx,x,y):\n",
    "        self.idx = idx\n",
    "        self.x = x\n",
    "        self.y = y\n",
    "    def get_width_index(self):\n",
    "        i=self.idx\n",
    "        for i in range(self.idx,len(self.y)-3,1):\n",
    "            if (self.y[i]<self.y[i+1] or self.y[i]<0):\n",
    "                break\n",
    "        r=i\n",
    "\n",
    "        for i in range(self.idx,1,-1):\n",
    "            if (self.y[i+1]<0):\n",
    "                break\n",
    "        m=i\n",
    "        for i in range(m,1,-1):\n",
    "            if (self.y[i-1]*self.y[i]<0):\n",
    "                break\n",
    "        l=i\n",
    "        self.M = m\n",
    "        return l,m,r\n",
    "    def get_width(self):\n",
    "        l,m,r= self.get_width_index()\n",
    "        return self.x[r]-self.x[l]\n",
    "    def half_height(self):\n",
    "        m = self.get_m()\n",
    "        y =  self.signal.get_gaus_data()\n",
    "        hh = (y[m]+y[self.idx])/2\n",
    "        return hh\n",
    "    def get_assimetry(self):\n",
    "        if hasattr(self,'assimetry'):\n",
    "            return self.assimetry\n",
    "        else:\n",
    "            x=self.x\n",
    "            l,m,r= self.get_width_index()\n",
    "            m= self.idx\n",
    "            self.assimetry= -(x[r]-x[m])/(x[l]-x[m])\n",
    "            return self.assimetry\n",
    "    def get_vector(self):\n",
    "        return np.array([\n",
    "            self.x[self.idx],\n",
    "            self.get_width(),\n",
    "            self.y[y],\n",
    "\n",
    "        ])\n",
    "    def get_heigth(self):\n",
    "        return self.y[self.idx]\n",
    "    def get_weight(self):\n",
    "        #l,r= self.get_width_index()\n",
    "        return self.get_heigth()\n",
    "    def get_lambda(self):\n",
    "        return self.x[self.idx]\n",
    "\n",
    "        return np.argmin(dist_list)\n",
    "def find_beter(v,candidat_list):\n",
    "    dist_list = [distance(v,x.get_vector()) for x in candidat_list]\n",
    "    return np.argmin(dist_list)\n",
    "\n",
    "\n",
    "class GetPeaks(BaseMethod):\n",
    "    height = 0\n",
    "    distance =100\n",
    "    kernel_size = 350\n",
    "    kernel_size2 = 50\n",
    "    peaks= None\n",
    "    windows_on = True\n",
    "    window_size=2\n",
    "    \n",
    "    shift = (350)//2\n",
    "\n",
    "    def __init__(self, n_peaks=5,peack=[]):\n",
    "        self.__n_peaks = n_peaks\n",
    "        self.peaks=peack\n",
    "\n",
    "    def signal_filter(self,x,y):\n",
    "        #newx = np.linspace(min(x),max(x),13000)\n",
    "        #newy = np.interp(newx,x,y)\n",
    "        #filters = [get_kernel(k_size,lam=2) for k_size in [250+30*x for x in range(10)]]\n",
    "        k = get_kernel(self.kernel_size,lam=2) \n",
    "        dy = np.convolve(y,k,'same')\n",
    "        #dy = process(y,filters)\n",
    "        shift =self.shift\n",
    "        return x[shift:-shift],dy[shift:-shift]\n",
    "\n",
    "    def window_mask(self,candidat_list):\n",
    "        ret = [0]*len(self.peaks)\n",
    "        for i,pk in enumerate(self.peaks):\n",
    "            tmp = []\n",
    "            for c in candidat_list:\n",
    "                if abs(c.get_lambda()-pk)<self.window_size:\n",
    "                    tmp.append(c)\n",
    "            if len(tmp)>0:\n",
    "                tmp.sort(key=lambda x:-x.get_weight())\n",
    "                self.peaks[i]= tmp[0].get_lambda()\n",
    "                ret[i]=tmp[0]\n",
    "        return ret\n",
    "\n",
    "\n",
    "    def select_uniq(self,candidat_list):\n",
    "        candidat_categoty =defaultdict(list)\n",
    "\n",
    "        cl=[]\n",
    "        for c in  candidat_list:\n",
    "            l,m,r = c.get_width_index()\n",
    "            candidat_categoty[m].append(c)\n",
    "        for k,v in candidat_categoty.items():\n",
    "            v=sorted(v,key=lambda x: -x.get_heigth())\n",
    "            cl.append(v[0])\n",
    "        return cl\n",
    "\n",
    "    def detect_bad_candidat(self,cl):\n",
    "        if self.__n_peaks<len(cl):\n",
    "            ln = (min(len(cl)-self.__n_peaks,3))\n",
    "            nois_w = cl[self.__n_peaks].get_weight()\n",
    "\n",
    "            ret = []\n",
    "            cl2 = cl[:self.__n_peaks][::-1]\n",
    "\n",
    "            for x in cl2:\n",
    "                if x.get_weight()>nois_w*1.1:\n",
    "                    ret.append(x)\n",
    "                else:\n",
    "                    nois_w=x.get_weight()\n",
    "            return ret[::-1]\n",
    "        else:\n",
    "            return cl\n",
    "\n",
    "    def execute(self,xx, signal):\n",
    "        x,dy = self.signal_filter(xx.tolist(),signal.tolist())\n",
    "        candidat_list = self.get_candidat(x,dy)\n",
    "        self._candidat_list = candidat_list\n",
    "        #candidat_list = self.select_uniq(candidat_list)\n",
    "        candidat_list = self.detect_bad_candidat(candidat_list)\n",
    "\n",
    "        if len(self.peaks)>0:\n",
    "            candidat_list= self.window_mask(candidat_list)\n",
    "        peaks=[]\n",
    "        for  p in candidat_list[:self.__n_peaks]:\n",
    "            if p:\n",
    "                peaks.append(p.get_lambda())\n",
    "            else:\n",
    "                peaks.append(0)\n",
    "        if self.windows_on and len(self.peaks)==0:\n",
    "            peaks.sort()\n",
    "            self.peaks=peaks[:self.__n_peaks]\n",
    "        peack=[]\n",
    "        for p in candidat_list[:self.__n_peaks]:\n",
    "            if p:\n",
    "                peack.append(p.idx+self.shift)\n",
    "            else:\n",
    "                peack.append(0)\n",
    "        peack.sort()\n",
    "\n",
    "\n",
    "\n",
    "   #     peack = [p.idx for p in candidat_list[:self.__n_peaks]]\n",
    "   #     peack.sort()\n",
    "   #     peack += [0 for x in range(self.__n_peaks-len(peack))]\n",
    "        return peack\n",
    "    def diff_smooth(self,x,y):\n",
    "        dy = -diff(x,y)\n",
    "        if self.kernel_size2>0:\n",
    "            dy= gaussian_filter1d(dy,self.kernel_size2)\n",
    "        return x,  dy\n",
    "    def get_candidat(self,x,dy):\n",
    "\n",
    "        peack = find_peaks(dy,height=self.height,distance=self.distance)\n",
    "        candidat_list = []\n",
    "        l_last = 0\n",
    "        r_last=0\n",
    "        for i in peack[0]:\n",
    "            if dy[i]>0:\n",
    "                c =Candidat(i,x,dy)\n",
    "                candidat_list.append(c)\n",
    "        candidat_list.sort(key=lambda x: -x.get_weight())\n",
    "        return candidat_list\n",
    "\n",
    "    def plot(self,x,y,fig=None,ax=None):\n",
    "        if not fig:\n",
    "            fig,ax = plt.subplots(2,sharex=True,figsize=(14,8))\n",
    "        x2, y2 = self.signal_filter(x,y)\n",
    "        ax[0].plot(x,y,markersize=1,marker='o',ls='')\n",
    "        ax[1].plot(x2,y2)\n",
    "        ax[0].set_title('0. Исходный сигнал')\n",
    "        #ax[2].plot(x1,gaussian_filter1d(y2,5))\n",
    "        candidata_list =self.get_candidat(x2,y2)\n",
    "        ax[1].set_title('1. После фильра')\n",
    "        for c  in candidata_list:\n",
    "            px = x2[c.idx]\n",
    "            py = y2[c.idx]\n",
    "            ax[1].plot([px,px],[py,py],ls='',marker='o',color=\"k\")\n",
    "            ax[1].text(px,py*1.01,'{:1.2f}'.format(c.get_weight()))\n",
    "\n",
    "        peack = self.execute(x,y)\n",
    "        good_candidata  = self.detect_bad_candidat(candidata_list)\n",
    "        for c in peack:\n",
    "            px = x[int(c)]\n",
    "            ax[1].axvline(px,ls='--')\n",
    "            ax[0].axvline(px,ls='--')\n",
    "        ax[1].set_xlabel('$\\lambda$,нм')\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    filepath = \"spectrum_2021-07-13_15-04-25.832.csv\"\n",
    "    data = pd.read_csv(filepath,\n",
    "                       delimiter=';',\n",
    "                       encoding=\"ISO-8859-1\",\n",
    "                       )\n",
    "    signal = data.iloc[:, -1]\n",
    "    x = data.iloc[:, 0]\n",
    "\n",
    "    command = GetPeaks(5)\n",
    "    retval = command.execute(x,signal)\n",
    "\n",
    "    plt.figure(figsize=(12, 12))\n",
    "    plt.plot(x,signal)\n",
    "    plt.plot(retval, signal[retval.astype(int)], 'o', markersize=14)\n",
    "    plt.grid()\n",
    "    plt.show()\n",
    "    print(retval)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}