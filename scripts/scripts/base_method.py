from abc import ABC, abstractmethod

from scipy.signal import  find_peaks
from methodsapp.scripts.candidat import Candidat

class BaseMethod(ABC):
    peak_typ="index"
    def set_params(self,params):
        for k,v in params.items():
            setattr(self,k,v)

    @abstractmethod
    def execute(self) -> None:
        pass
 
class BaseMethod2(ABC):
    peak_typ="lambda"
    height = 0
    distance =100
    peaks= None
    windows_on = True
    window_size=2


    def set_params(self,params):
        for k,v in params.items():
            setattr(self,k,v)

    @abstractmethod
    def execute(self) -> None:
        pass

    def execute(self,xx, signal):
        x,dy = self.signal_filter(xx.to_numpy(),signal.to_numpy())
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

    def get_candidat(self,x,dy):
        peack = find_peaks(dy,height=self.height,distance=self.distance)
        candidat_list= []
        for i in peack[0]:
            if dy[i]>0:
                c =Candidat(i,x,dy)
                candidat_list.append(c)
        candidat_list.sort(key=lambda x: -x.get_weight())
        return candidat_list

