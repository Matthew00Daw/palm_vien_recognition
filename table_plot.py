import numpy as np
import os
from matplotlib import pyplot as plt

def FMR_FNMR_plot(table,ls,style='-'):
    t_res=[]
    ERR=0
    ZERO_FMR=0
    for t in range(201):
        FNMR=0
        FMR=0
        FNMR_MAX=0
        FMR_MAX=0
        T=0.005*(t)
        for x in range(len(table)-1):
            for y in range(x+1,len(table[0])-1):
                if abs(table[x+1,0]-table[0,y+1])<1:
                    FNMR_MAX+=1
                    if table[x+1,y+1]<T:
                        FNMR+=1
                        if T<=0.2:
                            pass
                            print(ls[x],ls[y],x+1,y+1,table[x+1,y+1])
                          #  print(x+1,y+1,table[x+1,y+1])
                else:
                    FMR_MAX+=1
                    if table[x+1,y+1]>=T:
                        FMR+=1
                        if T>=0.10:
                            pass
                           # print(ls[x],ls[y],x+1,y+1,table[x+1,y+1])
                          #  print(ls[x],ls[y],x+1,y+1,table[x+1,y+1])
        print("---")
        print(" FNMR {} {} {} ".format(T,FNMR_MAX,FNMR))
        print(" FMR {} {} {} ".format(T,FMR_MAX,FMR))
        print("---")

        FMR_F=FMR/FMR_MAX
        FMNR_F=FNMR/FNMR_MAX
        if not(ERR) and(FMNR_F>FMR_F):
            ERR=(T,FMNR_F)
        if not(ZERO_FMR) and FMR_F<1e-6:
            ZERO_FMR=(T,FMNR_F)
        t_res.append([T,FMR_F,FMNR_F])
    t_res=np.array(t_res).T
    print('FNMR_MAX=', FNMR_MAX,'FMR_MAX=',FMR_MAX)
    plt.plot(t_res[0],t_res[1],label="FMR",ls=style)
    plt.plot(t_res[0],t_res[2],label='FNMR',ls=style)

    plt.plot([0,1],[ERR[1],ERR[1]],label='$ERR=%3.3f$'%ERR[1],ls="--")
    plt.plot([0,1],[ZERO_FMR[1],ZERO_FMR[1]],label='$ZERO_{FMNR}=%3.3f$'%ZERO_FMR[1],ls="--")
    plt.text(0,ERR[1]+0.001,"$ERR=%3.3f$"%ERR[1],size="x-small")
    plt.text(ZERO_FMR[0],ZERO_FMR[1]+0.001,"$ZERO_{FMNR}=%3.3f$"%ZERO_FMR[1],size="x-small")
    plt.legend(loc=2)
    plt.xlabel("$t$")
    plt.xlim(0,1)
    plt.ylim(0,1)
    #plt.savefig('FMR_FMNR.png')
    #plt.show()
    #fig, ax = plt.subplots()
    #ax.plot(t_res[1],t_res[2],label='FNMR',ls=style)
    #ax.loglog(t_res[1],t_res[2])
    #ax.loglog([0e-3,1],[0e-3,1])
    #ax.set_yscale("symlog")
    #ax.set_xscale("symlog")
    #ax.grid(True)
    #ax.set_xlabel("$FNR$")
    #ax.set_ylabel("$FMNR$")
    #plt.xlim(-3,1)
    #plt.ylim(-3,1)
    plt.show()

data = np.genfromtxt("table.txt")
#print(data)
#data2 = np.genfromtxt("compare_table_afine.txt")
ls = [y for y in os.listdir('./') if y.endswith('.bmp') and not y.endswith('tlp.bmp')  and not y.startswith('proc') and not y.startswith('F')  and not y.startswith('e')]
ls =  []
with open("list_full.txt") as f:
    for x in f:
        ls.append(x.replace('\n',''))
    print(ls[-1])
#ls =range(9999)
FMR_FNMR_plot(data,ls,"-")
#FMR_FNMR_plot(data2,ls,"-")
#plt.savefig('FMR_FMNR.png')
plt.show()
