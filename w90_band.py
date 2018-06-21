#%%time
import numpy as np
import matplotlib.pyplot as plt
import pymatgen as pmg
from pythtb import *
#import scipy
#import scipy.integrate as integrate
from multiprocessing import Pool

pnum=2 #define the number of parapell cores

yrange=[-8,-1] # define the ylimts



# get the real space lacttice vectors (a1,a2,a3), recipral vectors(b1,b2,b3), atom sites

def get_struc():
    lattice=pmg.Structure.from_file('CONTCAR')
    atom_sites=lattice.cart_coords
    A=lattice.lattice.matrix
    B=lattice.lattice.reciprocal_lattice.matrix
    return A[0],A[1],A[2],B[0],B[1],B[2],atom_sites

(a1,a2,a3,b1,b2,b3,atoms)=get_struc()

#a1=np.array([ 3.901,  0.   ,  0.   ])
#a2=np.array([ 0.   ,  3.901,  0.   ])
#a3=np.array([  0.    ,   0.   ,  17.94209662])
#b1=np.array([ 1.61066017,  0.  ,  0. ])
#b2=np.array([ 0. ,  1.61066017,  0.   ])
#b3=np.array([ 0.        ,  0.        ,  0.35019237])
#atoms=np.array([[  0.        ,   0.        ,   8.97104831],
#               [  1.9505    ,   1.9505    ,   8.97104831],
#                [  0.        ,   1.9505    ,   7.56713751],
#                [  1.9505    ,   0.        ,  10.37495911]])

ispin=2 # spin mode, 1 no soc, 2. soc
atlist=[atoms[1],atoms[2],atoms[0]] # the corresponding MLWF position
atnum=[3,3,5] # the number of orbitals of each site

def get_atominfo(atlist,atnum,ispin=ispin):
    atomslist=atlist*ispin
    atnums=atnum*ispin
    total=np.array(atnums).sum()
    listat=np.arange(total)
    
    atomsnumb=[]
    t=0
    for i in range(len(atnums)):      
        t0=t
        c=atnums[i]
        t=t+c
        
        atomsnumb.append(listat[t0:t])
    
    return atomslist,atomsnumb

(atomslist,atomsnumb)=get_atominfo(atlist,atnum,ispin=2)

#atomslist=[atoms[1],atoms[2],atoms[0],atoms[1],atoms[2],atoms[0]]
#atoms1=[0,1,2]
#atoms2=[3,4,5]
#atoms3=[6,7,8,9,10]
#atoms4=[11,12,13]
#atoms5=[14,15,16]
#atoms6=[17,18,19,20,21]
#atomsnumb=[atoms1,atoms2,atoms3,atoms4,atoms5,atoms6]

def get_atom(num):
    for ki in range(len(atomsnumb)):
        if num in atomsnumb[ki]:
            break
    return atomslist[ki]

# get the Kpath from pythtb
G=np.array([0.0,0.0,0.0])
M=np.array([0.5,0.5,0.0])
K=np.array([1/3.,2/3.,0.0])
K1=np.array([-1/3.,-2/3.,0.0])
k=[G,M,K,G]
lat=[a1,a2,a3]
k_num=50
(k_vec,k_dist,k_node)=tb_model(3,3,lat).k_path(k, k_num,report=False)


# process wannier90_hr.dat, get num_wann, nrkpts
def get_w90():
    file=open('wannier90_hr.dat','r')
    w90=file.readlines() 
    num_wann=int(w90[1].strip())
    nrkps=int(w90[2].strip())
    nrkps_num=int(np.ceil(nrkps/15))
    ndeges=[]
    for i in range(nrkps_num):
        A=w90[3+i].split()
        #ndeges.extend(A)
        for item in A:
            ndeges.append(int(item))
    with open('hr_mn.dat','w+') as hr:
        for line in w90[3+nrkps_num:]:
            hr.write(line)
    
    return num_wann,nrkps,ndeges


(num_wann,nrkps,ndeges)=get_w90()
hr_data=np.loadtxt('hr_mn.dat')
RRs=hr_data[:,0:3]
RRi=np.unique(RRs, axis=0)

hops=hr_data[:,5:]
#hnum=len(hr_data)
hopping=hr_data[:,5]+1j*hr_data[:,6]
hopH=hopping.reshape((nrkps,num_wann,num_wann))


# two Formalisms: http://physics.rutgers.edu/pythtb/_downloads/pythtb-formalism.pdf
'''
def hmnR_func_I(nrkp,kk,RR):
    hmnR=np.zeros((num_wann,num_wann),dtype='complex')
    
    k=0
    for i in range(num_wann):
        for ii in range(num_wann):
            atomA=get_atom(i)
            atomB=get_atom(ii)
            datom=atomB-atomA
            katom=kk[0]*datom[0]+kk[1]*datom[1]+kk[2]*datom[2]
            hop=hr_data[nrkp*num_wann*num_wann+k,5]+hr_data[nrkp*num_wann*num_wann+k,6]*1j
            hmnR[ii,i]=hop*np.exp(1j*katom)
            k=k+1
            
    return hmnR


def hmnR_func_II(nrkp,kk,RR):

    hmnR=np.zeros((num_wann,num_wann),dtype='complex')
    
    k=0
    for i in range(num_wann):
        for ii in range(num_wann):
            hmnR[ii,i]=hr_data[nrkp*num_wann*num_wann+k,5]+hr_data[nrkp*num_wann*num_wann+k,6]*1j
            k=k+1         
    return hmnR



#evals=np.zeros((len(k_vec),num_wann),dtype='float')
def get_eval_I(kv):
    print(kv)
    hm=np.zeros((num_wann,num_wann),dtype='complex')
    kk=k_vec[kv,0]*b1+k_vec[kv,1]*b2+k_vec[kv,2]*b3

    for nrkp in range(nrkps):
        #RR=hr_data[nrkp*num_wann*num_wann,0]*a1+hr_data[nrkp*num_wann*num_wann,1]*a2+hr_data[nrkp*num_wann*num_wann,2]*a3
        RR=RRi[nrkp,0]*a1+RRi[nrkp,1]*a2+RRi[nrkp,2]*a3
        kR=kk[0]*RR[0]+kk[1]*RR[1]+kk[2]*RR[2]
        hmnR=hmnR_func_I(nrkp,kk,RR)
        
        hm=hm+hmnR*np.exp(1j*kR)/ndeges[nrkp]
    return np.linalg.eig(hm)[0]

def get_eval_II(kv):
    print(kv)
    hm=np.zeros((num_wann,num_wann),dtype='complex')
    kk=k_vec[kv,0]*b1+k_vec[kv,1]*b2+k_vec[kv,2]*b3
    for nrkp in range(nrkps):
    
        RR=RRi[nrkp,0]*a1+RRi[nrkp,1]*a2+RRi[nrkp,2]*a3
        #RR=hr_data[nrkp*num_wann*num_wann,0]*a1+hr_data[nrkp*num_wann*num_wann,1]*a2+hr_data[nrkp*num_wann*num_wann,2]*a3
        kR=kk[0]*RR[0]+kk[1]*RR[1]+kk[2]*RR[2]
        hmnR=hmnR_func_II(nrkp,kk,RR)
        
        hm=hm+hmnR*np.exp(1j*kR)/ndeges[nrkp]
    return np.linalg.eig(hm)[0]

'''


def hmnR_func_I(kk):
    hatom=np.zeros((num_wann,num_wann),dtype='complex')
    
    for i in range(num_wann):
        for ii in range(num_wann):
            atomA=get_atom(i)
            atomB=get_atom(ii)
            datom=atomB-atomA
            katom=kk[0]*datom[0]+kk[1]*datom[1]+kk[2]*datom[2]
            #hop=hr_data[nrkp*num_wann*num_wann+k,5]+hr_data[nrkp*num_wann*num_wann+k,6]*1j
            hatom[i,ii]=np.exp(1j*katom)
            
    return hatom


def get_eval_I(kv):
    print(kv)
    hm=np.zeros((num_wann,num_wann),dtype='complex')
    kk=k_vec[kv,0]*b1+k_vec[kv,1]*b2+k_vec[kv,2]*b3
    Hatom=hmnR_func_I(kk)

    for nrkp in range(nrkps):
        #RR=hr_data[nrkp*num_wann*num_wann,0]*a1+hr_data[nrkp*num_wann*num_wann,1]*a2+hr_data[nrkp*num_wann*num_wann,2]*a3
        RR=RRi[nrkp,0]*a1+RRi[nrkp,1]*a2+RRi[nrkp,2]*a3
        kR=kk[0]*RR[0]+kk[1]*RR[1]+kk[2]*RR[2]
        #hmnR=np.multiply(hA[nrkp],Hd)
        hmnR=hopH[nrkp]*Hatom
        
        hm=hm+hmnR*np.exp(1j*kR)/ndeges[nrkp]
    return np.linalg.eig(hm)[0]


def get_eval_II(kv):
    print(kv)
    hm=np.zeros((num_wann,num_wann),dtype='complex')
    kk=k_vec[kv,0]*b1+k_vec[kv,1]*b2+k_vec[kv,2]*b3
    for nrkp in range(nrkps):
    
        RR=RRi[nrkp,0]*a1+RRi[nrkp,1]*a2+RRi[nrkp,2]*a3
        #RR=hr_data[nrkp*num_wann*num_wann,0]*a1+hr_data[nrkp*num_wann*num_wann,1]*a2+hr_data[nrkp*num_wann*num_wann,2]*a3
        kR=kk[0]*RR[0]+kk[1]*RR[1]+kk[2]*RR[2]
        hmnR=hopH[nrkp]
        
        hm=hm+hmnR*np.exp(1j*kR)/ndeges[nrkp]
    return np.linalg.eig(hm)[0]

print('===========Begin bands calculation I=========')
if __name__ == '__main__':
    with Pool(pnum) as p:
        evalsI=p.map(get_eval_I,range(len(k_vec)))
evalsI=np.real(evalsI)
for i in range(len(evalsI)):
    evalsI[i].sort()
np.savetxt('bands_w90_I.dat',evalsI)

print('===========End bands calculation I=========')

print('===========Begin bands calculation II=========')
if __name__ == '__main__':
    with Pool(pnum) as p:
        evalsII=p.map(get_eval_II,range(len(k_vec)))
evalsII=np.real(evalsII)
for i in range(len(evalsII)):
    evalsII[i].sort()
np.savetxt('bands_w90_II.dat',evalsII)

print('===========End bands calculation II=========')


fig, ax1 = plt.subplots()
ax1.plot(k_dist,evalsI,c='r',marker='.')
ax1.set_xlim(k_dist.min(),k_dist.max())
#plt.hlines(-2.1801,0,1)

ax2 = ax1.twiny() 
ax2.plot(k_dist,evalsII,c='b')
ax2.set_xlim(k_dist.min(),k_dist.max())
ax2.set_ylim(yrange)
plt.savefig('plot_w90.pdf')

plt.show()
