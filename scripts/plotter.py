from math import cosh
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bubble_collisions import simulation
import sys
from matplotlib import cm
import scipy.interpolate as intp

inFile = sys.argv[1]
data = simulation.readFromFile(inFile)

Ndata = np.array([d[0] for d in data])
x = np.linspace(0.01, 1.0, 400)
N_list = np.linspace(0.0,1.0,400)

phi=[]
alphaa = []
aa =[]
pix = []

for i in range(len(N_list)):
    Y=simulation.valsOnGrid(
    N_list[i]*np.ones_like(x),x, data, [d[0] for d in data], False)
    
    phi.append(Y[:,0,0])
    pix.append(Y[:,0,1])
    alphaa.append(Y[:,0,-2])
    aa.append(Y[:,0,-1])
    #ba.append(Y[:,0,-3])

phiintp = intp.interp2d(x, N_list, phi, kind='cubic')
phiintpx = phiintp(x, N_list, dx=1, dy=0)
aintp = intp.interp2d(x, N_list, aa, kind='cubic')
aintpx = aintp(x, N_list, dx=1, dy=0)
aintpxt = aintp(x, N_list, dx=1, dy=1)
aintpt = aintp(x, N_list, dx=0, dy=1) 
alphaintp = intp.interp2d(x, N_list, alphaa, kind='cubic')
alphaintpx = alphaintp(x, N_list, dx=1, dy=0)

X = np.array(x)
Phi = np.array(phi)
AA = np.array(aa)
Alpha = np.array(alphaa)
Pix = np.array(pix)
alphconstr = alphaintpx + Alpha*aintpx/AA + (4*np.pi*Alpha**2*Pix*phiintpx-Alpha*aintpxt)/aintpt

plt.contourf(x,N_list,phi,20,cmap='RdGy')
plt.colorbar()
plt.xscale('log')
plt.savefig("phi_contour.png")
plt.figure()
plt.contourf(x,N_list,alphaa,20,cmap='RdGy')
plt.colorbar()
plt.savefig("alpha_contour.png")
plt.figure()
plt.contourf(x,N_list,aa,20,cmap='RdGy')
plt.colorbar()
plt.savefig("a_contour.png")
plt.figure()
#plt.contourf(x,N_list,ba,20,cmap='RdGy')
#plt.colorbar()
#plt.savefig("b_contour.png")

phi = [abs(phi[i]) for i in range(len(phi))]
Phi = np.array(phi)
fig, ax = plt.subplots()
plt.xscale('log')
lev_exp = np.arange(-10,np.ceil(np.log10(Phi.max())+1))
levs = np.power(10, lev_exp)
cs = ax.contourf(x, N_list, Phi, levs, norm=cm.colors.LogNorm())
cbar = fig.colorbar(cs)
fig.savefig("phi_contour_new.png")

plt.figure()
for i in range(6):
    plt.plot(x,phi[len(N_list)/6*i],label='phi @ t={:03f}'.format(N_list[len(N_list)/6*i]))
plt.plot(x,phi[-4],label='phi @ t={:03f}'.format(N_list[-4]))
plt.xlabel("radius")
plt.xscale('log')
plt.title("phi")
plt.legend()
plt.savefig("phi_time_x.png")

plt.figure()
for i in range(6):
    plt.plot(x,aa[len(N_list)/6*i]/cosh(N_list[len(N_list)/6*i]),label='a @ t={:03f}'.format(N_list[len(N_list)/6*i]))
plt.plot(x,aa[-4]/cosh(N_list[-4]),label='a @ t={:03f}'.format(N_list[-4]))
plt.xlabel("radius")
plt.title("a")
plt.legend()
plt.savefig("a_time_x.png")

plt.figure()
plt.plot(x,phi[0],label='phi @ t={:03f}'.format(N_list[0]))
plt.xlabel("radius")
plt.title("phi")
plt.legend()
plt.savefig("phi_0_x.png")

plt.figure()
for i in range(6):
    plt.plot(X,abs(alphconstr[len(N_list)/6*i]),label='t={:03f}'.format(N_list[len(N_list)/6*i]))
plt.plot(x,abs(alphconstr[-4]),label='t={:03f}'.format(N_list[-4]))
plt.xlabel("radius")
plt.xscale('log')
plt.yscale('log')
plt.title("alpha constraint")
plt.legend()
plt.savefig("alphconstr_time_x.png")

