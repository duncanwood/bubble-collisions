import math
from math import cosh
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bubble_collisions import simulation
import sys
from matplotlib import cm
import scipy.interpolate as intp
from scipy.optimize import fmin
import pickle

inFile = sys.argv[1]
data = simulation.readFromFile(inFile)

Ndata = np.array([d[0] for d in data])
xnum = 1000
ynum = 1000
x = np.linspace(0.01,1069., xnum)
N_list = np.linspace(0.89,10.0, ynum)

phi1 = []
phi2 = []
alphaa = []
aa = []
pix1 = []
pix2 = []

try:
    with open('two_field.info', 'r') as f:
        (t0, minr, maxr, phiF, phiT, m, c, a, g, h, j, f, tfix ) = pickle.load(f)
except:
    m = 0.01
    c = 0.002
    a = 0.00083
    g = 0.000001
    f = 1.0
    h = 1.e-7
    j = 0.000001

def V2D(x,y):
    return m**2*(x**2 + y**2) - a*(x**2 + y**2)**2 + c*(x**2 + y**2)**3 + g*np.sin(x/f) - j*np.sin(y/f) + h
def V2Dvec(x):
    return V2D(x[0],x[1])
meta_phi = np.array([0.02, 0.002])
meta_vac=V2Dvec(meta_phi)
rH = math.sqrt(3/(8*math.pi)/meta_vac)
print(meta_phi, meta_vac, rH)

for i in range(len(N_list)):
    Y=simulation.valsOnGrid(
    N_list[i]*np.ones_like(x),x, data, [d[0] for d in data], False)
    phi1.append(Y[:,0,0])
    phi2.append(Y[:,0,1])
    pix1.append(Y[:,0,2])
    pix2.append(Y[:,0,3])
    alphaa.append(Y[:,0,-2])
    aa.append(Y[:,0,-1])

phi1intp = intp.interp2d(x, N_list, phi1, kind='quintic')
phi1intpx = phi1intp(x, N_list, dx=1, dy=0)
phi2intp = intp.interp2d(x, N_list, phi2, kind='quintic')
phi2intpx = phi2intp(x, N_list, dx=1, dy=0)
aintp = intp.interp2d(x, N_list, aa, kind='quintic')
aintpx = aintp(x, N_list, dx=1, dy=0)
aintpxt = aintp(x, N_list, dx=1, dy=1)
aintpt = aintp(x, N_list, dx=0, dy=1) 
alphaintp = intp.interp2d(x, N_list, alphaa, kind='quintic')
alphaintpx = alphaintp(x, N_list, dx=1, dy=0)

X = np.array(x)
Phi1 = np.array(phi1)
Phi2 = np.array(phi2)
AA = np.array(aa)
Alpha = np.array(alphaa)
Pix1 = np.array(pix1)
Pix2 = np.array(pix2)
momemconstr = alphaintpx - Alpha*(4*np.pi*Alpha*AA*(Pix1*phi1intpx + Pix2*phi2intpx) + AA*aintpxt - aintpx*aintpt)/(aintpt*AA)

"""ax0 = AA.T[0]
afit = np.polyfit(np.sqrt(N_list),ax0,deg=1)
print(afit)
plt.figure()
plt.plot(np.sqrt(N_list),ax0)
plt.savefig("a_x0_t.png")"""

plt.figure()
plt.contourf(x,N_list,phi1,20,cmap='RdGy')
plt.colorbar()
plt.savefig("phi1_contour.png")

plt.figure()
plt.contourf(x,N_list,phi2,20,cmap='RdGy')
plt.colorbar()
#plt.xscale('log')
plt.savefig("phi2_contour.png")

plt.figure()
plt.contourf(x,N_list,alphaa,20,cmap='RdGy')
plt.colorbar()
plt.savefig("alpha_contour.png")

plt.figure()
plt.contourf(x,N_list,aa,20,cmap='RdGy')
plt.colorbar()
plt.savefig("a_contour.png")
plt.figure()

plt.figure()
plt.plot(x,aintp(x,N_list[0]),label='a @ t={:03f}'.format(N_list[0]))
plt.plot(x,alphaintp(x,N_list[0]),label='alpha @ t={:03f}'.format(N_list[0]))
plt.xlabel("radius")
#plt.xscale('log')
plt.title("a alpha")
plt.legend()
plt.savefig("a_alph_init.png")

plt.figure()
for i in range(4):
    plt.plot(x,abs(momemconstr[len(N_list)/4*i]),label='alphconstr @ t={:03f}'.format(N_list[len(N_list)/4*i]))
plt.plot(x,abs(momemconstr[-4]),label='alphconstr @ t={:03f}'.format(N_list[-4]))
plt.xlabel("radius")
#plt.xscale('log')
plt.yscale('log')
plt.title("momentum constraint")
plt.legend()
plt.savefig("momemconstr_time_x.png")

plt.figure()
for i in range(6):
    plt.plot(x,phi1[len(N_list)/6*i],label='phi1 @ t={:03f}'.format(N_list[len(N_list)/6*i]))
plt.plot(x,phi1[-10],label='phi1 @ t={:03f}'.format(N_list[-10]))
plt.xlabel("radius")
#plt.xscale('log')
plt.title("phi1")
plt.legend()
plt.savefig("phi1_time_x.png")

plt.figure()
for i in range(6):
    plt.plot(x,phi2[len(N_list)/6*i],label='phi2 @ t={:03f}'.format(N_list[len(N_list)/6*i]))
plt.plot(x,phi2[-10],label='phi2 @ t={:03f}'.format(N_list[-10]))
plt.xlabel("radius")
#plt.xscale('log')
plt.title("phi2")
plt.legend()
plt.savefig("phi2_time_x.png")
plt.figure()

plt.figure()
nx = 100
Xp = np.linspace(min(Phi1.T[0])*1.2,max(Phi1.T[0])*.8,nx)[:,None] * np.ones((1,nx))
Y = np.linspace(min(Phi2.T[0])*1.2,max(Phi2.T[0])*.8,nx)[None,:] * np.ones((nx,1))
XY = np.rollaxis(np.array([Xp,Y]), 0, 3)
Z = V2D(Xp,Y)
plt.contour(Xp,Y,Z, np.linspace(np.min(Z), np.max(Z), 200), linewidths=0.5, zorder=0)
plt.colorbar()
plt.plot(Phi1.T[0],Phi2.T[0],'-k', zorder=5)
plt.plot(Phi1.T[0][0],Phi2.T[0][0],'go', zorder=5)
plt.plot(Phi1.T[0][len(Phi1.T[0])/2],Phi2.T[0][len(Phi1.T[0])/2],'yo', zorder=5)
plt.plot(Phi1.T[0][-1],Phi2.T[0][-1],'ro', zorder=5)
plt.xlabel("phi1")
plt.title("phi1 phi2, x=0")
plt.legend()
plt.savefig("phi1_phi2.png")

plt.figure()
nx = 100
Xp = np.linspace(min(Phi1.T[-1])*1.2,max(Phi1.T[-1])*1.2,nx)[:,None] * np.ones((1,nx))
Y = np.linspace(min(Phi2.T[-1])*1.2,max(Phi2.T[-1])*1.2,nx)[None,:] * np.ones((nx,1))
XY = np.rollaxis(np.array([Xp,Y]), 0, 3)
Z = V2D(Xp,Y)
plt.contour(Xp,Y,Z, np.linspace(np.min(Z), np.max(Z), 200), linewidths=0.5, zorder=0)
plt.colorbar()
plt.plot(Phi1.T[-1],Phi2.T[-1], zorder=5)
plt.plot(Phi1.T[-1][0],Phi2.T[-1][0],'go', zorder=5)
plt.plot(Phi1.T[-1][len(Phi1.T[-1])/2],Phi2.T[-1][len(Phi1.T[-1])/2],'yo', zorder=5)
plt.plot(Phi1.T[-1][-1],Phi2.T[-1][-1],'ro', zorder=5)
plt.xlabel("phi1")
plt.title("phi1 phi2, x={:01f}".format(x[-1]))
plt.legend()
plt.savefig("phi1_phi2_xmax.png")

plt.figure()
nx = 100
Xp = np.linspace(min(Phi1.T[xnum/2])*1.2,max(Phi1.T[xnum/2])*1.2,nx)[:,None] * np.ones((1,nx))
Y = np.linspace(min(Phi2.T[xnum/2])*1.2,max(Phi2.T[xnum/2])*1.2,nx)[None,:] * np.ones((nx,1))
XY = np.rollaxis(np.array([Xp,Y]), 0, 3)
Z = V2D(Xp,Y)
plt.contour(Xp,Y,Z, np.linspace(np.min(Z), np.max(Z), 200), linewidths=0.5, zorder=0)
plt.colorbar()
plt.plot(Phi1.T[xnum/2],Phi2.T[xnum/2], zorder=5)
plt.plot(Phi1.T[xnum/2][0],Phi2.T[xnum/2][0],'go', zorder=5)
plt.plot(Phi1.T[xnum/2][len(Phi1.T[xnum/2])/2],Phi2.T[xnum/2][len(Phi1.T[-1])/2],'yo', zorder=5)
plt.plot(Phi1.T[xnum/2][-1],Phi2.T[xnum/2][-1],'ro', zorder=5)
plt.xlabel("phi1")
plt.title("phi1 phi2, x={:01f}".format(x[xnum/2]))
plt.legend()
plt.savefig("phi1_phi2_xhalf.png")

plt.figure()
nx = 100
Xp = np.linspace(min(Pix1.T[-1])*1.2,max(Pix1.T[-1])*1.2,nx)[:,None] * np.ones((1,nx))
Y = np.linspace(min(Pix2.T[-1])*1.2,max(Pix2.T[-1])*1.2,nx)[None,:] * np.ones((nx,1))
XY = np.rollaxis(np.array([Xp,Y]), 0, 3)
Z = V2D(Xp,Y)
plt.contour(Xp,Y,Z, np.linspace(np.min(Z), np.max(Z), 200), linewidths=0.5, zorder=0)
plt.colorbar()
plt.plot(Pix1.T[-1],Pix2.T[-1], zorder=5)
plt.plot(Pix1.T[-1][0],Pix2.T[-1][0],'go', zorder=5)
plt.plot(Pix1.T[-1][len(Pix1.T[-1])/2],Pix2.T[-1][len(Pix1.T[-1])/2],'yo', zorder=5)
plt.plot(Pix1.T[-1][-1],Pix2.T[-1][-1],'ro', zorder=5)
plt.xlabel("pi1")
plt.title("pi1 pi2, x={:01f}".format(x[-1]))
plt.legend()
plt.savefig("pi1_pi2_xmax.png")