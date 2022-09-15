import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bubble_collisions import simulation
import sys
from scipy.integrate import odeint
import scipy.interpolate as intp
from matplotlib import cm

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
aintpx = intp.interp2d(x, N_list, aintp(x, N_list, dx=1, dy=0), kind='cubic')
aintpxt = intp.interp2d(x, N_list, aintp(x, N_list, dx=1, dy=1), kind='cubic')
aintpt = intp.interp2d(x, N_list, aintp(x, N_list, dx=0, dy=1) , kind='cubic')
alphaintp = intp.interp2d(x, N_list, alphaa, kind='cubic')
alphaintpx = intp.interp2d(x, N_list, alphaintp(x, N_list, dx=1, dy=0), kind='cubic')

def null_diffeq(w, s):
    dxds = w[2]
    d2xds2 = -w[2]**2*aintpx(w[0], w[1])/aintp(w[0], w[1]) - w[2]**2*(alphaintpx(w[0], w[1]) + 2*aintpt(w[0], w[1]))/alphaintp(w[0], w[1])
    dNds = aintp(w[0], w[1])*w[2]/alphaintp(w[0], w[1])
    d2Nds2 = -w[2]**2*aintp(w[0], w[1])*aintpt(w[0], w[1])/alphaintp(w[0], w[1])**2 - aintp(w[0], w[1])*w[2]/alphaintp(w[0], w[1])*(2*w[2]*alphaintpx(w[0], w[1]))/alphaintp(w[0], w[1])
    dwds = [dxds, dNds, d2xds2, d2Nds2]
    return dwds

w0x0N0 = [0.01,0.0,0.014,0]
sx0N0 = np.linspace(0, 100, 100)
wsx0N0 = odeint(null_diffeq, w0x0N0, sx0N0)

w0x0N02 = [0.01,0.2,0.0085,0]
sx0N02 = np.linspace(0, 100, 100)
wsx0N02 = odeint(null_diffeq, w0x0N02, sx0N02)

w0x0N05 = [0.01,0.5,0.004,0]
sx0N05 = np.linspace(0, 100, 100)
wsx0N05 = odeint(null_diffeq, w0x0N05, sx0N05)

w0x0N08 = [0.01,0.8,0.0012,0]
sx0N08 = np.linspace(0, 100, 100)
wsx0N08 = odeint(null_diffeq, w0x0N08, sx0N08)

xsx0N0 = wsx0N0[:,0]
Nsx0N0 = wsx0N0[:,1]

xsx0N02 = wsx0N02[:,0]
Nsx0N02 = wsx0N02[:,1]

xsx0N05 = wsx0N05[:,0]
Nsx0N05 = wsx0N05[:,1]

xsx0N08 = wsx0N08[:,0]
Nsx0N08 = wsx0N08[:,1]


phi = [abs(phi[i]) for i in range(len(phi))]
Phi = np.array(phi)
fig, ax = plt.subplots()
plt.plot(xsx0N0,Nsx0N0,'r')
plt.plot(xsx0N02,Nsx0N02,'r')
plt.plot(xsx0N05,Nsx0N05,'r')
plt.plot(xsx0N08,Nsx0N08,'r')
plt.xscale('log')
lev_exp = np.arange(-10,np.ceil(np.log10(Phi.max())+1))
levs = np.power(10, lev_exp)
cs = ax.contourf(x, N_list, Phi, levs, norm=cm.colors.LogNorm())
cbar = fig.colorbar(cs)
fig.savefig("phi_contour_null.png")
