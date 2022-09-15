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

xnum = 400
ynum = 400
Ndata = np.array([d[0] for d in data])
x = np.linspace(0.01, 50, xnum)
N_list = np.linspace(0.0,150.0,ynum)

phi1 = []
phi2 = []
alphaa = []
aa =[]
pix1 = []
pix2 = []

for i in range(len(N_list)):
    Y=simulation.valsOnGrid(
    N_list[i]*np.ones_like(x),x, data, [d[0] for d in data], False)
    
    phi1.append(Y[:,0,0])
    phi2.append(Y[:,0,1])
    pix1.append(Y[:,0,2])
    pix2.append(Y[:,0,3])
    alphaa.append(Y[:,0,-2])
    aa.append(Y[:,0,-1])
    #ba.append(Y[:,0,-3])

phi1intp = intp.interp2d(x, N_list, phi1, kind='cubic')
phi1intpx = phi1intp(x, N_list, dx=1, dy=0)
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

# x0, y0, x'0, y'0
npts = 10000
endpt = 10000
vel = 1.0
Noffset = 9.0

w0x0N0 = [0.01,0.0,vel,0]
sx0N0 = np.linspace(0, endpt, npts)
wsx0N0 = odeint(null_diffeq, w0x0N0, sx0N0)

w0x0N1 = [0.01,1.0,vel,0]
sx0N1 = np.linspace(0, endpt, npts)
wsx0N1 = odeint(null_diffeq, w0x0N1, sx0N1)

w0x0N2 = [0.01,2.0,vel,0]
sx0N2 = np.linspace(0, endpt, npts)
wsx0N2 = odeint(null_diffeq, w0x0N2, sx0N2)

w0x0N4 = [0.01,4.0,vel,0]
sx0N4 = np.linspace(0, endpt, npts)
wsx0N4 = odeint(null_diffeq, w0x0N4, sx0N4)

w0x0Nm1 = [0.01,-1.0,vel,0]
sx0Nm1 = np.linspace(0, endpt, npts)
wsx0Nm1 = odeint(null_diffeq, w0x0Nm1, sx0Nm1)

w0x0Nm2 = [0.01,-2.0,vel,0]
sx0Nm2 = np.linspace(0, endpt, npts)
wsx0Nm2 = odeint(null_diffeq, w0x0Nm2, sx0Nm2)

w0x0Nm3 = [0.01,-3.0,vel,0]
sx0Nm3 = np.linspace(0, endpt, npts)
wsx0Nm3 = odeint(null_diffeq, w0x0Nm3, sx0Nm3)

w0x0Nm4 = [0.01,-4.0,vel,0]
sx0Nm4 = np.linspace(0, 3*endpt, 3*npts)
wsx0Nm4 = odeint(null_diffeq, w0x0Nm4, sx0Nm4)

xsx0N0 = wsx0N0[:,0]
Nsx0N0 = wsx0N0[:,1]
Nsnx0N0 = Noffset-wsx0N0[:,1]

xsx0N1 = wsx0N1[:,0]
Nsx0N1 = wsx0N1[:,1]
Nsnx0N1 = Noffset-wsx0N1[:,1]

xsx0N2 = wsx0N2[:,0]
Nsx0N2 = wsx0N2[:,1]
Nsnx0N2 = Noffset-wsx0N2[:,1]

xsx0N4 = wsx0N4[:,0]
Nsx0N4 = wsx0N4[:,1]
Nsnx0N4 = Noffset-wsx0N4[:,1]

xsx0Nm1 = wsx0Nm1[:,0]
Nsx0Nm1 = wsx0Nm1[:,1]
Nsnx0Nm1 = Noffset-wsx0Nm1[:,1]

xsx0Nm2 = wsx0Nm2[:,0]
Nsx0Nm2 = wsx0Nm2[:,1]
Nsnx0Nm2 = Noffset-wsx0Nm2[:,1]

xsx0Nm3 = wsx0Nm3[:,0]
Nsx0Nm3 = wsx0Nm3[:,1]
Nsnx0Nm3 = Noffset-wsx0Nm3[:,1]

xsx0Nm4 = wsx0Nm4[:,0]
Nsx0Nm4 = wsx0Nm4[:,1]
Nsnx0Nm4 = Noffset-wsx0Nm4[:,1]


alphaline=0.5
Phi1 = np.array(phi1)
fig, ax = plt.subplots()
plt.plot(xsx0N4,Nsx0N4,'r',alpha=alphaline)
plt.plot(xsx0N4,Nsnx0N4,'k',alpha=alphaline)
plt.plot(xsx0N2,Nsx0N2,'r',alpha=alphaline)
plt.plot(xsx0N2,Nsnx0N2,'k',alpha=alphaline)
plt.plot(xsx0N1,Nsx0N1,'r',alpha=alphaline)
plt.plot(xsx0N1,Nsnx0N1,'k',alpha=alphaline)
plt.plot(xsx0N0,Nsx0N0,'r',alpha=alphaline)
plt.plot(xsx0N0,Nsnx0N0,'k',alpha=alphaline)
plt.plot(xsx0Nm1,Nsx0Nm1,'r',alpha=alphaline)
plt.plot(xsx0Nm1,Nsnx0Nm1,'k',alpha=alphaline)
plt.plot(xsx0Nm2,Nsx0Nm2,'r',alpha=alphaline)
plt.plot(xsx0Nm2,Nsnx0Nm2,'k',alpha=alphaline)
plt.plot(xsx0Nm3,Nsx0Nm3,'r',alpha=alphaline)
plt.plot(xsx0Nm3,Nsnx0Nm3,'k',alpha=alphaline)
plt.plot(xsx0Nm4,Nsx0Nm4,'r',alpha=alphaline)
plt.plot(xsx0Nm4,Nsnx0Nm4,'k',alpha=alphaline)
plt.ylim((min(N_list),max(N_list)))
plt.xlim((min(x),max(x)))
cs = ax.contourf(x, N_list, Phi1)
plt.ylabel("time")
plt.xlabel("radius")
plt.title("Phi 1")
fig.colorbar(cs)
fig.savefig("phi_contour_null.png")
