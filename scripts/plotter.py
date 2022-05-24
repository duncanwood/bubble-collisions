import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bubble_collisions import simulation
import sys

inFile = sys.argv[1]
data = simulation.readFromFile(inFile)

Ndata = np.array([d[0] for d in data])
x = np.linspace(0.01, 0.500, 100)
N_list = np.linspace(0.0,.1,40)

phi=[]
alphaa = []
aa =[]
fig = plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
fig = plt.figure()

for i in range(len(N_list)):
    Y=simulation.valsOnGrid(
    N_list[i]*np.ones_like(x),x, data, [d[0] for d in data], False)
    
    phi.append(Y[:,0,0])
    alphaa.append(Y[:,0,2])
    aa.append(Y[:,0,3])

plt.contourf(x,N_list,phi,20,cmap='RdGy')
plt.colorbar()
plt.xscale('log');
plt.savefig("phi_contour.pdf")
plt.figure()
plt.contourf(x,N_list,alphaa,20,cmap='RdGy')
plt.colorbar()
plt.xscale('log');
plt.savefig("alpha_contour.pdf")
plt.figure()
plt.contourf(x,N_list,aa,20,cmap='RdGy')
plt.colorbar()
plt.xscale('log');
plt.savefig("a_contour.pdf")

