import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from bubble_collisions import simulation
import sys

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
        comment='Movie support!')
framerate = 30
duration = 15
writer = FFMpegWriter(fps=framerate, metadata=metadata, bitrate=2000)

inFile = sys.argv[1]
data = simulation.readFromFile(inFile)

Ndata = np.array([d[0] for d in data])
x = np.linspace(-4.9, 4.9, 5000)
N_list = np.linspace(0.0,4.0,framerate*duration)

fig = plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

with writer.saving(fig, "collision_movie.mp4", 160):
    for N in N_list:
        Y=simulation.valsOnGrid(
            N*np.ones_like(x),x, data, [d[0] for d in data], False)
        y = Y[:,0,0]
        alpha = Y[:,0,2]
        a = Y[:,0,3]
        x2 = x
        ax1.cla()
        ax1.plot(x2,y, 'b', lw=1.5)
        ax1.set_ylabel(r"Inflaton field values ($M_{\rm Pl}$)")
        ax1.axis(xmin=x2[0],xmax=x2[-1], 
            ymin=-.002, ymax=.02)
        ax1.text(.85, .9, "$N=%0.2f$"%N, ha='left', va='center', 
            transform=ax1.transAxes)
        ax1.set_xticklabels([])
        ax1.set_title("Colliding bubble universes")
        ax2.cla()
        ax2.plot(x2,alpha, 'c', lw=1.5)
        ax2.plot(x2,a, color=(1,.5,0), lw=1.5)
        ax2.set_ylabel(r"Metric perturbations")
        ax2.axis(xmin=x2[0],xmax=x2[-1], 
            ymin=0, 
            ymax=2)
        ax2.set_xlabel(r"Physical distance ($r_{\rm dS}$)")
        plt.subplots_adjust(hspace=.1, top=.95, right=.9)
        writer.grab_frame()
