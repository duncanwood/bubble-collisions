import numpy as np
from bubble_collisions import simulation, models, collisionRunner
from bubble_collisions.cosmoTransitions import pathDeformation as pd
from bubble_collisions.cosmoTransitions import tunneling1D
from bubble_collisions.derivsAndSmoothing import deriv14
import matplotlib.pyplot as plt
import scipy.interpolate as intp

def runScript(res, fname, xsep=1.0):
    """
    Run a high-resolution simulation for the benchmark point.
    The 'res' input parameter specifies the overall resolution relative
    to the default value that was used for the collision_pheno paper.
    """
    """
    m = 0.2
    c = 0.1
    a = 0.123
    g = 0.002
    f = 1.0
    h = 0.0008
    j = 0.001
    phiF = (0.7,0.05), phiT = (0.01,-0.01) can go for a long time before failing t ~ 500
    """
    m = 0.1
    c = 0.04
    a = 0.0393
    g = 0.0003
    f = 1.0
    h = 0.000005
    j = 0.0001

    model = models.TiltedHat(
        m=m, a=a, c=c, g=g, f=f, h=h, j=j)

    def V(y):
        return model.V(y,True)
    def dV(y):
        return model.dV(y,True)

    phiF = (0.68,0.005)
    phiT = (-0.015,0.005)
    
    path2D = (np.array((phiT, phiF)))
    tobj = pd.fullTunneling(path2D, model.V, model.dV)

    r = tobj.profile1D.R

    dphi1list = []
    dphi2list = []
    for i in range(len(r)):
        if i == len(r)-1:
            dphi1list.append(dphi1i)
            dphi2list.append(dphi2i)
            break
        dphi1i=(tobj.Phi[:,0][i+1]-tobj.Phi[:,0][i])/(r[i+1]-r[i])
        dphi2i =(tobj.Phi[:,1][i+1]-tobj.Phi[:,1][i])/(r[i+1]-r[i])
        dphi1list.append(dphi1i)
        dphi2list.append(dphi2i)
        
    dphi1 = np.array(dphi1list)
    dphi2 = np.array(dphi2list)
    dphi2D = np.vstack((dphi1, dphi2)).T 

    inst = dict(r=r, phi=tobj.Phi, dphi=dphi2D)
    inst1 = inst2 = inst
    
    plt.figure()
    nx = 100
    X = np.linspace(-1.,1.,nx)[:,None] * np.ones((1,nx))
    Y = np.linspace(-1.,1.,nx)[None,:] * np.ones((nx,1))
    XY = np.rollaxis(np.array([X,Y]), 0, 3)
    Z = model.V(XY)
    plt.contour(X,Y,Z, np.linspace(np.min(Z), max(np.max(Z)*.05,np.min(Z)*1.2), 200), linewidths=0.5)
    plt.colorbar()
    plt.plot(tobj.Phi[:,0], tobj.Phi[:,1], 'b')
    plt.savefig("V2D.png")


    plt.figure()
    plt.plot(r,tobj.Phi[:,0])
    plt.plot(r,dphi1,'r')
    plt.title("phi1(r)")
    plt.savefig("phi1_r.png")
    plt.figure()
    plt.plot(r,tobj.Phi[:,1])
    plt.plot(r,dphi2,'r')
    plt.title("phi2(r)")
    plt.savefig("phi2_r.png")
    plt.figure()

    simulation.setModel(model)
    tfix = 3.0
    simulation.setFileParams(fname, xres=2, tout=tfix/10.)
    #simulation.setIntegrationParams(mass_osc = model.dV(phiF)[0]/.01)
    #what does mass_osc do??
    t0,x0,y0 = collisionRunner.calcInitialDataFromInst(
        model, inst1, None, phiF, xsep=1.0, xmin=0.01, xmax = 100)
    simulation.setMonitorCallback(
        collisionRunner.monitorFunc2D(50., 120., 1))
    t, x, y = simulation.runCollision(x0,y0,t0,tfix, growBounds=False)
    if (t < tfix*.9999):
       raise RuntimeError("Didn't reach tfix. Aborting.")

    print('rmax=',max(r))
    print('phiF=',phiF)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=
        'Run the benchmark quartic barrier simulation with high-resolution '
        'output near the constant field slice phi=1.5.')
    parser.add_argument('resolution', type=float, help=
        'Integration resolution relative to the benchmark case. So, for '
        'example, if resolution=2 the grid spacing will be half that of the '
        'default benchmark case.')
    parser.add_argument("-n", "--non_ident", action="store_true", help=
        "Run non-identical bubble collision")


    args = parser.parse_args()
    res=args.resolution
    fname_base = "tilted_hat"
    if args.non_ident:
        fname_base += "_nonident"
    if int(res) == res:
        fname = fname_base+"_%i.dat" % res
    else:
        fname = fname_base+"_%s.dat" % res
    runScript(res, fname)


