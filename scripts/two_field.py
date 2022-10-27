import numpy as np
from bubble_collisions import simulation, models, collisionRunner
from bubble_collisions.cosmoTransitions import pathDeformation as pd
from bubble_collisions.cosmoTransitions import tunneling1D
from bubble_collisions.derivsAndSmoothing import deriv14
import matplotlib.pyplot as plt
import scipy.interpolate as intp
import scipy.optimize
import pickle
import ConfigParser

def runScript(res, fname, config, xsep=1.0):
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
    h = 0.0003
    j = 0.001
    phiF = (0.7,0.05), phiT = (0.01,-0.01) can go for a long time before failing t ~ 500
    """
    try:
        with open(config) as f:
            cp = ConfigParser.ConfigParser()
            cp.readfp(f)
            m = cp.getfloat('params', "m") 
            c = cp.getfloat('params', "c") 
            a = cp.getfloat('params', "a") 
            g = cp.getfloat('params', "g") 
            f = cp.getfloat('params', "f") 
            h = cp.getfloat('params', "h") 
            j = cp.getfloat('params', "j") 
    except Exception as e:
        m = 0.009
        c = 0.01
        a = 0.00174
        g = 0.0000017
        f = 1.0
        h = 7.e-8
        j = 0.0000005
    print(config,m,c,a,g,f,h,j)

    model = models.TiltedHat(
        m=m, a=a, c=c, g=g, f=f, h=h, j=j)

    def V(y):
        return model.V(y,True)
    def dV(y):
        return model.dV(y,True)

    plt.figure()
    phi_list = np.linspace(-0.1,1.1,1000)
    V_list = [model.V((i, 0.005)) for i in phi_list]
    plt.plot(phi_list,V_list)
    plt.yscale("log")
    plt.savefig("V_1_test.pdf")

    phiT = scipy.optimize.minimize(model.V, (0.02,0.002)).x
    phiF_guess = np.array([.25, 0.005])
    phiF_bounds = ((0.5*phiF_guess[0],1.5*phiF_guess[0]),(phiF_guess[1], phiF_guess[1]))
    phiF = scipy.optimize.minimize(model.V, phiF_guess, bounds=phiF_bounds).x
    print 'phiF: {}\nphiT: {}\nVF: {}\nVT: {}'.format(phiF, phiT, model.V(phiF), model.V(phiT))

    path2D = (np.array((phiT, phiF)))
    tobj = pd.fullTunneling(path2D, model.V, model.dV, tunneling_class=tunneling1D.InstantonWithGravity, tunneling_init_params={}, 
    tunneling_findProfile_params={"phitol":1e-4, "xtol": 1.e-12, "thinCutoff": 5e-3, "npoints": 5000})
    #tobj = pd.fullTunneling(path2D, model.V, model.dV, tunneling_findProfile_params=
    #{"phitol": 1e-5, "xtol": 1e-12, "thinCutoff": 5e-3, "npoints": 5000}, deformation_class= pd.Deformation_Spline, verbose=True)

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
    nx = 10
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
    tfix = 10.0
    simulation.setFileParams(fname, xres=2, tout=tfix/10.)
    #simulation.setIntegrationParams(mass_osc = model.dV(phiF)[0]/.01)
    t0,x0,y0 = collisionRunner.calcInitialDataFromInst(
        model, inst1, None, phiF, xsep=1.0, xmin=0.01, xmax = 1.2*np.max(r))
    
    simulation.setMonitorCallback(
        collisionRunner.monitorFunc2D(50., 120., 1))
    t, x, y = simulation.runCollision(x0,y0,t0,tfix, growBounds=False)
    if (t < tfix*.9999):
       raise RuntimeError("Didn't reach tfix. Aborting.")

    print('tmin = ', t0)
    print('rmin=', min(r))
    print('rmax = ', 1.2*max(r))
    print('phiF = ', phiF)
    print('phiT = ', phiT)
    print(m,c,a,g,h,j,f)
    print('tmax=',tfix)

    exportVariables = (t0, min(r), 1.2*max(r), phiF, phiT, m, c, a, g, h, j, f, tfix ) 

    with open('two_field.info', 'w') as f:
        pickle.dump(exportVariables ,f)

    #make a text file and fill it with numbers

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
    parser.add_argument("-c", "--config", dest="config", type=str, default="two_field.config", help="Point to configuration file")



    args = parser.parse_args()
    res=args.resolution
    fname_base = "tilted_hat"
    if args.non_ident:
        fname_base += "_nonident"
    if int(res) == res:
        fname = fname_base+"_%i.dat" % res
    else:
        fname = fname_base+"_%s.dat" % res
    runScript(res, fname, args.config)


