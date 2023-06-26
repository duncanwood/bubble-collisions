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
    m = 0.0058
    c = 0.002
    a = 0.000519
    g = 0.0000001
    f = 1.0
    h = 7.e-8
    j = 0.0000001
    phiF = (0.36,0.005), phiT = (0.0,0.002)
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
        m = 0.0058
        c = 0.002
        a = 0.000519
        g = 0.0000007
        f = 1.0
        h = 0.45e-7
        j = 0.0000001
    print(config,m,c,a,g,f,h,j)

    model = models.TiltedHat(
        m=m, a=a, c=c, g=g, f=f, h=h, j=j)

    def V(y):
        return model.V(y,True)
    def dV(y):
        return model.dV(y,True)

    phiT_guess = np.array([-0.0,0.002])
    phiF_guess = np.array([0.36, 0.005])
    phiT_bounds = ((0.9*phiT_guess[0],1.1*phiT_guess[0]),(0.9*phiT_guess[1], 1.1*phiT_guess[1]))
    phiF_bounds = ((0.9*phiF_guess[0],1.1*phiF_guess[0]),(0.9*phiF_guess[1], 1.1*phiF_guess[1]))
    phiT = scipy.optimize.brute(model.V, phiT_bounds)
    phiF = scipy.optimize.minimize(model.V, phiF_guess, bounds=phiF_bounds).x
    print 'phiF: {}\nphiT: {}\nVF: {}\nVT: {}'.format(phiF, phiT, model.V(phiF), model.V(phiT))

    plt.figure()
    phi_list = np.linspace(-0.05, 1.2*phiF[0], 1000)
    V_list = [model.V((i, 0.0)) for i in phi_list]
    plt.plot(phi_list,V_list)
    plt.savefig("V_1D.pdf")

    #Mpl = 1.1393
    Mpl = 1.1951

    path2D = (np.array((phiT, phiF)))
    tobj = pd.fullTunneling(path2D, model.V, model.dV, tunneling_class=tunneling1D.InstantonWithGravity, tunneling_init_params={"M_Pl":Mpl}, 
    tunneling_findProfile_params={"phitol":1.e-6, "xtol": 1.e-14, "thinCutoff": 5.e-3, "npoints": 4000, "xvalue": 5.0, "extend": 2.0, "rhotol" : 1.})
    #tobj = pd.fullTunneling(path2D, model.V, model.dV, tunneling_findProfile_params=
    #{"phitol": 1e-5, "xtol": 1e-12, "thinCutoff": 5e-3, "npoints": 5000}, deformation_class= pd.Deformation_Spline, verbose=True)

    r = tobj.profile1D.R
    rho = tobj.profile1D.Rho
    rhorE = intp.interp1d(r, rho)
    phirE = intp.interp1d(r, tobj.Phi[:,0])

    ds_ext = np.sqrt(3/(8*np.pi*model.V(phiF)))
    ds_int = np.sqrt(3/(8*np.pi*model.V(phiT)))

    fig, ax1 = plt.subplots()
    ax1.plot(r, rho/ds_ext,'r', label='Bubble')
    #ax1.plot(r, r/ds_ext,'k', label='Flat')
    ax1.plot(r, np.zeros_like(r),'b')
    ax1.plot(r, np.ones_like(r),'b')
    #ax1.axvline(r[np.argmax(rho)], color='r', linestyle='--')
    #ax1.axvline(r[-1], color='b', linestyle='--', label="$r={}$".format(r[-1]))
    #ax1.set_ylim(0, 1.5*np.max(rho))
    ax1.legend(loc='upper left')
    #plt.plot(r, np.zeros_like(r), 'k--')
    #plt.plot(r, r[-1]/(np.pi)*np.sinh(r*np.pi/r[-1]), 'g', label="Lorentzian Bubble?, sinh")
    #plt.plot(r, np.sqrt(3/(8*np.pi*model.V(phiT)))*np.sin(r/np.sqrt(3/(8*np.pi*model.V(phiT)))), 'b', label="Pure dS, sin")
    ax1.set_xlabel("Euclidean radius r_E")
    ax1.set_ylabel("Euclidean Scale Factor /dS false", color='r')
    ax2 = ax1.twinx()
    ax2.plot(r, tobj.Phi.T[0], 'g')
    ax2.set_ylim(min(1.5*np.min(tobj.Phi.T[0]),0.5*np.min(tobj.Phi.T[0])),1.2*np.max(tobj.Phi.T[0]))
    #ax2.axvline(r[np.argmax(tobj.profile1D.dPhi[:-10])], color='g', linestyle='--', alpha=0.5)
    ax2.set_ylabel("Field value", color='g')
    plt.savefig("rho_inst_rE.pdf")

    drhodr = np.gradient(rho,r)
    x4int = np.array([ds_int - np.trapz(np.sqrt(1-(drhodr[:n])**2)) for n in range(len(drhodr))])
    x4real = ds_int-ds_ext+np.sqrt(ds_ext**2-np.array(rho)**2)
    x4realm = ds_int-ds_ext-np.sqrt(ds_ext**2-np.array(rho)**2)
    x4 = np.concatenate((x4real[:np.argmin(x4real)],x4realm[np.argmin(x4real):]))

    plt.figure()
    plt.plot(r, x4, 'b')
    plt.plot(r, x4real, 'k')
    plt.plot(r, x4int, 'c')
    plt.plot(r, ds_int*np.ones_like(r), 'r')
    plt.plot(r, ds_ext*np.ones_like(r), 'g')
    plt.plot(r, (ds_int-2*ds_ext)*np.ones_like(r), 'g')
    plt.savefig("x4.pdf")

    plt.figure()
    plt.plot(r/ds_ext, tobj.Phi.T[0], 'b')
    plt.savefig("phi_units.pdf")

    x4rev = np.linspace(min(x4), max(x4), 10000, endpoint=False)
    rEx4 = intp.interp1d(x4, r)

    def x4x(x, t):
        return ds_ext*np.cosh(1/ds_ext*t)-1/(2*ds_ext)*np.exp(1/ds_ext*t)*x**2 + ds_int - ds_ext
    def xmax(t):
        return np.sqrt(2 * (ds_ext**2 * np.exp(-1/ds_ext * t)) * (1 + np.cosh(1/ds_ext * t)))
    def xmin(t):
        return np.sqrt(2 * (ds_ext**2 * np.exp(-1/ds_ext * t)) * (-1 + np.cosh(1/ds_ext * t)))
    def xTruemin(t):
        return ds_ext * np.sqrt(2*ds_int) * np.sqrt(((1/ds_ext - 1/ds_int) * (-1 + np.cosh(1/ds_ext * t))) 
            / (ds_int*np.exp(1/ds_ext * t) * (1/ds_ext + 1/ds_int * (-1 + np.exp(1/ds_ext * t)))))

    ts = np.linspace(0, 5*ds_int, 100)
    xr = np.linspace(0.01 + xmin(ts[0]), 0.99999*xmax(ts[0]), len(r))

    def x0x(x, t):
        return ds_ext*np.sinh(1/ds_ext*t)+1/(2*ds_ext)*np.exp(1/ds_ext*t)*x**2
    def xksq(x,t):
        return x**2 * np.exp(2/ds_ext*t)
    def xi(x,t):
        return np.arcsinh(x0x(x,t)/np.sqrt(x0x(x,t)**2 + xksq(x,t)))
    def Rwall(x,t):
        return rhorE(rEx4(x4x(x, t)))*np.cosh(xi(x, t))
    def Rfalse(x,t):
        return np.sqrt(-(-x0x(x,t)**2 + (x4x(x,t)- ds_int + ds_ext)**2 - ds_ext**2))
    
    xrlarge = np.linspace(xmax(ts[0]), 1.2*xmax(ts[0]), 100)
    xrlarge = np.concatenate((xr, xrlarge))
    Rlarge = []
    for i in range(len(xrlarge)):
        if xrlarge[i] < xmax(ts[0]):
            Rlarge.append(Rwall(xrlarge[i], ts[0]))
        else:
            Rlarge.append(Rfalse(xrlarge[i], ts[0])-xmax(ts[0])*np.exp(1/ds_ext*ts[0]))
    Rlarge = np.array(Rlarge)

    plt.figure()
    plt.plot(xr, x4x(xr,ts[0]), 'b')
    plt.plot(xr, ds_int*np.ones_like(xr), 'r')
    plt.plot(xr, ds_ext*np.ones_like(xr), 'g')
    plt.plot(xr, (ds_int-2*ds_ext)*np.ones_like(xr), 'g')
    plt.savefig("x4_r.pdf")

    plt.figure()
    plt.plot(ts, xmin(ts), 'b')
    plt.plot(ts, xmax(ts), 'r')
    plt.savefig("r_min_max.pdf")
    
    plt.figure()
    plt.plot(xrlarge, Rlarge , 'r', label='Bubble')
    plt.plot(xrlarge, np.zeros_like(xrlarge), 'k')
    plt.plot(xrlarge, xrlarge*np.exp(1/ds_ext*ts[0]), 'b', label='False vacuum only')
    #plt.plot(xr, Rwall(xr,ts[0])/xr, 'r')
    plt.title("Radius of curvature on initial flat slice, $t=0$")
    plt.xlabel("Flat r coordinate")
    plt.legend()
    plt.savefig("Rfalse.pdf")

    """
    plt.figure()
    plt.plot(ts, xTruemin(ts), 'r')     
    plt.savefig("rtruemin.pdf")
    """

    plt.figure()
    plt.plot(x4rev, rEx4(x4rev), 'b')
    plt.plot(x4, r[-1]*np.ones_like(x4), 'g')
    plt.savefig("rE_x4.pdf")
    
    plt.figure()
    #plt.plot(xr, x4x(xr, tw), 'b')
    plt.plot(xr, rEx4(x4x(xr, ts[0])), 'r')
    plt.plot(xr, r[-1]*np.ones_like(xr), 'g')
    #plt.plot(xr, rEx4(x4x(xr, ts[50])), 'b')
    #plt.plot(r, ds_int*np.ones_like(r), 'r')
    #plt.plot(r, ds_ext*np.ones_like(r), 'g')
    #plt.plot(r, (ds_int-2*ds_ext)*np.ones_like(r), 'g')
    plt.savefig("rE_x_t.pdf")

    plt.figure()
    #plt.plot(xr, x4x(xr, tw), 'b')
    #plt.plot(xr, Rwall(xr, ts[0]), 'r')
    plt.plot(xr[-10:], rhorE(rEx4(x4x(xr, ts[0])))[-10:], 'g')
    #plt.plot(xr, xr, 'k')
    #plt.plot(xr, rhorE(rEx4(x4x(xr, ts[-1])))*np.cosh(xi(xr, ts[-1])), 'b')
    #plt.plot(r, ds_int*np.ones_like(r), 'r')
    #plt.plot(r, ds_ext*np.ones_like(r), 'g')
    #plt.plot(r, (ds_int-2*ds_ext)*np.ones_like(r), 'g')
    plt.savefig("ra_x_t.pdf")

    plt.figure()
    plt.plot(xr, phirE(rEx4(x4x(xr, ts[0]))), 'r')
    #plt.plot(xr, phirE(rEx4(x4x(xr, ts[50]))), 'b')
    plt.savefig("phi_x_t.pdf")

    """
    #Contour plot r*a(t, r)
    X, Y = np.meshgrid(xr, ts[:20])
    Z = rhorE(rEx4(x4x(X, Y)))*np.cosh(xi(X, Y))
    plt.figure()
    plt.contour(X,Y,Z)
    plt.colorbar()
    plt.xlabel('r')
    plt.ylabel('t')
    plt.title('r*a(t,r)')
    plt.savefig("ra.pdf")
    """

    # Want to change all r to xr/xrlarge but not linear transformation
    # Need to make phi's functions of r and r a function of xr then get phi(xr)
    
    Phi1r = intp.UnivariateSpline(r, tobj.Phi[:,0], k=4, s=0)
    dPhi1r = Phi1r.derivative()
    d2Phi1r = dPhi1r.derivative()
    Phi2r = intp.UnivariateSpline(r, tobj.Phi[:,1], k=4, s=0)
    dPhi2r = Phi2r.derivative()
    d2Phi2r = dPhi2r.derivative()

    """
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
    """

    # Arrays of transformed phi's into xr from r  
    phi1 = np.array(Phi1r(rEx4(x4x(xr, ts[0]))))
    phi2 = np.array(Phi2r(rEx4(x4x(xr, ts[0]))))
    dphi1 = np.array(dPhi1r(rEx4(x4x(xr, ts[0]))))
    dphi2 = np.array(dPhi2r(rEx4(x4x(xr, ts[0]))))
    phi2D = np.vstack((phi1, phi2)).T
    dphi2D = np.vstack((dphi1, dphi2)).T  

    plt.figure()
    plt.plot(r, phirE(r), 'r')
    plt.plot(r, Phi1r(r))
    plt.savefig("phi_rE.pdf")

    #inst = dict(r=r, phi=tobj.Phi, dphi=dphi2D, rho=rho)
    inst = dict(r=xr, phi=phi2D, dphi=dphi2D, rho=rhorE(rEx4(x4x(xr, ts[0])))*np.cosh(xi(xr, ts[0])))
    inst1 = inst2 = inst
    
    plt.figure()
    nx = 10
    X = np.linspace(-0.3,0.3,nx)[:,None] * np.ones((1,nx))
    Y = np.linspace(-0.3,0.3,nx)[None,:] * np.ones((nx,1))
    XY = np.rollaxis(np.array([X,Y]), 0, 3)
    Z = model.V(XY)
    plt.contour(X,Y,Z, np.linspace(np.min(Z), np.max(Z), 200), linewidths=0.5)
    plt.colorbar()
    plt.plot(phi1, phi2, 'b')
    plt.savefig("V2D.pdf")

    simulation.setModel(model)
    rmin = 0.01
    tfix = 10.
    tau_choice = 0.001
    rmax = xr[-1]
    t_steps = 30.
    t_metric = 0.3

    print "dS exterior = {}".format(ds_ext)
    print "dS interior = {}".format(ds_int)
    print "bubble max size = {}".format(np.max(rho))

    simulation.setFileParams(fname, xres=2, tout=tfix/10.)
    simulation.setIntegrationParams(minStepsPerPeriod = t_steps, t_metric = t_metric)
    t0,x0,y0 = collisionRunner.calcInitialDataFromInst(
        model, inst1, None, phiF, xsep=1.0, xmin=rmin, xmax=rmax, rel_t0=tau_choice)
    
    plt.figure()
    plt.plot(x0, y0.T[0], 'r')
    plt.plot(r,tobj.Phi.T[0] , 'b')
    plt.savefig("initial_fields.pdf")

    simulation.setMonitorCallback(
        collisionRunner.monitorFunc2D(35., 100., 1))
    t, x, y = simulation.runCollision(x0, y0, t0, tfix, growBounds=False)
    if (t < tfix*.9999):
       raise RuntimeError("Didn't reach tfix. Aborting.")

    print('tmin = ', t0)
    print('ext dS = ', ds_ext)
    print('phiF = ', phiF)
    print('phiT = ', phiT)

    exportVariables = (t0, rmin, rmax, phiF, phiT, m, c, a, g, h, j, f, tfix) 

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


