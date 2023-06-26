import numpy as np
from bubble_collisions import simulation, models, collisionRunner
from bubble_collisions.cosmoTransitions import tunneling1D
#from bubble_collisions.cosmoTransitions_old import tunneling1D as tunneling1D_old
from bubble_collisions.derivsAndSmoothing import deriv14
import matplotlib.pyplot as plt

def runScript(res=1.0, fname=None, xsep=1.0):
    """
    Run a high-resolution simulation for the benchmark point.
    The 'res' input parameter specifies the overall resolution relative
    to the default value that was used for the collision_pheno paper.
    """
    a = -0.1
    C = 2.
    
    model = models.Johnson1D(
        a = a, C = C)

    def V(y):
        return model.V(y,True)
    def dV(y):
        return model.dV(y,True)
    phiF = 1./2*(a+np.sqrt(a**2+4))
    phiT = 1./2*(a-np.sqrt(a**2+4))

    #tobj0 = tunneling1D_old.bubbleProfile(phiT, phiF, V, dV, alpha=3)
    #tobj0.kappa = 8*np.pi
    #p0 = tobj0.findProfile(xtol=1e-7, phitol=phiT*1e-6, thinCutoff=2e-3,
    #                        npoints=3000, verbose=True)
    tobj = tunneling1D.SingleFieldInstanton(phiT, phiF, V, dV)
    profile = tobj.findProfile(
        xtol=1e-4, phitol=1e-4, thinCutoff=0.01, npoints=500)

    r, phi = profile.R, profile.Phi[:,np.newaxis] # get phi to have shape (nx, 1)
    dphi = profile.dPhi[:,np.newaxis]
    inst = dict(r=r, phi=phi, dphi=dphi)
    
    plt.figure()
    plt.plot(r,phi,'r')
    #plt.plot(r, profile.Rho, 'b')
    plt.title("phi, rho")
    plt.savefig("phi_rh0.pdf")
    plt.figure()
    plt.plot(phi,V(phi))
    plt.title("V(phi)")
    plt.savefig("V_phi.pdf")
    
    print("\n \n \n \n Now do it again with gravity \n \n \n \n")


    tobj2 = tunneling1D.InstantonWithGravity(phiT, phiF, V, dV, M_Pl=50.)
    profile2 = tobj2.findProfile(
        xtol=1e-4, phitol=1e-4, thinCutoff=0.001, npoints=3000)

    r2, phi2 = profile2.R, profile2.Phi[:,np.newaxis] # get phi to have shape (nx, 1)
    dphi2 = profile2.dPhi[:,np.newaxis]
    inst2 = dict(r=r2, phi=phi2, dphi=dphi2)
    
    
    plt.figure()
    plt.plot(r2,phi2,'r')
    plt.plot(r,phi, 'k')
    plt.plot(r2, profile2.Rho, 'b')
    plt.title("phi, rho")
    plt.savefig("phi_rho2.pdf")
    plt.figure()
    plt.plot(phi2,V(phi2))
    plt.title("V(phi)")
    plt.savefig("V_phi2.pdf")
    

    """
    model.setParams(phi0=phi_vac)
    # At first, the output should be kind of coarse.
    simulation.setModel(model)
    tfix = 1.0
    simulation.setFileParams(fname, xres=2, tout=tfix/1000.)
    simulation.setIntegrationParams(mass_osc = dV(phi_vac+.01)/.01)
    t0,x0,y0 = collisionRunner.calcInitialDataFromInst(
        model, inst1, None, phiF, xsep=1.0, xmin=0.01, xmax=1.0)
    simulation.setMonitorCallback(
        collisionRunner.monitorFunc1D(40., 200., 1))

    t, x, y = simulation.runCollision(x0,y0,t0,tfix, growBounds=False)
    if (t < tfix*.9999):
       raise RuntimeError("Didn't reach tfix. Aborting.")


    # Truncate the simulation
    truncation = .95
    bubble_radius = 2*np.arctan(np.tanh(t/2.0))
    dx = bubble_radius * (1-truncation)
    xmin, xmax = -bubble_radius+dx, bubble_radius + xsep - dx
    i = (x < xmax) & (x > xmin)
    x,y = x[i], y[i]

    # t_start and t_end should encompass the phi=1.5 surface (which is the 
    u surface that I used before to calculate full-sky bubbles).
    t_start_hr_out = 42.0
    t_end_hr_out = 47.0
    t,x,y = simulation.runCollision(x,y,t,t_start_hr_out, 
        growBounds=False, overwrite=False)
    simulation.setFileParams(xres=1, tout=0.0) # output every time slice
    t,x,y = simulation.runCollision(x,y,t,t_end_hr_out, 
        growBounds=False, overwrite=False)
    return t,x,y


def runNonIdentScript(res, fname, xsep=1.0):
   """ """
    Run a high-resolution simulation for the benchmark model in a non-identical
    bubble simulation.
    The 'res' input parameter specifies the overall resolution relative
    to the default value that was used for the collision_pheno paper.
   """ """
    mu = 0.01
    omega = 0.054288352331898125
    Delta_phi = 0.000793447464875
    phi_vac = 3.0
    mu_neg = 0.005
    omega_neg = 0.024128156591954723
    Delta_phi_neg = -0.0011901711973125    

    model = models.GenericPiecewise_NoHilltop_Model(
        mu=mu, omega=omega, Delta_phi=Delta_phi, phi0=0.0)
    model.setParams(mu=mu_neg, omega=omega_neg, Delta_phi=Delta_phi_neg, 
        phi0=0.0, posneg=-1)

    def V(y):
        return model.V(y,True)
    def dV(y):
        return model.dV(y,True)
    phiF = 0.0
    phiT = Delta_phi

    tobj = tunneling1D.InstantonWithGravity(phiT, phiF, V, dV)
    profile = tobj.findProfile(
        xtol=1e-7, phitol=1e-6, thinCutoff=2e-3, npoints=5000)
    r, phi = profile.R, profile.Phi[:,np.newaxis] # get phi to have shape (nx, 1)
    dphi = profile.dPhi[:,np.newaxis]
    inst1 = dict(r=r, phi=phi, dphi=dphi)

    tobj = tunneling1D.InstantonWithGravity(Delta_phi_neg, phiF, V, dV)
    profile = tobj.findProfile(
        xtol=1e-7, phitol=1e-6, thinCutoff=2e-3, npoints=5000)
    r, phi = profile.R, profile.Phi[:,np.newaxis] # get phi to have shape (nx, 1)
    dphi = profile.dPhi[:,np.newaxis]
    inst2 = dict(r=r, phi=phi, dphi=dphi)

    model.setParams(phi0=phi_vac)
    # At first, the output should be kind of coarse.
    simulation.setModel(model)
    simulation.setFileParams(fname, xres=8, tout=.1)
    simulation.setIntegrationParams(mass_osc = dV(phi_vac+.01)/.01)
    t0,x0,y0 = collisionRunner.calcInitialDataFromInst(
        model, inst1,inst2, phiF, xsep)
    simulation.setMonitorCallback(
        collisionRunner.monitorFunc1D(100.*res, 500.*res, 4))
    tfix = 5.0
    t, x, y = simulation.runCollision(x0,y0,t0,tfix)
    if (t < tfix*.9999):
        raise RuntimeError("Didn't reach tfix. Aborting.")

    # Truncate the simulation; make sure to exclude the domain wall
    truncation = .95
    bubble_radius = 2*np.arctan(np.tanh(t/2.0))
    dx = bubble_radius * (1-truncation)
    xmin, xmax = -bubble_radius+dx, bubble_radius + xsep - dx
    i = (x < xmax) & (x > xmin)
    x,y = x[i], y[i]
    alpha = y[:, -2]
    dalpha = abs(deriv14(alpha, x))
    xmax = x[dalpha==max(dalpha)][0] - dx
    i = (x < xmax) & (x > xmin)
    x,y = x[i], y[i]    

    # t_start and t_end should encompass the phi=1.5 surface (which is the 
    # surface that I used before to calculate full-sky bubbles).
    t_start_hr_out = 42.0
    t_end_hr_out = 47.0
    t,x,y = simulation.runCollision(x,y,t,t_start_hr_out, 
        growBounds=False, overwrite=False)
    simulation.setFileParams(xres=1, tout=0.0) # output every time slice
    t,x,y = simulation.runCollision(x,y,t,t_end_hr_out, 
        growBounds=False, overwrite=False)
    return t,x,y
"""

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
    fname_base = "super_simulation"
    if args.non_ident:
        fname_base += "_nonident"
    if int(res) == res:
        fname = fname_base+"_%i.dat" % res
    else:
        fname = fname_base+"_%s.dat" % res
    #if args.non_ident:
    #    runNonIdentScript(res, fname)
    #else:
    runScript(res, fname)


