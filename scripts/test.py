from __future__ import absolute_import
from __future__ import print_function

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

def runSimulations(overwrite):
    """    
    m = 0.0058
    c = 0.002
    a = 0.000519
    g = 0.0000007
    f = 1.0
    h = 0.45e-7
    j = 0.0000001
    phi2 = 0.0

    model = models.TiltedHat1D(
        m=m, a=a, c=c, g=g, f=f, h=h, j=j, phi2=phi2)
    """

    a = 0.1
    C = 2.0

    model = models.Johnson1D(
        a=a, C=C)

    def V(y):
        return model.V(y,True)
    def dV(y):
        return model.dV(y,True)
    plt.figure()
    phi_list = np.linspace(-1.2,1.2, 1000)
    V_list = [V((i)) for i in phi_list]
    plt.plot(phi_list,V_list)
    plt.savefig("V_1D.pdf")

    phiF = -0.95
    phiT = 1.05
    mpl = 385.
    eps = np.sqrt(8.*np.pi/(3*mpl**2))

    """tobj = tunneling1D.SingleFieldInstanton(phiT, phiF, V, dV)
    profile = tobj.findProfile(
        xtol=1e-10, phitol=1e-5, thinCutoff=1e-4, npoints=1000)"""
    tobj = tunneling1D.InstantonWithGravity(phiT, phiF, V, dV, M_Pl=mpl)
    profile = tobj.findProfile(
        xtol=1e-10, phitol=1e-5, thinCutoff=1e-4, npoints=1000, xvalue=2.)

    print(eps)

    r, phi = profile.R, profile.Phi[:,np.newaxis] # get phi to have shape (nx,1)
    rho = profile.Rho
    dphi = profile.dPhi[:,np.newaxis]
    inst = dict(r=r, phi=phi, dphi=dphi)
    inst1 = inst2 = inst

    ds_ext = np.sqrt(3/(8*np.pi*V(phiF)))

    plt.figure()
    plt.plot(r, rho, 'r')
    plt.plot(r, ds_ext*np.ones_like(r),'b')
    plt.savefig("rho.pdf")

    plt.figure()
    plt.plot(r, phi, 'b')
    plt.savefig("phi_units.pdf")

    fjdklfskdfjlsfsdljs
    """
    model.setParams(phi0=phi_vac)
    simulation.setFileParams("test/test_collision_fixed.dat", xres=4, tout=.05)
    simulation.setIntegrationParams(mass_osc = dV(phi_vac+.01)/.01)
    if (overwrite or not os.path.exists("test/test_collision_fixed.dat")):
        output = collisionRunner.runModelFromInstanton_fixedgrid(
            model, inst1, inst2, phiF, xsep=1, xdensity=200.0)

    simulation.setFileParams("test/test_collision.dat", 
        "test/test_collision_chris.dat", xres=4)
    simulation.setMonitorCallback(collisionRunner.monitorFunc1D(50., 250., 4))
    if (overwrite or not os.path.exists("test/test_collision.dat")):
        output = collisionRunner.runModelFromInstanton(
            model, inst1, inst2, phiF, xsep=1.0, tfix=4.0, tmax=50.0)
    """
    simulation.setFileParams("test/test_no_collision.dat", 
        "test/test_no_collision_chris.dat", xres=4)
    simulation.setMonitorCallback(collisionRunner.monitorFunc1D(50., 250., 2))
    if (overwrite or not os.path.exists("test/test_no_collision.dat")):
        output = collisionRunner.runModelFromInstanton(
            model, inst1, None, phiF, xsep=1.0, tfix=4.0, tmax=4.0)      

from bubble_collisions import full_sky
import matplotlib.pyplot as plt
def runFullSky():
    print("\nCreating full sky perturbation plot...")
    fsp = full_sky.FullSkyPerturbation("test/test_collision.dat", phi0=1.0)
    fig=plt.figure()
    ax=plt.subplot(111)
    for x0 in np.arange(-.62,-.25,.05):
        ax.plot(*fsp.perturbationCenteredAt(x0))
    ax.axis(xmin=-1, xmax=0, ymin=-.01, ymax=.05)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\mathcal{R}(\xi$)")
    ax.set_title("Full-sky perturbations")
    fig.savefig("test/test_fullsky_perturbations.pdf")

def comparePerturbationCalculations(make_plot=True):
    print("\nComparing different perturbation calculations...")
    # Using full sky ---
    fsp = full_sky.FullSkyPerturbation("test/test_collision.dat", phi0=1.0)
    xi_fs, R_fs = fsp.perturbationCenteredAt(-0.7)
    # Using geodesic integration ---
    xi_geo = np.linspace(-2,2,1001)
    tau = np.append(np.linspace(0,48,49),np.linspace(49,51,15))
    geo_data0 = geodesics.findGeodesics(
        xi_geo, tau,
        "test/test_no_collision_chris.dat",
        "test/test_no_collision.dat")
    geo_data1 = geodesics.findGeodesics(
        xi_geo, tau,
        "test/test_collision_chris.dat",
        "test/test_collision.dat")
    # Find the metrics at the observer's position (X=Y=0) for observers at
    # different locations (different xi)
    # Note that all inputs must be arrays
    tau0 = np.array([50.0])
    X = Y = np.array([0.0])
    a1,H1 = geodesics.scaleFactor(geo_data0, tau0)
    g0 = geodesics.observerMetric(tau0, X, Y, xi_geo, geo_data0)
    g1 = geodesics.observerMetric(tau0, X, Y, xi_geo, geo_data1)
    D, E, phi_term, R_geo = geodesics.perturbationsFromMetric(
        g0,g1, a1, H1, divideOutCurvature=True)
    # Using analytic approximations ---
    xi_anal = np.linspace(-2,2,5000)
    R_anal = bubble_analytics.analyticPerturbations(
        xi_anal, "test/test_collision.dat", 1.0)
    # Make the figure ---
    if make_plot:
        fig=plt.figure()
        ax=plt.subplot(111)
        ax.plot(xi_fs, R_fs, 'k', label="full-sky")
        ax.plot(xi_geo, R_geo, 'b', label="geodesics")
        ax.plot(xi_anal, R_anal, 'm', label="analytic")
        ax.axis(xmin=-.75, xmax=-.15, ymin=-.003, ymax=.03)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\mathcal{R}(\xi$)")
        ax.set_title("Perturbation calculted 3 ways")
        ax.legend(loc='upper left')
        fig.savefig("test/test_perturbation_comparison.pdf")    
    return xi_fs, R_fs, xi_geo, R_geo, xi_anal, R_anal


    
def runFitting():
    print("\nCalculating best-fit parameters...")
    fsp = full_sky.FullSkyPerturbation("test/test_collision.dat", phi0=1.0)
    xi,R = fsp.perturbationCenteredAt(-.7)
    import bubble_collisions.perturbation_fits as pf
    pow_params = tuple(pf.calcFit(xi,R,pf.powerFit))
    quad_params = tuple(pf.calcFit(xi,R,pf.quadFit,weight_small_R=False))
    print("Power-law params: xi0=%0.5f, A=%0.5f, kappa=%0.5f" % pow_params)
    print("Quadratic fit params: xi0=%0.5f, a=%0.5f, b=%0.5f" % quad_params)
    fig=plt.figure()
    ax=plt.subplot(111)
    ax.plot(xi, R, 'k', label='perturbation')
    ax.plot(xi, pf.powerFit(xi, *pow_params), 'r', label="power-law fit")
    ax.plot(xi, pf.quadFit(xi, *quad_params), 'm', label="quadratic fit")
    ax.axis(xmin=-.7, xmax=-.5, ymin=-.001, ymax=.005)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\mathcal{R}(\xi$)")
    ax.set_title("Perturbation fits")
    plt.legend()
    fig.savefig("test/test_perturbation_fits.pdf")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="count", default=0,
        help="If set, simulation data will get overwritten.")
    args = parser.parse_args()

    try:
        os.mkdir('test')
    except:
        pass
    runSimulations(args.overwrite)
    #runFullSky()
    #runFitting()
    #comparePerturbationCalculations()
    





