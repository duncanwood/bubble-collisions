"""
This module provides a set of convenience functions to set up and run a
simulation.

The easiest way to run a simulation is to use :func:`runModelFromInstanton`.
The necessary input will be the scalar field model, 
the initial instantons for each bubble (as calculated using 
:func:`cosmoTransitions.pathDeformation.fullTunneling`),
the field value of the false vacuum, and the kinematic separation between the
bubbles. Additionally, the user will need to set the monitor function prior
to running the simulation.
All of the rest can be handled by the default arguments.
"""

import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy import interpolate, optimize

from .derivsAndSmoothing import deriv14, deriv23
from . import simulation
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def timeSlicesForShockParams(
		xsep, obs_inst_radius, col_wall_inner, col_wall_outer, 
		tmax, minstep=5e-4, stepsPerShock=20, no_collision_step=0.1):
	"""
	Outputs a list of time slices at which to save the simulation.

	Parameters
	----------
	xsep : 
		Separation between bubbles, or zero for single bubble
	obs_inst_radius : 
		Radius of the observer bubble at N = 0
	col_wall_inner :
		Initial inner radius of the collision instanton
	col_wall_outer :
		Initial outer radius of the collision instanton
	tmax : 
		Maximum allowed time output
	minstep : 
		Smallest allowed time step
	stepsPerShock : 
		Number of times to save during a single geodesic crossing of the shock
	no_collision_step : 
		The output time step will be no bigger than 
		``obs_radius(N) * no_collision_step``.
	"""
	if minstep <= 0:
		return 0.0
	def wall_traj(x0, N):
		return np.arctan(np.sqrt((np.sinh(N)/np.cos(x0))**2 + np.tan(x0)**2))
	
	N_first_hit = np.arcsinh(np.tan(xsep*0.5))
	# just make deltax_wall a constant that depends on where the two first collide
	deltax_wall = (wall_traj(col_wall_outer,N_first_hit) 
		- wall_traj(col_wall_inner,N_first_hit))
		
	tout = [0]
	if (xsep == 0): N_first_hit = tmax*2
	while (tout[-1] < tmax):
		N = tout[-1]
		dt_no_col = wall_traj(obs_inst_radius,N) * no_collision_step
		if (N + 2*dt_no_col > N_first_hit):
			x = xsep - wall_traj(col_wall_inner, N) 
				# (the x value of the collision wall/shock)
			timeToCross = deltax_wall * np.cosh(N) * (
				np.cos(x)**2*np.sinh(N)) / (np.cos(x)**2*np.sinh(N) + np.sin(x))
			dt_col = timeToCross / (stepsPerShock+1e-100)
			dt = min(dt_no_col, dt_col)
		else:
			dt = dt_no_col
		dt = max(dt, minstep)
		tout.append(N+dt)
	return np.array(tout)
	
def wallPositionFromInstanton(r, phi, cutoff = 0.05):
	"""
	Returns the inner and outer positions of the bubble wall.

	Parameters
	----------
	r : array of shape (nx,)
	phi : array of shape (nx, nfields)
	cutoff : float, optional
		Defines the edge of the bubble wall. For example, if ``cutoff=0.05``
		(the default), then outer edge of the bubble wall is defined as the
		radius that contains 95% of the field values between the bubble's
		center and r = infinity.
	"""
	phi = phi-phi[0]
	phi = np.sum(phi*phi, axis=1)**0.5
	phi_norm = phi/phi[-1]
	r_inner = r[phi_norm > cutoff][0]
	r_outer = r[phi_norm > 1-cutoff][0]
	return r_inner, r_outer

def makeTwoSidedInstanton_withFill(r, phi, phiF):
	"""
	Takes as input the bubble wall for a single instanton, and outputs
	an instanton going from -rmax to rmax. It fills in the center smoothly,
	and adjusts the vertical scale (slightly) so that *phi(rmax) = phiF* 
	exactly.

	This function is no longer necessary with the new version of
	CosmoTransitions, since the new version already outputs an instanton that
	goes all the way down to r=0.
	
	Parameters
	----------
	r : array of shape (nx,)
	phi : array of shape (nx, nfields)
	"""
	assert ((r[1:]-r[:-1]) > 0).all(), "r needs to be an increasing array"
	assert r[0] > 0
	
	# First, make the center of the bubble.
	dx0 = r[1]-r[0]
	phi0 = phi[1:2] # shape (1,nfields)
	dphi0 = deriv14(phi[:5].T, r[:5]).T[1:2] # shape (1,nfields)
	r0 = r[1]
	num_r_center = np.floor(2*r0/dx0) + 1
	r_center = np.linspace(-r0, r0, num_r_center)[1:-1]
	
	# Make phi a parabola across the center, fit to match dphi0.
	# phi = 0.5*A*r^2 + B
	# A = dphi0/r0
	# B = phi0 - 0.5*dphi0*r0
	phi0 = phi0 - 0.5*dphi0*r0
	phi_center = 0.5*(dphi0/r0)*r_center[:,np.newaxis]**2 + phi0
	
	# Assemble the 3 pieces.
	inst = {}
	inst["x"] = np.append(np.append(-r[:0:-1], r_center), r[1:])
	inst["phi"] = np.append(
		np.append(phi[:0:-1], phi_center, axis=0), phi[1:], axis=0)
	
	# Now adjust the whole thing so that  phi[-1] = phiF
	phi = inst["phi"]
	inst["phi"] = ( (phi - phi[-1:])/(phi0 - phi[-1:]) ) * (phi0 - phiF) + phiF
	return inst

def makeTwoSidedInstanton(inst, phiF):
	"""
	Takes as input the bubble wall for a single instanton, and outputs
	an instanton going from -rmax to rmax. It adjusts the vertical scale 
	(slightly) so that *phi(rmax) = phiF* exactly.
	
	Parameters
	----------
	inst : dictionary
		Contains arrays for keys 'r' (length nx), 'phi' (shape (nx, nfields)),
		and 'dphi' (shape (nx, nfields)).
	phiF : array of length nfields
		Field value at the false vacuum
	"""
	r,phi,dphi = inst['r'], inst['phi'], inst['dphi']
	x = np.append(-r[::-1], r[1:])
	dphi = dphi * (phi[0] - phiF) / (phi[0] - phi[-1])
	dphi = np.append(-dphi[::-1], dphi[1:], axis=0)
	phi = ( (phi - phi[-1])/(phi[0] - phi[-1]) ) * (phi[0] - phiF) + phiF
	phi = np.append(phi[::-1], phi[1:], axis=0)
	return dict(x=x, phi=phi, dphi=dphi)

def calcInitialDataFromInst(model, inst1, inst2, phiF, xsep, rel_t0 = 0.001, 
	tail_extension = 0.1, xmin=None, xmax=None):
	"""
	Calculate initial conditions for the simulation starting from instantons.

	This function works by making two-sided instantons, fitting splines to the
	instantons, then intepolating along the spline and summing the two 
	instantons together.

	Parameters
	----------
	model : ModelObject
		The model to be used in the simulation. Should have functions *V* and
		*dV*.
	inst1 : instanton object
		The observer instanton centered at *x=0*.
		Should be a dictionary-like object with keys 'r', 'phi', and 'dphi', 
		where the 'r' array has length *nx* and the phi and dphi arrays have 
		shape *(nx, nfields)*.
	inst2 : instanton object
		The collision instanton centered at *x=xsep*.
	phiF : array of length *nfields*
		The false-vacuum field value.
	xsep : float
	rel_t0 : float, optional
		Initial time variable relative to the size of the observer instanton.
	tail_extension : float, optional
		How far to extend the simulation boundaries beyond the instanton
		end-points (only used if xmin and xmax are not set)
	xmin : float, optional
		The left boundary of the simulation. Useful for when not growing the
		bounds of the simulation.
	xmax : float, optional
		The right boundary of the simulation.

	Returns
	-------
		t0 : float
			The initial time value.
		x : array of length nx
			The initial grid.
		Y : array of shape ``(nx, 2*nfields+3)``
			The initial field values :math:`\\phi`, their rescaled momenta
			:math:`\\Pi`, and the initial metric functions :math:`\\alpha` and
			:math:`a`. and b
	"""
	phiF = np.array(phiF).ravel() 
	# so that we can just input a number when there's only 1 field

	# First, make the instantons two-sided.
	inst1 = makeTwoSidedInstanton(inst1, phiF)
	inst1['phi'] -= phiF
	if inst2:
		inst2 = makeTwoSidedInstanton(inst2, phiF)
		inst2['phi'] -= phiF

	# Set up the spatial grid.
	dx = inst1['x'][1] - inst1['x'][0]
	if xmin is None or xmax is None:
		xmin = inst1['x'][0]*(1+tail_extension)
		if inst2:
			xmax = xsep + inst2['x'][-1]*(1+tail_extension)
		else:
			xmax = inst1['x'][-1]*(1+tail_extension)
	x = np.arange(xmin, xmax, dx)
	y = np.zeros((len(x), len(phiF)), dtype=float) + phiF
	dy = np.zeros((len(x), len(phiF)), dtype=float)
	d2y = np.zeros((len(x), len(phiF)), dtype=float)

	if inst1:
		tck = interpolate.splprep(inst1['phi'].T, u=inst1['x'], k=5, s=0)[0]
		dtck = interpolate.splprep(inst1['dphi'].T, u=inst1['x'], k=5, s=0)[0]
		i = (x >= inst1['x'][0]) & (x <= inst1['x'][-1])
		y[i] += np.array(interpolate.splev(x[i], tck, der=0)).T
		dy[i] += np.array(interpolate.splev(x[i], dtck, der=0)).T
		d2y[i] += np.array(interpolate.splev(x[i], dtck, der=1)).T
	if inst2:
		tck = interpolate.splprep(inst2['phi'].T, u=inst2['x'], s=0)[0]
		dtck = interpolate.splprep(inst2['dphi'].T, u=inst2['x'], s=0)[0]
		i = (x >= inst2['x'][0]+xsep) & (x <= inst2['x'][-1]+xsep)
		y[i] += np.array(interpolate.splev(x[i]-xsep, tck, der=0)).T
		dy[i] += np.array(interpolate.splev(x[i]-xsep, dtck, der=0)).T
		d2y[i] += np.array(interpolate.splev(x[i]-xsep, dtck, der=1)).T

	# From here on, it's the same as before....

	# At this point we have the x grid and the fields on that grid.
	# At t0=0, the time derivs are zero
	# At t0>0, we've got to be a little more careful.

	V = model.V
	dV = model.dV

	t0 = rel_t0 * inst1["x"][-1]
	if (inst2):
		t0 = min(t0, rel_t0 * inst2["x"][-1])

	Vinterp = interpolate.interp1d(x, V(y),fill_value="extrapolate", kind="cubic")
	dVinterp = interpolate.interp1d(x, dV(y).T[0],fill_value="extrapolate")
	Vmin = Vinterp(optimize.minimize(Vinterp, 0.).x)
	#yinterp = interpolate.interp1d(x, y.T[0],fill_value="extrapolate")
	dyinterp = interpolate.interp1d(x, dy.T[0],fill_value="extrapolate")
	d2yinterp = interpolate.interp1d(x, d2y.T[0], fill_value="extrapolate")
	if len(phiF) > 1:
		dV2interp = interpolate.interp1d(x, dV(y).T[1],fill_value="extrapolate")
		#y2interp = interpolate.interp1d(x, y.T[1],fill_value="extrapolate")
		dy2interp = interpolate.interp1d(x, dy.T[1],fill_value="extrapolate")
		d2y2interp = interpolate.interp1d(x, d2y.T[1],fill_value="extrapolate")
	
	plt.figure()
	plt.plot(x,Vinterp(x),'r')
	plt.yscale('log')
	plt.savefig("V_x.pdf")
	plt.figure()

    #w = [a0, a1, alpha0, da0/dx, da1/dx, dalpha0/dx]
	def diffeq(w, r):
		da0dx = w[3]
		d2a0dx2 = (3*r*w[0]**2*w[1]**2 - 4*np.pi*r*dyinterp(r)**2*w[0]**2*w[2]**2 
		- 4*np.pi*r*dy2interp(r)**2*w[0]**2*w[2]**2
		- 8*np.pi*Vinterp(r)*r*w[0]**4*w[2]**2 - 4*w[0]*w[2]**2*w[3]
		+ r*w[2]**2*w[3]**2)/(2*r*w[0]*w[2]**2)
		dalphdx = w[5]
		d2alphdx2 = 1/(r*w[0]**2)*(-8*np.pi*r*dyinterp(r)**2*w[0]**2*w[2] 
		- 8*np.pi*r*dy2interp(r)**2*w[0]**2*w[2] + w[0]*w[2]*w[3] 
		+ 2*r*w[2]*w[3]**2 + w[0]**2*w[5] + 2*r*w[0]*w[3]*w[5] 
		- r*w[0]*w[2]*d2a0dx2)
		da1dx = w[4]
		d2a1dx2 = (-w[1]**2*w[2]*w[3]**2 + w[0]**2*w[2]*w[4]**2 
		+ w[0]*w[1]**2*w[3]*w[5] - w[0]**2*w[1]*w[4]*w[5] 
		+ w[0]*w[1]**2*w[2]*d2a0dx2 + w[0]**2*w[1]**2*d2alphdx2)/(w[0]**2*w[1]*w[2])
		dwdx = [da0dx, da1dx, dalphdx, d2a0dx2, d2a1dx2, d2alphdx2]
		return dwdx

	ds_radius = np.sqrt(3/(8*np.pi*Vmin))[0]
	ds_radius_ext = np.sqrt(3/(8*np.pi*V(phiF)))
	print "dS radius interior: {}".format(ds_radius)
	print "dS radius exterior: {}".format(ds_radius_ext)
	winit = [1.0, 1/ds_radius, 1.0, 0.0, 0.0, 0.0]
	wsoln = odeint(diffeq, winit, x)

	a0interp = interpolate.UnivariateSpline(x, wsoln[:,0], k=4, s=0)
	a0interpx = a0interp.derivative()
	a0interpxx = a0interpx.derivative()
	a1interp = interpolate.UnivariateSpline(x, wsoln[:,1], k=4, s=0)
	a1interpx = a1interp.derivative()
	a1interpxx = a1interpx.derivative()	
	alpha0interp = interpolate.UnivariateSpline(x, wsoln[:,2], k=4, s=0)
	alpha0interpx = alpha0interp.derivative()
	alpha0interpxx = alpha0interpx.derivative()
	alpha1arr = (wsoln[:,1]/wsoln[:,0])
	alpha1interp = interpolate.UnivariateSpline(x, alpha1arr, k=4, s=0)

	K3scalar = -3*a1interp(x)/(a0interp(x)*alpha0interp(x))
	R3scalar = (2*x*a0interpx(x)**2-4*a0interp(x)*(2*a0interpx(x) + x*a0interpxx(x)))/(x*a0interp(x)**4)
	R4scalar = 1/(x*a0interp(x)**4*alpha0interp(x)**3)*(2*x*alpha0interp(x)**3*a0interpx(x)**2 
	- 2*a0interp(x)*alpha0interp(x)**2*(x*a0interpx(x)*alpha0interpx(x) + 2*alpha0interp(x)*(2*a0interpx(x) + x*a0interpxx(x))) 
	- 2*a0interp(x)**2*alpha0interp(x)*(alpha0interp(x)*(2*alpha0interpx(x) + x*alpha0interpxx(x)) - 3*x*a1interp(x)**2) 
	+ a0interp(x)**3*(-6*x*a1interp(x)*alpha1interp(x) + 6*x*alpha0interp(x)*(a1interp(x)**2/a0interp(x) + a1interp(x)*alpha1interp(x)/alpha0interp(x))))
	
	"""
	ds_coord = ds_radius
	ds_coord_ext = ds_radius_ext
	for i in range(len(x)):
		if abs(wsoln[:,0][i]*x[i]-ds_radius_ext)/ds_radius_ext < 1e-2:
			ds_coord_ext = x[i]
			print 'r={}, i={}, R={}'.format(x[i], i, wsoln[:,0][i]*x[i])
			continue
		if abs(wsoln[:,0][i]*x[i]-ds_radius)/ds_radius < 1e-2:
			ds_coord = x[i]
			break
	"""

	plt.figure()
	plt.plot(x, a0interp(x), 'b', label=r"Calculated $a$")
	plt.plot(x, alpha0interp(x), 'g', label=r"Calculated $\alpha$")
	#plt.axvline(x=ds_coord, color='black', label=r"Interior dS rad")
	#plt.axvline(x=ds_coord_ext, color='gray', label=r"Exterior ds rad")
	plt.legend()
	plt.xlabel("Coordinate r")
	plt.ylim(-0.0,2.0)
	plt.savefig("initial.pdf")

	plt.figure()
	plt.plot(x, x*a0interp(x), 'b')
	plt.title("Radius of curvature")
	#plt.axvline(x=ds_coord, color='black', label=r"Interior dS rad")
	#plt.axvline(x=ds_coord_ext, color='gray', label=r"Exterior ds rad")
	plt.legend()
	plt.xlabel("Coordinate r")
	plt.savefig("rad_curvature.pdf")

	plt.figure()
	plt.plot(x, R3scalar, 'r', label=r"Ricci 3 scalar curvature")
	plt.plot(x, 0*np.ones_like(x), 'r--', label=r"Ricci 3 scalar curvature dS")
	plt.plot(x, R4scalar, 'g', label=r"Ricci 4 scalar curvature")
	plt.plot(x, 12/ds_radius**2*np.ones_like(x), 'g--', label=r"Ricci 4 scalar curvature dS in")
	plt.plot(x, 12/ds_radius_ext**2*np.ones_like(x), 'b--', label=r"Ricci 4 scalar curvature dS out")
	#plt.axvline(x=ds_coord, color='black', label="Interior dS rad")
	#plt.axvline(x=ds_coord_ext, color='gray', label="Exterior dS rad")
	plt.legend()
	plt.ylim(-100/ds_radius_ext**2, 100/ds_radius_ext**2)
	plt.xlabel("Coordinate r")
	plt.savefig("ricci_scalars.pdf")

	plt.figure()
	plt.plot(x, K3scalar, 'b', label=r"Extrinsic scalar curvature")
	plt.plot(x, -3/ds_radius*np.ones_like(x), 'b--', label=r"Extrinsic scalar curvature dS in")
	plt.plot(x, -3/ds_radius_ext*np.ones_like(x), 'g--', label=r"Extrinsic scalar curvature dS out")
	#plt.axvline(x=ds_coord, color='black', label="Interior dS rad")
	#plt.axvline(x=ds_coord_ext, color='gray', label="Exterior dS rad")
	plt.legend()
	plt.xlabel("r")
	plt.savefig("ext_scalar.pdf")

	if len(phiF) > 1:
		phi2 = np.array((wsoln[:,2]*(-x*dVinterp(x)*wsoln[:,0]**3*wsoln[:,2] 
		+ x*dyinterp(x)*wsoln[:,2]*wsoln[:,3] + wsoln[:,0]*((x*d2yinterp(x) 
		+ 2*dyinterp(x))*wsoln[:,2] + x*dyinterp(x)*wsoln[:,5])), 
		wsoln[:,2]*(-x*dV2interp(x)*wsoln[:,0]**3*wsoln[:,2] 
		+ x*dy2interp(x)*wsoln[:,2]*wsoln[:,3] + wsoln[:,0]*((x*d2y2interp(x) 
		+ 2*dy2interp(x))*wsoln[:,2] + x*dy2interp(x)*wsoln[:,5]))))
	else:
		phi2 = np.array()
	phi2T = phi2.T
	if len(phiF)<=1:
		a0 = 0
		a1 = 0
		a2 = 0
	else:
		a0 = np.array(wsoln[:, 0])
		a1 = np.array(wsoln[:, 1])
		a2 = 0
		alpha0 = np.array(wsoln[:, 2])
		alpha1 = np.array(alpha1arr)
		#alpha1 = np.array(0*wsoln[:,1])

	plt.figure()
	plt.plot(x,a1, 'b', label='a dot')
	plt.plot(x,alpha1,'g', label='alpha dot')
	plt.plot(x, a1/(a0*alpha0), 'k', label='a dot = # * a * alpha')
	plt.legend()
	plt.savefig("initial_time_deriv.pdf")

	N = len(phiF)
	Y = np.empty((len(x), N*2+2))
	
	Y[:, :N] = y + t0*t0*phi2T # phi
	Y[:, N:2*N] = 2*t0*phi2T # Pi = (dphi/dN) * (a/alpha) ~ dphi/dN
	#Y[:, -3] = x + t0*a1*x # b
	Y[:, -2] = alpha0 + t0*alpha1 # alpha
	Y[:, -1] = a0 + t0*a1 # a

	return t0,x,Y

        

class monitorFunc1D(object):
	"""
	A simple monitor function for use with single-field potentials.

	The calculated grid density is :math:`m(x) \\propto d\\Pi/dx`,
	with the proportionaly such that there are the specified number
	of points per bubble wall.
	"""
	def __init__(self, pnts_per_wall, min_density, num_walls):
		self.pnts_per_wall = pnts_per_wall
		self.min_density = min_density
		self.num_walls = num_walls

	def __call__(self, t,x,y):
		Pi = y[:,1]
		dPidx = deriv14(Pi, x)
		m = abs(dPidx)
		m_mid = (m[1:] + m[:-1])*0.5
		dx = x[1:] - x[:-1]
		total_pnts = np.sum(m_mid * dx)
		m *= self.pnts_per_wall * self.num_walls / total_pnts
		m[m<self.min_density] = self.min_density
		return m

class monitorFunc2D(object):
	"""
	A simple monitor function for use with multi-field potentials.

	The calculated grid density is :math:`m(x) \\propto d\\Pi/dx`,
	with the proportionaly such that there are the specified number
	of points per bubble wall.
	"""
	def __init__(self, pnts_per_wall, min_density, num_walls):
		self.pnts_per_wall = pnts_per_wall
		self.min_density = min_density
		self.num_walls = num_walls

	def __call__(self, t,x,y):
		Pi1 = y[:,2]
		dPi1dx = deriv14(Pi1, x)
		m1 = abs(dPi1dx)
		m1_mid = (m1[1:] + m1[:-1])*0.5
		dx = x[1:] - x[:-1]
		total_pnts1 = np.sum(m1_mid * dx)
		m1 *= self.pnts_per_wall * self.num_walls / total_pnts1
		m1[m1<self.min_density] = self.min_density
		Pi2 = y[:,3]
		dPi2dx = deriv14(Pi2, x)
		m2 = abs(dPi2dx)
		m2_mid = (m2[1:] + m2[:-1])*0.5
		total_pnts2 = np.sum(m2_mid * dx)
		m2 *= self.pnts_per_wall * self.num_walls / total_pnts2
		m2[m2<self.min_density] = self.min_density
		return np.maximum(m1,m2)


def runModelFromInstanton(
		model, inst1, inst2, phiF, xsep=1.0, 
		t0_rel=0.001, tfix=7.0, tmax=50.0, truncation=0.95,
		time_slice_params = {}):
	"""
	Runs the simulation for the given model and instanton data. 

	The recorded time step is not constant, but varies such that there are 
	more steps where the shock is briefest and the geodesic goes through 
	it most quickly.

	The user must still set the simulation's monitor function, file parameters
	(other than *tout*), and integration parameters (or leave them at their
	default values).

	Parameters
	----------
	model : ModelObject
		The model to be used in the simulation. Should have functions *V* and
		*dV*.
	inst1 : instanton object
		The observer instanton centered at *x=0*.
		Should be a dictionary-like object with keys 'r', 'phi', and 'dphi', 
		where the 'r' array has length *nx* and the phi and dphi arrays have 
		shape *(nx, nfields)*.
	inst2 : instanton object
		The collision instanton centered at *x=xsep*.
	phiF : array of length *nfields*
		The false-vacuum field value.
	xsep : float, optional
	t0_rel : float, optional
		Initial time variable relative to the size of the observer instanton.
	tfix : float, optional
		Stop the simulation once it reaches *tfix* and fix the boundaries
		to exclude the bubble walls.
	tmax : float, optional
		Final stopping point for the simulation after *tfix*.
	truncation : float, optional
		Amount of the simulation to include inside of the bubble walls.
	time_slice_params : dictionary, optional
		Optional parameters to pass to :func:`timeSlicesForShockParams`. 
		To get uniform time outputs, set ``no_collision_step = 0.0`` and 
		*minstep* to the desired output time step.

	Returns
	-------
	t : float
	x : array
		Final simulation grid
	y : array
		Field values along the final time slice
	"""
	phiF = np.array(phiF).ravel()
	nfields = len(phiF)

	# First, find out how thick the bubble walls are.
	r_inner1, r_outer1 = wallPositionFromInstanton(inst1['r'], inst1['phi'])
	if inst2:
		r_inner2, r_outer2 = wallPositionFromInstanton(inst2['r'], inst2['phi'])
	else:
		r_inner2, r_outer2 = r_inner1, r_outer1
		
	# Now find out how dense the time slices need to be
	tout = timeSlicesForShockParams(
		xsep, r_outer1, r_inner2, r_outer2, tmax, **time_slice_params)	
	if inst2 is None:
		xsep = 0.0
	simulation.setFileParams(tout=tout)
							
	print "\nGetting initial data..."
	t0,x,y = calcInitialDataFromInst(model, inst1, inst2, phiF, xsep, t0_rel)

	print "\nRunning simulation..."
	simulation.setModel(model)
	t, x, y = simulation.runCollision(x,y,t0,tfix)
	
	if (t < tfix*.9999):
		print "Didn't reach the end of the simulation. Aborting."
		return t,x,y
		
	if (tmax <= tfix):
		return t,x,y
	
	print "\nReached tfix."
	print "Fixing the bounds of the simulation (excluding bubble walls)."
	bubble_radius = 2*np.arctan(np.tanh(t/2.0))
	dx = bubble_radius * (1-truncation)
	xmin, xmax = -bubble_radius+dx, bubble_radius + xsep - dx
	i = (x < xmax) & (x > xmin)
	x,y = x[i], y[i]
	#if (inst2 != inst1 and xsep > 0):
	if (inst2 is not None and 
			(inst1['phi'].shape != inst2['phi'].shape 
				or (inst1['phi'] != inst2['phi']).any()) and xsep > 0):
		# There will still be a wall between them. Need to find it.
		alpha = y[:, -2]
		dalpha = abs(deriv14(alpha, x))
		xmax = x[dalpha==max(dalpha)][0] - dx
		i = (x < xmax) & (x > xmin)
		x,y = x[i], y[i]
	
	print "\nRunning simulation with fixed bounds..."
	tfin, xfin, yfin = simulation.runCollision(x,y,t,tmax, 
		growBounds=False, overwrite=False)
	return tfin, xfin, yfin


def runModelFromInstanton_fixedgrid(model, inst1, inst2, phiF,
		xsep=1.0, xdensity=100.0, t0_rel=0.001, tmax=5.0):
	"""
	Run the simulation on a constant grid. The monitor function gets computed
	automatically.

	Parameters
	----------
	model : ModelObject
		The model to be used in the simulation. Should have functions *V* and
		*dV*.
	inst1 : instanton object
		The observer instanton centered at *x=0*.
		Should be a dictionary-like object with keys 'r', 'phi', and 'dphi', 
		where the 'r' array has length *nx* and the phi and dphi arrays have 
		shape *(nx, nfields)*.
	inst2 : instanton object
		The collision instanton centered at *x=xsep*.
	phiF : array of length *nfields*
		The false-vacuum field value.
	xsep : float
	t0_rel : float
		Initial time variable relative to the size of the observer instanton.
	xdensity : float
		Density of grid points.

	Returns
	-------
	t : float
	x : array
		Final simulation grid
	y : array
		Field values along the final time slice
	"""
	phiF = np.array(phiF).ravel()
	nfields = len(phiF)
		
	if inst2 is None:
		xsep = 0.0
	def monitorFunc(t,x,y, min_density = xdensity):
		return np.ones_like(x)*xdensity
	simulation.setMonitorCallback(monitorFunc)
	simulation.setModel(model)

	print "\nGetting initial data..."
	t0,x,y = calcInitialDataFromInst(
		model, inst1, inst2, phiF, xsep, t0_rel, 
		xmin=-np.pi/2, xmax=xsep+np.pi/2)

	print "\nRunning simulation..."
	t, x, y = simulation.runCollision(x,y,t0,tmax, growBounds=False)
	
	if (t < tmax*.9999):
		print "Didn't reach the end of the simulation."

	return t,x,y
