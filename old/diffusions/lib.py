import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm
from scipy.integrate import solve_ivp
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

from utils import *

def constrained_laplacian(G: nx.Graph, dirichlet_bc: dict={}, neumann_bc: dict={}):
	''' Returns graph Laplacian which is (a) unconstrained or solves (b) Dirichlet problem or (c) Neumann problem ''' 
	assert len(dirichlet_bc.keys() & neumann_bc.keys()) == 0, 'Dirichlet and Neumann conditions cannot overlap'

	# Using standard (unweighted) discrete Laplacian
	L = csc_matrix(nx.laplacian_matrix(G))

	for (i, node) in enumerate(G.nodes):
		if node in dirichlet_bc.keys():
			L[i,:] = 0
		elif node in neumann_bc.keys():
			raise Exception('Neumann conditions not implemented yet') # TODO

	return -L


def match_ic_bc(u0: np.ndarray, keyfunc: Callable, dirichlet_bc: dict={}, neumann_bc: dict={}):
	''' Match initial conditions w/ specified boundary conditions ''' 
	for i in range(len(u0)):
		k = keyfunc(i)
		if k in dirichlet_bc.keys():
			u0[i] = dirichlet_bc[k]
	return u0


def solve_exact(G: nx.Graph, u0: np.ndarray, dirichlet_bc: dict={}, neumann_bc: dict={}, alpha=1.):
	''' Returns exact solution for heat eq. on a graph as Callable
	Args:
		G: graph
		u0: initial condition
		dirichlet_bc: dict mapping boundary nodes to dirichlet conditions 
		neumann_bc: dict mapping boundary nodes to neumann conditions 
	'''
	assert u0.shape[0] == len(G.nodes), 'Incorrect number of initial conditions'
	nodelist = list(G.nodes)
	L = constrained_laplacian(G, dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
	u0 = match_ic_bc(u0, lambda i: nodelist[i], dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
	u = lambda t: u0@(expm(alpha*L*t).T)
	return u


def solve_numeric(G: nx.Graph, u0: np.ndarray, dirichlet_bc: dict={}, neumann_bc: dict={}, dt: float=1e-3, T_max: float=1.0, alpha=1., method='RK45'):
	''' Returns numerically integrated solution for heat eq. on a graph as Callable (using scipy, 'LSODA')
	Args:
		G: graph
		u0: initial condition
		dirichlet_bc: dict mapping boundary nodes to dirichlet conditions 
		neumann_bc: dict mapping boundary nodes to neumann conditions 
		dt: time discretization (i.e. when calling u(), this is the resolution)
		T_max: end time step to solve out to
	'''
	assert u0.shape[0] == len(G.nodes), 'Incorrect number of initial conditions'
	nodelist = list(G.nodes)
	L = constrained_laplacian(G, dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
	f = lambda t, u: alpha*u@L.T
	u0 = match_ic_bc(u0, lambda i: nodelist[i], dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
	sol = solve_ivp(f, (0., T_max), u0, max_step=dt, method=method)
	def u(t):
		i = np.searchsorted(sol.t, t, side='left')
		return sol.y[:,i]
	return u


def solve_lattice(dx: tuple, mx: tuple, u0: np.ndarray, dirichlet_bc: dict={}, neumann_bc: dict={}, periodic_bc: bool=False, dt: float=1e-3, T_max: float=1.0, alpha=1., method='RK45'):
	''' Solve heat eq by finite difference discretization of spatial derivatives (method of lines). 
	Use for comparison with graph solutions. 
	Args:
		dx: tuple of differences (dx1, dx2, ... dxn)
		mx: tuple of # points (mx1, mx2, ... mxn)
		u0: initial condition
		dirichlet_bc: dict mapping boundary nodes to dirichlet conditions 
		neumann_bc: dict mapping boundary nodes to neumann conditions 
		periodic_bc:
		dt: time discretization (i.e. when calling u(), this is the resolution)
		T_max: end time step to solve out to

	TODOs: 
		* neumann condition
		* banded jacobian?
	'''
	assert u0.shape[0] == np.prod(mx), 'Incorrect number of initial conditions'
	assert len(dx) == len(mx), 'Dimensions of steps and extents mismatch'
	assert len(dirichlet_bc.keys() & neumann_bc.keys()) == 0, 'Dirichlet and Neumann conditions cannot overlap'
	d = len(dx) # Dimension of problem
	def f(t, u): # Finite difference ODE
		dudt = np.empty_like(u)
		for i in range(len(u)):
			coord = map_1d_to_nd(mx, i) # embedding nd coords in 1d for solve_ivp() API & to match other funcs
			# Check Dirichlet conditions
			if coord in dirichlet_bc:
				dudt[i] = 0
			# Compute discrete Laplacian respecting Neumann conditions
			else:
				Lu = 0
				for c_i in range(d):
					above = map_nd_to_1d(mx, (*coord[:c_i], coord[c_i]+1, *coord[c_i+1:]))
					below = map_nd_to_1d(mx, (*coord[:c_i], coord[c_i]-1, *coord[c_i+1:]))
					if coord[c_i] == 0:
						assert coord in neumann_bc, f'Boundary condition at {coord} not specified'
						ghost = u[above] - 2*dx[c_i]*neumann_bc[coord]
						Lu += (ghost - 2*u[i] + u[above]) / (dx[c_i] ** 2)
					elif coord[c_i] == mx[c_i] - 1:
						assert coord in neumann_bc, f'Boundary condition at {coord} not specified'
						ghost = u[below] - 2*dx[c_i]*neumann_bc[coord]
						Lu += (u[below] - 2*u[i] + ghost) / (dx[c_i] ** 2)
					else:
						Lu += (u[below] - 2*u[i] + u[above]) / (dx[c_i] ** 2)
				dudt[i] = alpha*Lu
		return dudt
	u0 = match_ic_bc(u0, lambda i: map_1d_to_nd(mx, i), dirichlet_bc=dirichlet_bc, neumann_bc=neumann_bc)
	sol = solve_ivp(f, (0., T_max), u0, max_step=dt, method=method)
	def u(t):
		i = np.searchsorted(sol.t, t, side='left')
		return sol.y[:,i]
	return u


def plot_snapshots(G: nx.Graph, u: Callable, T: float, n: int, absolute_colors=True):
	'''
	Args:
		G: graph
		u: solution
		T: time extent
		n: # snapshots
		absolute_colors: whether to set heatmap extents globally (default: True) or based on individual snapshots
	'''
	fig, axs = plt.subplots(1, n)
	pos = nx.spring_layout(G, iterations=1000)
	for i in range(n):
		ax = axs[i]
		t = 0 + i*(T/(n-1))
		vals = u(t)
		if absolute_colors:
			vmax = max(u(0)) # Assume no other driving
			nx.draw(G, pos=pos, cmap=heat_cmap, node_color=vals, ax=ax, vmin=0., vmax=vmax)
		else:
			nx.draw(G, pos=pos, cmap=heat_cmap, node_color=vals, ax=ax)
		ax.set_title(f'T={t}')


def plot_rasterized(G: nx.Graph, u: Callable, T: float):
	'''
	Args:
		G: graph
		u: solution
		T: time extent
	'''
	tspace = np.linspace(0, T, 100)
	sol = np.array(list(map(u, tspace))).squeeze()
	plt.figure(figsize=(20,40))
	sns.heatmap(sol.T, cmap=heat_cmap)
	plt.title(f'Heat equation for T={T} seconds')


def plot_rmse(sols: dict, dt: float, T: float):
	'''Plot RMSE against exact solution '''
	fig, axs = plt.subplots(1, len(sols)-1)
	tspace = np.linspace(0, T, T/dt)
	i = 0
	for key in sols.keys():
		if key != 'exact':
			err = np.array([rms(sols['exact'](t) - sols[key](t)) for t in tspace])
			axs[i].plot(tspace, err)
			axs[i].set_title(key)
			i += 1



