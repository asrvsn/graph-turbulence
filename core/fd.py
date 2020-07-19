''' Finite difference methods for PDEs, for comparison with graph solutions ''' 

import numpy as np

from utils import *

def fd_diffusion(dx: tuple, mx: tuple, dirichlet_bc: dict={}, neumann_bc: dict={}, alpha=1., corner_once=True):
	''' Diffusion PDE using central difference discretization of spatial derivatives 
	Args:
		dx: tuple of differences (dx1, dx2, ... dxn)
		mx: tuple of # points (mx1, mx2, ... mxn)
		dirichlet_bc: dict mapping boundary nodes to dirichlet conditions 
		neumann_bc: dict mapping boundary nodes to neumann conditions 
		corner_once: only apply Neumann conditions at corners once
	'''
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
				neumann_off = False # Use a flag to avoid over-applying Neumann conditions at corners
				for c_i in range(d):
					above = map_nd_to_1d(mx, (*coord[:c_i], coord[c_i]+1, *coord[c_i+1:]))
					below = map_nd_to_1d(mx, (*coord[:c_i], coord[c_i]-1, *coord[c_i+1:]))
					if coord[c_i] == 0:
						assert coord in neumann_bc, f'Boundary condition at {coord} not specified'
						ghost = 2*dx[c_i]*(0. if neumann_off else neumann_bc[coord]) - u[above]
						Lu += (ghost - 2*u[i] + u[above]) / (dx[c_i] ** 2)
						neumann_off = corner_once
					elif coord[c_i] == mx[c_i] - 1:
						assert coord in neumann_bc, f'Boundary condition at {coord} not specified'
						ghost = 2*dx[c_i]*(0. if neumann_off else neumann_bc[coord]) - u[below]
						Lu += (u[below] - 2*u[i] + ghost) / (dx[c_i] ** 2)
						neumann_off = corner_once
					else:
						Lu += (u[below] - 2*u[i] + u[above]) / (dx[c_i] ** 2)
				dudt[i] = alpha*Lu
		return dudt
	return f
