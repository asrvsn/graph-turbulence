''' Class for specifying, running, and visualizing differential equations on graphs ''' 

import networkx as nx
import numpy as np
import scipy.integrate
import dill as pickle # For pickling lambdas
from typing import Callable, List, Tuple, Union, Any
from networkx.readwrite.json_graph import node_link_data, node_link_graph
import ujson
import pdb
from matplotlib.colors import rgb2hex
from bokeh.plotting import figure, from_networkx
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker
from bokeh.models.glyphs import Oval, MultiLine
from bokeh.transform import linear_cmap
import colorcet as cc
from enum import Enum
from abc import ABC, abstractmethod

from utils import *

''' Basic types ''' 

class Domain(Enum):
	Vertex = 0
	Edge = 1
	# Can extend...
	# Face = 2

Vertex = Any
Edge = Tuple[Vertex, Vertex]
# Face = Tuple[Vertex, ...]
GeoObject = Union[Vertex, Edge]

class Observable(ABC):
	''' A real-valued function defined on a graph 
	Note: abstract base class, not intended to be instantiated.
	''' 

	def __init__(self, G: nx.Graph):
		self.G = G
		self.init_domain()
		self.ode = None
		self.y = np.zeros(len(self.domain))
		self.t = 0.
		self.dirichlet_values = dict()
		self.neumann_values = dict()
		self.plot = None

	@abstractmethod
	def init_domain(self):
		pass

	''' ODE ''' 

	def set_ode(self, f: Callable[float], order: int=1, solver: str='dopri5', **solver_args):

		self.ode = scipy.integrate.ode(lambda t, y: f(t)).set_integrator(solver, **solver_args)
		self.order = order

	''' Initial & Boundary Conditions ''' 

	def set_initial(self, t0: float=0., y0: Any=0.):
		self.t = t0
		if isinstance(y0, Iterable):
			self.y = np.array(y0)
		elif isintance(y0, repeat):
			self.y = np.array(take(len(self), y0))
		else:
			self.y = np.full(len(self), y0)
		if self.ode is not None:
			self.ode.set_initial_value(self.y, self.t)

	def set_boundary(self, dirichlet_values: Dict[GeoObject, float]={}, neumann_values: Dict[GeoObject, float]={}):
		intersect = dirichlet_values.keys() & neumann_values.keys()
		assert len(intersect) == 0, f'Dirichlet and Neumann conditions overlap on {intersect}'
		self.dirichlet_values = dirichlet_values
		self.neumann_values = neumann_values
		y = replace(self.y, [self.domain[k] for k in dirichlet_values.keys()], list(dirichlet_values.values()))
		self.set_initial(t0=self.t, y0=y)

	''' Integration ''' 

	def step(self, dt: float):
		if self.ode is not None:
			self.ode.integrate(self.t + dt)

	def measure(self):
		if self.ode is not None:
			self.t = self.ode.t
			self.y = self.ode.y
		if self.plot is not None:
			self.render()
		return self.y

	''' Operators ''' 

	@abstractmethod
	def partial(self, x1: GeoObject, x2: GeoObject) -> float:
		pass

	@abstractmethod
	def grad(self) -> np.ndarray:
		pass

	@abstractmethod
	def div(self) -> np.ndarray:
		pass

	@abstractmethod
	def laplacian(self) -> np.ndarray: 
		pass

	''' Builtins ''' 

	def __len__(self):
		return len(self.domain)

	def __call__(self, x: GeoObject):
		return self.y[self.domain[x]]

	''' Rendering ''' 

	def create_plot(self):
		''' Create plot for rendering with Bokeh ''' 
		G = nx.convert_node_labels_to_integers(self.G) # Bokeh cannot handle non-primitive node keys (eg. tuples)
		layout = nx.spring_layout(G, scale=0.9, center=(0,0), iterations=500, seed=1)
		plot = figure(x_range=(-1.1,1.1), y_range=(-1.1,1.1), tooltips=[])
		plot.axis.visible = None
		plot.xgrid.grid_line_color = None
		plot.ygrid.grid_line_color = None
		renderer = from_networkx(G, layout)
		plot.renderers.append(renderer)
		self.plot = plot
		return plot

	@abstractmethod
	def render(self):
		''' Render current values to the plot ''' 
		pass


class VertexObservable(Observable):
	def init_domain(self):
		self.domain = dict(zip(self.G.nodes(), itertools.count(0)))

	def weight(edge: Edge): float:
		return 1.0 # TODO

	def partial(edge: Edge) -> float:
		return np.sqrt(self.weight(edge)) * self(edge[1]) - self(edge[0]) 

	def grad(self) -> np.ndarray:
		return np.array([self.partial(edge) for edge in self.G.edges()])

	def div(self) -> np.ndarray:
		raise Exception('implement me')

	def laplacian_at(self, x: Vertex) -> float:
		''' Compute Laplacian that solves Neumann problem using phantom-node method '''
		ret = sum([self.weight((x, n)) * (self(n) - self(x)) for n in self.G.neighbors(x)])
		if x in self.neumann_values:
			ret += self.neumann_values[x]
		return ret

	def laplacian(self) -> np.ndarray:
		return np.array([self.laplacian_at(x) for x in self.G.nodes()])

	def create_plot(self):
		super().create_plot()
		palette = cc.fire
		lo, hi = 0., 1.
		self.plot.renderers[0].node_renderer.data_source.data['nodes'] = self.y
		self.plot.renderers[0].node_renderer.glyph = Oval(height=0.08, width=0.08, fill_color=linear_cmap('nodes', palette, lo, hi))
		cbar = ColorBar(color_mapper=LinearColorMapper(palette=palette, low=lo, high=hi), ticker=BasicTicker(), title='Node')
		self.plot.add_layout(cbar, 'right')

	def render(self):
		self.plot.renderers[0].node_renderer.data_source.data['nodes'] = self.y


class EdgeObservable(Observable):
	def init_domain(self):
		self.domain = dict(zip(self.G.edges(), itertools.count(0)))

	def partial(edge: Edge) -> float:
		raise Exception('implement me')

	def grad(self) -> np.ndarray:
		raise Exception('implement me')

	def div_at(self, x: Vertex) -> float:
		return sum([np.sqrt(self.weight(x, n)) * (self(n) - self(x)) for n in self.G.neighbors(x)])

	def div(self) -> np.ndarray:
		return np.array([self.div_at(x) for x in self.G.nodes()])

	def create_plot(self):
		super().create_plot()
		palette = cc.gray
		lo, hi = 0., 1.
		self.plot.renderers[0].edge_renderer.data_source.data['edges'] = self.y
		self.plot.renderers[0].edge_renderer.glyph = MultiLine(line_color=linear_cmap('edges', palette, lo, hi), line_width=5)
		cbar = ColorBar(color_mapper=LinearColorMapper(palette=palette, low=lo, high=hi), ticker=BasicTicker(), title='Edge')
		self.plot.add_layout(cbar, 'right')

	def render(self):
		self.plot.renderers[0].edge_renderer.data_source.data['edges'] = self.y


''' Main class ''' 

class GraphDiffEq: 
	def __init__(
			self, G: nx.Graph, v0: np.ndarray, l0: np.ndarray, dv_dt: Callable, dl_dt: Callable, 
			t0: float=0., desc: str=None, node_palette=cc.fire, edge_palette=cc.gray, solver: str='dopri5', **solver_args
		):
		''' General class for running initial+boundary value problems on graphs.
		Args:
			G: graph
			v0: initial values for vertices
			l0: initial values for edges
			dv_dt: difference equation for vertices taking args (time, vertex_values, edge_values)
			dl_dt: difference equation for edges taking args (time, vertex_values, edge_values)
			t0: start time
			desc: description (will be displayed if rendered)
			node_palette: color palette for node values 
			edge_palette: color palette for edge values
			solver: scipy.integrate.ode solver designation (default: 'dopri5' aka Runge-Kutta 4/5)
			**solver_args: additional arguments to be passed to integrator
		'''
		# Consistency checks
		assert len(G) == len(v0) == len(dv_dt(t0, v0, l0)), 'Vertex dimensions do not match'
		assert len(G.edges) == len(l0) == len(dl_dt(t0, v0, l0)), 'Edge dimensions do not match'
		self.G = G

		# Initialize integrator (allow replacement of difference values at runtime)
		def dv_dt_with_replacement(t: float, v: np.ndarray, l: np.ndarray, replace_at: list, replace_with: list):
			return replace(dv_dt(t, v, l), replace_at, replace_with)
		def dl_dt_with_replacement(t: float, l: np.ndarray, v: np.ndarray, replace_at: list, replace_with: list):
			return replace(dl_dt(t, v, l), replace_at, replace_with)

		self.v_ode = scipy.integrate.ode(dv_dt_with_replacement).set_integrator(solver, **solver_args).set_initial_value(v0, t0).set_f_params(l0, [], [])
		self.l_ode = scipy.integrate.ode(dl_dt_with_replacement).set_integrator(solver, **solver_args).set_initial_value(l0, t0).set_f_params(v0, [], [])
		self.vertex_bc = ([], []) # Substitions in vertex difference equation
		self.edge_bc = ([], []) # Substitions in edge difference equation

		# Rendering
		self.plot = None
		self.desc = desc
		self.node_palette = node_palette
		self.edge_palette = edge_palette
		self.node_lo, self.node_hi = 0., 1.
		self.edge_lo, self.edge_hi = 0., 1.

	@property 
	def t(self):
		''' Current time '''
		return self.v_ode.t

	@property
	def v(self):
		''' Current vertex values '''
		return self.v_ode.y

	@property 
	def l(self):
		''' Current edge values ''' 
		return self.l_ode.y

	def set_vertex_boundary(self, dirichlet: dict={}, neumann: dict={}):
		''' Set boundary conditions for vertex-valued function. 
		Args:
			dirichlet: map of nodes to f(t), dictating their values at time t
			neumann: map of nodes to g(t), dictating the value of the gradient at the node at time t
		''' 
		# TODO: ensure consistency of edges & nodes here..
		assert len(dirichlet.keys() & neumann.keys()) == 0, 'Dirichlet and Neumann conditions cannot overlap'
		nodes = list(self.G.nodes)
		d_vals = list(dirichlet.values())
		n_vals = list(neumann.values())
		d_idx = [nodes.index(n) for n in dirichlet.keys()]
		n_idx = [nodes.index(n) for n in neumann.keys()]
		# Modify difference equation with boundary conditions -- zero difference for dirichlet, specified difference for neumann
		replace_at = d_idx + n_idx 
		replace_with = np.zeros(len(d_idx)).tolist() + n_vals 
		self.vertex_bc = (replace_at, replace_with)
		# Reset the current value per Dirichlet conditions
		self.v_ode.set_initial_value(replace(self.v, d_idx, d_vals), self.t)

	def set_edge_boundary(self, dirichlet: dict={}, neumann: dict={}):
		''' Set boundary conditions on edges ''' 
		# TODO: ensure consistency of edges & nodes here..
		assert len(dirichlet.keys() & neumann.keys()) == 0, 'Dirichlet and Neumann conditions cannot overlap'
		edges = list(self.G.edges)
		d_vals = list(dirichlet.values())
		n_vals = list(neumann.values())
		d_idx = [edges.index(n) for n in dirichlet.keys()]
		n_idx = [edges.index(n) for n in neumann.keys()]
		# Modify difference equation with boundary conditions -- zero difference for dirichlet, specified difference for neumann
		replace_at = d_idx + n_idx 
		replace_with = np.zeros(len(d_idx)).tolist() + n_vals 
		self.edge_bc = (replace_at, replace_with)
		# Reset the current value per Dirichlet conditions
		self.l_ode.set_initial_value(replace(self.l, d_idx, d_vals), self.t)

	def set_vertex_limits(self, lower: float=-float('inf'), upper: float=float('inf'), which: list=[]):
		pass # TODO

	def set_edge_limits(self, lower: float=-float('inf'), upper: float=float('inf'), which: list=[]):
		pass # TODO

	def step(self, dt: float):
		''' Solve system forward for dt time ''' 
		self.v_ode.set_f_params(self.l, self.vertex_bc[0], self.vertex_bc[1]) # Pass edge values, boundary conditions to vertex function
		self.v_ode.integrate(self.t + dt)
		self.l_ode.set_f_params(self.v, self.edge_bc[0], self.edge_bc[1]) # Pass vertex values, boundary conditions to edge function
		self.l_ode.integrate(self.t + dt)
		if self.plot is not None:
			self._render()

	def create_plot(self): 
		''' Create plot for rendering with Bokeh ''' 
		G = nx.convert_node_labels_to_integers(self.G) # Bokeh cannot handle non-primitive node keys (eg. tuples)
		n_v = len(G)
		n_e = len(G.edges())
		tooltips = [('value', '@nodes')]
		layout = nx.spring_layout(G, scale=0.9, center=(0,0), iterations=500, seed=1)
		plot = figure(title=self.desc, x_range=(-1.1,1.1), y_range=(-1.1,1.1), tooltips=tooltips, aspect_ratio=1.2)
		plot.axis.visible = None
		plot.xgrid.grid_line_color = None
		plot.ygrid.grid_line_color = None
		renderer = from_networkx(G, layout)
		renderer.node_renderer.data_source.data['nodes'] = self.v.tolist()
		renderer.node_renderer.glyph = Oval(height=0.08, width=0.08, fill_color=linear_cmap('nodes', self.node_palette, self.node_lo, self.node_hi))
		renderer.edge_renderer.data_source.data['edges'] = self.l.tolist()
		renderer.edge_renderer.glyph = MultiLine(line_color=linear_cmap('edges', self.edge_palette, self.edge_lo, self.edge_hi), line_width=5)
		# # TODO: render edge direction w/ arrows somehow
		# using: https://discourse.bokeh.org/t/hover-over-tooltips-on-network-edges/2439/7
		plot.renderers.append(renderer)
		node_cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.node_palette, low=self.node_lo, high=self.node_hi), ticker=BasicTicker(), title='Node')
		edge_cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.edge_palette, low=self.edge_lo, high=self.edge_hi), ticker=BasicTicker(), title='Edge')
		plot.add_layout(node_cbar, 'right')
		plot.add_layout(edge_cbar, 'right')
		self.plot = plot
		return plot

	def _render(self):
		''' Draw glyphs based on current graph values ''' 
		self.plot.renderers[0].node_renderer.data_source.data['nodes'] = self.v
		self.plot.renderers[0].edge_renderer.data_source.data['edges'] = self.l
		# # TODO: render edge direction w/ arrows somehow

	def reset(self):
		pass # TODO


class LatticeDiffEq(GraphDiffEq):
	''' Differential equations on 2D lattices ''' 
	pass
