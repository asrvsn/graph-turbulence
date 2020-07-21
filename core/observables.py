''' Class for specifying, running, and visualizing differential equations on graphs ''' 

import networkx as nx
import numpy as np
import scipy.integrate
import dill as pickle # For pickling lambdas
from typing import Callable, List, Tuple, Union, Any, Dict
from networkx.readwrite.json_graph import node_link_data, node_link_graph
import ujson
import pdb
from matplotlib.colors import rgb2hex
from bokeh.plotting import figure, from_networkx
from bokeh.models import ColorBar, LinearColorMapper, BasicTicker, HoverTool
from bokeh.models.glyphs import Oval, MultiLine
from bokeh.transform import linear_cmap
import colorcet as cc
from itertools import count
from enum import Enum
from abc import ABC, abstractmethod

from utils import *

''' Basic types ''' 

Vertex = Any
Edge = Tuple[Vertex, Vertex]
Face = Tuple[Vertex, ...]
GeoObject = Union[Vertex, Edge]

class Observable(ABC):
	''' A real-valued function defined on a graph domain
	Note: abstract base class, not intended to be instantiated.
	''' 

	def __init__(self, G: nx.Graph, desc: str='', weight_key: str=None, default_weight=1.0):
		self.G = G
		self.init_domain()
		self.n = len(self.domain)
		self.ode = None
		self.y = np.zeros(len(self.domain))
		self.y0 = lambda _: 0.
		self.t = 0.
		self.t0 = 0.
		self.init_kwargs = {}
		self.dirichlet_values = dict()
		self.neumann_values = dict()
		self.plot = None
		self.desc = desc
		self.set_render_params()
		self.w_key = weight_key
		self.default_weight = default_weight

	@abstractmethod
	def init_domain(self):
		pass

	def weight(self, e: Edge) -> float:
		if self.w_key is None:
			return self.default_weight
		else:
			return self.G[e[0]][e[1]][self.w_key]

	''' ODE ''' 

	def set_ode(self, f: Callable[[float], np.ndarray], order: int=1, solver: str='dopri5', **solver_args):

		n = len(self)

		def g(t: float, y: np.ndarray, fixed_vals: list):
			dydt = np.zeros_like(y)
			for i in range(order-1):
				dydt[n*i:n*(i+1)] = y[n*(i+1):n*(i+2)]
			dydt[n*(order-1):] = replace(f(t), fixed_vals, [0.]*len(fixed_vals))
			return dydt

		self.ode = scipy.integrate.ode(g).set_integrator(solver, **solver_args).set_f_params([])
		self.order = order

	''' Initial & Boundary Conditions ''' 

	def set_initial(self, t0: float=0., y0: Callable[[GeoObject], float]=lambda _: 0., **kwargs):

		def fill_arr(f: Callable[[GeoObject], float]) -> np.ndarray:
			return np.array([f(x) for x in self.domain.keys()])

		self.t0 = t0
		self.t = t0
		self.y0 = y0
		self.y = fill_arr(y0)
		self.init_kwargs = kwargs
		if self.ode is not None:
			n = len(self)
			y0 = np.concatenate((self.y, np.zeros((self.order-1)*n)))
			assert len(kwargs) == self.order - 1, f'{len(kwargs)+1} initial conditions provided but {self.order} needed'
			for i, y0_i in enumerate(kwargs.values()):
				y0[(i+1)*n:(i+2)*n] = fill_arr(y0_i)
			self.ode.set_initial_value(y0, t=self.t)

	def set_boundary(self, dirichlet_values: Dict[GeoObject, float]={}, neumann_values: Dict[GeoObject, float]={}):
		intersect = dirichlet_values.keys() & neumann_values.keys()
		assert len(intersect) == 0, f'Dirichlet and Neumann conditions overlap on {intersect}'
		self.dirichlet_values = dirichlet_values
		self.neumann_values = neumann_values
		fixed_idx = [self.domain[k] for k in dirichlet_values.keys()]
		fixed_vals = list(dirichlet_values.values())
		self.y = replace(self.y, fixed_idx, fixed_vals)
		if self.ode is not None:
			self.ode.set_initial_value(replace(self.ode.y, fixed_idx, fixed_vals), t=self.t)
			self.ode.set_f_params(fixed_idx)

	@property
	def boundary(self) -> List[GeoObject]:
		return list(self.dirichlet_values.keys()) + list(self.neumann_values.keys())

	''' Integration ''' 

	def step(self, dt: float):
		if self.ode is not None:
			self.ode.integrate(self.t + dt)

	def measure(self) -> np.ndarray:
		if self.ode is not None:
			self.t = self.ode.t
			self.y = self.ode.y[:len(self)]
		if self.plot is not None:
			self.render()
		return self.y

	def reset(self):
		self.set_initial(t0=self.t0, y0=self.y0, **self.init_kwargs)

	''' Builtins ''' 

	def __len__(self):
		return self.n

	def __call__(self, x: GeoObject):
		return self.y[self.domain[x]]

	''' Rendering ''' 

	def set_render_params(self, palette=cc.fire, lo=0., hi=1., layout_func=None, n_spring_iters=500):
		self.palette = palette
		self.lo = lo
		self.hi = hi
		if layout_func is None:
			self.layout_func = lambda G:  nx.spring_layout(G, scale=0.9, center=(0,0), iterations=n_spring_iters, seed=1)
		else:
			self.layout_func = layout_func

	def create_plot(self):
		''' Create plot for rendering with Bokeh ''' 
		if self.plot is None:
			G = nx.convert_node_labels_to_integers(self.G) # Bokeh cannot handle non-primitive node keys (eg. tuples)
			layout = self.layout_func(G)
			plot = figure(x_range=(-1.1,1.1), y_range=(-1.1,1.1), tooltips=[])
			plot.axis.visible = None
			plot.xgrid.grid_line_color = None
			plot.ygrid.grid_line_color = None
			renderer = from_networkx(G, layout)
			plot.renderers.append(renderer)
			self.plot = plot
		return self.plot

	@abstractmethod
	def render(self):
		''' Render current values to the plot ''' 
		pass

''' Observables on specific geometric objects ''' 

class VertexObservable(Observable):

	def init_domain(self):
		self.domain = dict(zip(self.G.nodes(), count()))

	def create_plot(self):
		super().create_plot()
		self.plot.renderers[0].node_renderer.data_source.data['node'] = list(self.G.nodes())
		self.plot.renderers[0].node_renderer.data_source.data['node_data'] = self.y 
		self.plot.renderers[0].node_renderer.glyph = Oval(height=0.08, width=0.08, fill_color=linear_cmap('node_data', self.palette, self.lo, self.hi))
		cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.palette, low=self.lo, high=self.hi), ticker=BasicTicker(), title=self.desc)
		self.plot.add_layout(cbar, 'right')
		self.plot.add_tools(HoverTool(tooltips=[(self.desc, '@node_data'), ('node', '@node')]))
		return self.plot

	def render(self):
		self.plot.renderers[0].node_renderer.data_source.data['node_data'] = self.y


class EdgeObservable(Observable):

	def init_domain(self):
		self.domain = dict(zip(self.G.edges(), count()))

	def create_plot(self):
		super().create_plot()
		self.plot.renderers[0].edge_renderer.data_source.data['edge_data'] = self.y
		self.plot.renderers[0].edge_renderer.glyph = MultiLine(line_color=linear_cmap('edge_data', self.palette, self.lo, self.hi), line_width=5)
		cbar = ColorBar(color_mapper=LinearColorMapper(palette=self.palette, low=self.lo, high=self.hi), ticker=BasicTicker(), title=self.desc)
		self.plot.add_layout(cbar, 'right')
		# self.plot.tooltips.append((self.desc, '@edge_data'))
		return self.plot

	def render(self):
		self.plot.renderers[0].edge_renderer.data_source.data['edge_data'] = self.y
		# TODO: render edge direction using: https://discourse.bokeh.org/t/hover-over-tooltips-on-network-edges/2439/7


''' Differential operators on observables ''' 

def partial(obs: VertexObservable, e: Edge) -> float: 
	return np.sqrt(obs.weight(e)) * obs(e[1]) - obs(e[0])

def grad(obs: VertexObservable) -> np.ndarray:
	return np.array([partial(obs, e) for e in obs.G.edges()])

def div_at(obs: EdgeObservable, x: Vertex) -> float:
	return sum([np.sqrt(obs.weight((x, n))) * (obs((n, x)) - obs((x, n))) for n in obs.G.neighbors(x)])

def div(self) -> np.ndarray:
	return np.array([div_at(obs, x) for x in obs.G.nodes()])

def laplacian_at(obs: VertexObservable, x: Vertex) -> float:
	''' Compute Laplacian that solves Neumann problem using phantom-node method '''
	ret = sum([obs.weight((x, n)) * (obs(n) - obs(x)) for n in obs.G.neighbors(x)])
	if x in obs.neumann_values:
		ret += np.sqrt(obs.default_weight) * obs.neumann_values[x] 
	return ret

def laplacian(obs: VertexObservable) -> np.ndarray:
	return np.array([laplacian_at(obs, x) for x in obs.G.nodes()])

def bilaplacian_at(obs: VertexObservable, x: Vertex) -> float:
	ret = sum([obs.weight((x, n)) * (laplacian_at(obs, n) - laplacian_at(obs, x)) for n in obs.G.neighbors(x)])
	if x in obs.neumann_values:
		ret += np.sqrt(obs.default_weight) * obs.neumann_values[x] 
	return ret

def bilaplacian(obs: VertexObservable) -> np.ndarray:
	return np.array([bilaplacian_at(obs, x) for x in obs.G.nodes()])

''' Multiple observables running concurrently on a graph ''' 

class System:
	def __init__(self, observables: List[Observable], desc=''):
		assert len(observables) > 0, 'Pass some observables'
		assert len(observables) == len(set([type(o) for o in observables])), 'Multiple observables on the same domain not currently supported'
		assert all([observables[0].G is o.G for o in observables]), 'All observables must be on the same graph instance'
		self.observables = observables
		self.plot = None
		self.desc = desc

	def create_plot(self):
		for i, p in enumerate(self.observables):
			if i == 0:
				self.plot = p.create_plot()
				self.plot.title.text = self.desc
			else:
				p.plot = self.plot
				p.create_plot()
		return self.plot

	def step(self, dt: float):
		for obs in self.observables:
			obs.step(dt)

	def measure(self) -> List[np.ndarray]:
		return [obs.measure() for obs in self.observables]

	def reset(self):
		for obs in self.observables:
			obs.reset()

	@property
	def t(self) -> float:
		return self.observables[0].t