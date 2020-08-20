import numpy as np

from .observable import *

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
	# return np.array([laplacian_at(obs, x) for x in obs.G.nodes()])
	return obs.laplacian@obs.y + np.sqrt(obs.default_weight)*obs.neumann_vec

def bilaplacian_at(obs: VertexObservable, x: Vertex) -> float:
	ret = sum([obs.weight((x, n)) * (laplacian_at(obs, n) - laplacian_at(obs, x)) for n in obs.G.neighbors(x)])
	if x in obs.neumann_values:
		ret += np.sqrt(obs.default_weight) * obs.neumann_values[x] 
	return ret

def bilaplacian(obs: VertexObservable) -> np.ndarray:
	# TODO correct way to handle Neumann in this case? (Gradient constraint only specifies one neighbor beyond)
	return obs.laplacian@(obs.laplacian@obs.y + np.sqrt(obs.default_weight)*obs.neumann_vec) 

def advect_vertex(x: Vertex, obs: VertexObservable, v_field: EdgeObservable) -> float:
	''' Advection of a scalar field ''' 
	return sum([v_field((x, y)) * partial(obs, (x, y)) for y in obs.G.neighbors(x)])

def advect_edge(e: Edge, obs: EdgeObservable, v_field: EdgeObservable) -> float:
	''' Advection of a vector field ''' 
	return sum([v_field(e) * obs(e_n) / obs.weight(e, e_n) for e_n in obs.vertex_dual.neighbors(e)])	

def advect(obs: Observable, v_field: EdgeObservable) -> np.ndarray:
	if type(obs) == VertexObservable:
		return np.array([advect_vertex(x, obs, v_field) for x in obs.domain])
	elif type(obs) == EdgeObservable:
		return np.array([advect_edge(e, obs, v_field) for v in obs.domain])
	else:
		raise NotImplementedError

''' Views & transformations of observables ''' 

def project_cycle_basis(obs: VertexObservable) -> List[VertexObservable]: 
	''' Cycle-basis decomposition '''
	ret = []
	basis = nx.cycle_basis(obs.G)
	for i, cycle in enumerate(basis):
		G_cyc = nx.Graph()
		nx.add_cycle(G_cyc, cycle)
		obs_cyc = VertexObservable(G_cyc, desc=f'Cycle {i}')
		obs_cyc.track(obs) # Project values from larger system
		obs_cyc.set_render_params(palette=obs.palette, lo=obs.lo, hi=obs.hi, show_bar=False)
		ret.append(obs_cyc)
	return ret

def couple(*obs: Tuple[Observable]):
	''' Couple multiple PDEs on different domains into a single system. No-op on any observables which do not have an instantiated PDE. ''' 
	pass