''' Differential operators & forms on graphs '''

from typing import Callable
import numpy as np
import networkx as nx

def grad(G: nx.Graph, u: Callable, w_key: str='w') -> Callable: 
	''' Weighted gradient of a vertex function ''' 
	def v(e: tuple):
		if w_key in G[e[0]][e[1]]: w = G[e[0]][e[1]][w_key]
		else: w = 1.0
		return np.sqrt(w)*(u(e[1]) - u(e[0]))
	return v

def div(G: nx.Graph, v: Callable, w_key: str='w') -> Callable:
	''' Weighted divergence of an edge function ''' 
	def u(x1):
		ret = 0.
		for x2 in G.neighbors(x1):
			if w_key in G[x1][x2]: w = G[x1][x2][w_key]
			else: w = 1.0
			ret += np.sqrt(w)*(v(x2, x1) - v(x1, x2))
		return -ret 
	return u

