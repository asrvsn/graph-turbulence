from utils import *
import networkx as nx

mx = (4,7)
coord = (0,2)
assert map_nd_to_1d(mx, coord) == 2
assert map_1d_to_nd(mx, map_nd_to_1d(mx, coord)) == coord

mx = (4,7,2)
coord = (3,5,1)
assert map_nd_to_1d(mx, coord) == 53
assert map_1d_to_nd(mx, map_nd_to_1d(mx, coord)) == coord

mx = (10,10)
G = nx.grid_2d_graph(10,10)
vs = [map_nd_to_1d(mx, v) for v in G.nodes]
assert is_sorted(vs)