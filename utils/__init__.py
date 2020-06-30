import numpy as np
import random
import torch

def set_seed(seed=None):
	random.seed(seed)
	np.random.seed(seed)
	if seed is None: 
		torch.manual_seed(random.randint(1,1e6))
	else:
		torch.manual_seed(seed)

def map_nd_to_1d(coord: tuple):
	pass

def map_1d_to_nd(i: int):
	pass