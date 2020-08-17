import numpy as np
import random
import torch
from collections.abc import Iterable
from itertools import repeat
from typing import Callable

def set_seed(seed=None):
	random.seed(seed)
	np.random.seed(seed)
	if seed is None: 
		torch.manual_seed(random.randint(1,1e6))
	else:
		torch.manual_seed(seed)

def map_nd_to_1d(mx: tuple, coord: tuple):
	''' Embed n-d coordinates in 1d using dictionary ordering. All args must be nonnegative
	Args:
		mx: tuple containing elements per dimension
		coord: nd coordinate
	''' 
	n = 0
	d = len(mx)
	if d == 0: return 0
	for c_i in range(d):
		if c_i == d-1:
			n += coord[c_i]
		else:
			n += coord[c_i] * np.prod(mx[c_i+1:])
	return n

def map_1d_to_nd(mx: tuple, i: int):
	''' Inverse of `map_nd_to_1d`. All args must be nonnegative
	Args:
		mx: tuple containing elements per dimension
		i: 1d coordinate
	''' 
	d = len(mx)
	c = [0 for _ in range(d)]
	rem = i
	if d == 0: return ()
	for c_i in range(d):
		if c_i == d-1:
			c[c_i] = rem
		else:
			(q, rem) = divmod(rem, np.prod(mx[c_i+1:]))
			c[c_i] = q
	return tuple(c)

def is_sorted(l: list):
	return all(l[i] <= l[i+1] for i in range(len(l)-1))

def rms(arr: np.ndarray):
	return np.sqrt((arr ** 2).mean())

def replace(arr: np.ndarray, replace_at: list, replace_with: np.ndarray):
	arr[replace_at] = replace_with
	return arr
