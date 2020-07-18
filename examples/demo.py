''' Demos of PDEs on graphs ''' 

from examples import heat, wave, pattern
from rendering import *

if __name__ == '__main__':
	render_live([heat.sys1(), wave.sys1(), pattern.spots()])

