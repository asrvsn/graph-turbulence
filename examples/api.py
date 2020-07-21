pressure = VertexObservable(G, desc='Pressure')
velocity = EdgeObservable(G, desc='Velocity')

velocity.set_ode(lambda t: -velocity.y@grad(velocity) - grad(pressure))

pressure.set_initial(y0=lambda _: 0)
pressure.set_boundary(dirichlet_values={
	(3,3): 1.0, 
	(7,7): -1.0
})

velocity.set_initial(y0=lambda _: 0)
velocity.set_boundary(dirichlet_values={
	((3,3), (3,4)): 1.0
})

sys = System([pressure, velocity], desc=f'A test fluid flow with pressure inlet and outlet')
render_live(sys)