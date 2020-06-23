using DifferentialEquations
α=1
β=1
u₀=1/2
f(u,p,t) = α*u
g(u,p,t) = β*u
dt = 1//2^(4)
tspan = (0.0,1.0)
prob = SDEProblem(f,g,u₀,(0.0,1.0))

sol = solve(prob,EM(),dt=dt)

using Plots; plotly() # Using the Plotly backend
plot(sol)