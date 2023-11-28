using Gridap, GridapGmsh
using DrWatson

# model =

D = 2
k = 2

# reffeᵤ =
# reffeₚ =
# V =
# Q =

const Tth = 2
const Uₘ = 1.5
const H  = 0.41
ξ(t) = (t <= Tth) ? sin(π*t/(2*Tth)) : 1.0
# u_in(x,t::Real) =
# u_w(x,t::Real)  =
# u_c(x,t::Real)  =
u_in(t::Real)   = x -> u_in(x,t)
u_w(t::Real)    = x -> u_w(x,t)
u_c(t::Real)    = x -> u_c(x,t)

# U =
# P =
# Y =
# X =

degree = k
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)

Γ_out = BoundaryTriangulation(model,tags="outlet")
n_Γout = get_normal_vector(Γ_out)
dΓ_out = Measure(Γ_out,degree)

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u

# m(t,(u,p),(v,q)) =
# a(t,(u,p),(v,q)) =
# c(u,v) =
# res(t,(u,p),(v,q)) =

op = TransientFEOperator(res,X,Y)

using LineSearches: BackTracking
nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())

# Δt =
# θ  =
# ode_solver =

u₀ = interpolate_everywhere([VectorValue(0.0,0.0),0.0],X(0.0))
t₀ = 0.0
T  = Tth
xₕₜ = solve(ode_solver,op,u₀,t₀,T)

dir = datadir("ins_transient")
!isdir(dir) && mkdir(dir)
createpvd(dir) do pvd
  for (xₕ,t) in xₕₜ
    println(" > Computing solution at time $t")
    uₕ,pₕ = xₕ
    file = dir*"/solution_$t"*".vtu"
    pvd[t] = createvtk(Ω,file,cellfields=["u"=>uₕ,"p"=>pₕ])
  end
end
