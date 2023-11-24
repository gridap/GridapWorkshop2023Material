#
# ## Transient problem: p-Laplacian heat equation
#

# $$
# \left\lbrace
# \begin{aligned}
# \frac{\partial u(t)}{\partial t} -\kappa(t) \nabla \cdot \left( |\nabla u|^{p-2} \ \nabla u \right) = f(t)  \ &\text{ in } \ \Omega,\\
# u(t) = 0 \ &\text{ on }\ \Gamma_{\rm D},\\
# u(0) = 0 \ &\text{ in }\ \Omega, \\
# n \cdot \left( |\nabla u|^{p-2}\ \nabla u \right) = 0 \ &\text{ on } \ \Gamma_{\rm N}
# \end{aligned}
# \right.
# $$

using Gridap
using LineSearches: BackTracking
using DrWatson

model = CartesianDiscreteModel((0,1,0,1),(20,20))

p = 3
order = 3
reffe = ReferenceFE(lagrangian,Float64,order)
V0 = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")

degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

gt(x,t::Real) = 0.0
gt(t::Real) = x -> gt(x,t)
Ut = TransientTrialFESpace(V0,gt)

κ(t) = 1.0 + 0.95*sin(2π*t)
f(t) = 0.1 + sin(π*t)

flux(∇u) = norm(∇u)^(p-2) * ∇u
dflux(∇du,∇u) = (p-2)*norm(∇u)^(p-4)*(∇u⊙∇du)*∇u+norm(∇u)^(p-2)*∇du

res(t,u,v) = ∫( ∂t(u)*v + κ(t)⋅(∇(v)⊙(flux∘∇(u))) - f(t)*v )dΩ
jac(t,u,du,v) = ∫( κ(t)*(∇(v)⊙(dflux∘(∇(du),∇(u)))) )dΩ
jac_t(t,u,duₜ,v) = ∫( duₜ*v )dΩ
op = TransientFEOperator(res,Ut,V0)

Δt = 0.05
θ  = 0.5
nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
ode_solver = ThetaMethod(nls,Δt,θ)

u₀ = interpolate_everywhere(0.1,Ut(0.0))
t₀ = 0.0
T  = 1.0
uₕₜ = solve(ode_solver,op,u₀,t₀,T)

dir = datadir("p_laplacian_transient")
!isdir(dir) && mkdir(dir)
createpvd(dir) do pvd
  for (uₕ,t) in uₕₜ
    file = dir*"/solution_$t"*".vtu"
    pvd[t] = createvtk(Ω,file,cellfields=["u"=>uₕ])
  end
end
