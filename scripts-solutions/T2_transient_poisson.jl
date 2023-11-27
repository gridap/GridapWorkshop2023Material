using Gridap
using DrWatson

model = CartesianDiscreteModel((0,1,0,1),(20,20))
Ω  = Triangulation(model)
dΩ = Measure(Ω,2)

reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe,dirichlet_tags="boundary")

g(x,t::Real) = 0.0
g(t::Real) = x -> g(x,t)
U = TransientTrialFESpace(V,g)

κ(t) = 1.0 + 0.95*sin(2π*t)
f(t) = sin(π*t)
res(t,u,v) = ∫( ∂t(u)*v + κ(t)*(∇(u)⋅∇(v)) - f(t)*v )dΩ
jac(t,u,du,v) = ∫( κ(t)*(∇(du)⋅∇(v)) )dΩ
jac_t(t,u,duₜ,v) = ∫( duₜ*v )dΩ
op = TransientFEOperator(res,jac,jac_t,U,V)

op_AD = TransientFEOperator(res,U,V)

m(t,u,v) = ∫( u*v )dΩ
a(t,u,v) = ∫( κ(t)*(∇(u)⋅∇(v)) )dΩ
b(t,v) = ∫( f(t)*v )dΩ
op_Af = TransientAffineFEOperator(m,a,b,U,V)

m₀(u,v) = ∫( u*v )dΩ
a₀(u,v) = ∫( κ(0.0)*(∇(u)⋅∇(v)) )dΩ
op_CM = TransientConstantMatrixFEOperator(m₀,a₀,b,U,V)

b₀(v) = ∫( f(0.0)*v )dΩ
op_C = TransientConstantFEOperator(m₀,a₀,b₀,U,V)

linear_solver = LUSolver()

Δt = 0.05
θ = 0.5
ode_solver = ThetaMethod(linear_solver,Δt,θ)

u₀ = interpolate_everywhere(0.0,U(0.0))
t₀ = 0.0
T = 10.0
uₕₜ = solve(ode_solver,op,u₀,t₀,T)

dir = datadir("poisson_transient_solution")
!isdir(dir) && mkdir(dir)
createpvd(dir) do pvd
  for (uₕ,t) in uₕₜ
    file = dir*"/solution_$t"*".vtu"
    pvd[t] = createvtk(Ω,file,cellfields=["u"=>uₕ])
  end
end
