using Gridap, GridapDistributed, GridapGmsh, PartitionedArrays
using DrWatson

np = 2
ranks = with_debug() do distribute
  distribute(LinearIndices((np,)))
end

# model =

# reffeᵤ =
# reffeₚ =

# V =
# Q =
const Tth = 1
const Uₘ = 1.5
const H  = 0.41
ξ(t) = (t <= Tth) ? sin(π*t/(2*Tth)) : 1.0
u_in(x,t::Real) = VectorValue( 4 * Uₘ * x[2] * (H-x[2]) / (H^2) * ξ(t), 0.0 )
u_w(x,t::Real)  = VectorValue(0.0,0.0)
u_c(x,t::Real)  = VectorValue(0.0,0.0)
u_in(t::Real)   = x -> u_in(x,t)
u_w(t::Real)    = x -> u_w(x,t)
u_c(t::Real)    = x -> u_c(x,t)

# U =
# P =
# Y =
# X =

degree = 2*k
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)

Γ_out = BoundaryTriangulation(model,tags="outlet")
n_Γout = get_normal_vector(Γ_out)
dΓ_out = Measure(Γ_out,degree)

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u

m((u,p),(v,q)) = ∫( u⋅v )dΩ
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(conv∘(u,∇(du)) + conv∘(du,∇(u))) )dΩ

res(t,(u,p),(v,q)) = m((∂t(u),p),(v,q)) + a((u,p),(v,q)) + c(u,v)
jac(t,(u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)
jac_t(t,(u,p),(dut,dpt),(v,q)) = m((dut,dpt),(v,q))
op = TransientFEOperator(res,jac,jac_t,X,Y)

using Gridap.Algebra
nls = NewtonRaphsonSolver(LUSolver(),1.e-6,10)

# Δt =
# θ  =
# ode_solver =

x₀ = interpolate_everywhere([VectorValue(0.0,0.0),0.0],X(0.0))
t₀ = 0.0
T  = Tth
xₕₜ = solve(ode_solver,op,x₀,t₀,T)

dir = datadir("ins_distributed")
i_am_main(ranks) && !isdir(dir) && mkdir(dir)
for (xₕ,t) in xₕₜ
  println(" > Computing solution at time $t")
  uₕ,pₕ = xₕ
  file = dir*"/solution_$t"*".vtu"
  writevtk(Ω,file,cellfields=["u"=>uₕ,"p"=>pₕ])
end
