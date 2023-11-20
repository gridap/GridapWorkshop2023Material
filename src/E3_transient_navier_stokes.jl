
# Follows https://link.springer.com/chapter/10.1007/978-3-322-89849-4_39

using Gridap
using DrWatson

msh_file = projectdir("meshes/perforated_plate.json")
model = DiscreteModelFromFile(msh_file)

writevtk(model,datadir("perforated_plate"))

model = Gridap.Adaptivity.refine(model)
writevtk(model,datadir("perforated_plate_2"))

D = 2
order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["inlet","noSlip","hole"])

reffeₚ = ReferenceFE(lagrangian,Float64,order-1)
Q = TestFESpace(model,reffeₚ,conformity=:C0)

const Uₘ = 1.5
const H  = 0.41
u_in(x,t::Real) = VectorValue( 4 * Uₘ * x[2] * (H-x[2]) / (H^2), 0.0 )
u_0(x,t::Real)  = VectorValue(0,0)
u_in(t::Real)   = x -> u_in(x,t)
u_0(t::Real)    = x -> u_0(x,t)

U = TransientTrialFESpace(V,[u_in,u_0,u_0])
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V, Q])
X = TransientMultiFieldFESpace([U, P])

degree = order
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)

Γ_out = BoundaryTriangulation(model,tags="outlet")
n_Γout = get_normal_vector(Γ_out)
dΓ_out = Measure(Γ_out,degree)

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

hN(x) = VectorValue( 0.0, 0.0 )
l_out(v) = ∫( v⋅hN )dΓ_out

m(t,(ut,p),(v,q)) = ∫( ∂t(ut)⋅v )dΩ
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ

c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

res(t,(u,p),(v,q)) = m(t,(u,p),(v,q)) + a((u,p),(v,q)) + c(u,v) - l_out(v)
jac(t,(u,p),(du,dp),(v,q)) = m(t,(du,dp),(v,q)) + a((du,dp),(v,q)) + dc(u,du,v)
jac_t(t,(u,p),(dut,dpt),(v,q)) = m(t,(dut,dpt),(v,q))

op = TransientFEOperator(res,X,Y)

using LineSearches: BackTracking
nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())

Δt = 0.05
θ = 0.5
ode_solver = ThetaMethod(LUSolver(),Δt,θ)

u₀ = interpolate_everywhere([VectorValue(0.0,0.0),0.0],X(0.0))
t₀ = 0.0
T = 2.0
xₕₜ = solve(ode_solver,op,u₀,t₀,T)

dir = datadir("ins_stokes_transient")
!isdir(dir) && mkdir(dir)
createpvd(dir) do pvd
  for (xₕ,t) in xₕₜ
    uₕ,pₕ = xₕ
    file = dir*"/solution_$t"*".vtu"
    pvd[t] = createvtk(Ω,file,cellfields=["u"=>uₕ,"p"=>pₕ])
  end
end
