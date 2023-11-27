using Gridap, GridapGmsh
using DrWatson

msh_file = projectdir("meshes/perforated_plate_tiny.msh")
model = GmshDiscreteModel(msh_file)

D = 2
k = 2

reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},k)
reffeₚ = ReferenceFE(lagrangian,Float64,k-1)

V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["inlet","walls","cylinder"])
Q = TestFESpace(model,reffeₚ,conformity=:C0)

const Tth = 2
const Uₘ = 1.5
const H  = 0.41
ξ(t) = (t <= Tth) ? sin(π*t/(2*Tth)) : 1.0
u_in(x,t::Real) = VectorValue( 4 * Uₘ * x[2] * (H-x[2]) / (H^2) * ξ(t), 0.0 )
u_w(x,t::Real)  = VectorValue(0.0,0.0)
u_c(x,t::Real)  = VectorValue(0.0,0.0)
u_in(t::Real)   = x -> u_in(x,t)
u_w(t::Real)    = x -> u_w(x,t)
u_c(t::Real)    = x -> u_c(x,t)

U = TransientTrialFESpace(V,[u_in,u_w,u_c])
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V, Q])
X = TransientMultiFieldFESpace([U, P])

degree = k
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)

Γ_out = BoundaryTriangulation(model,tags="outlet")
n_Γout = get_normal_vector(Γ_out)
dΓ_out = Measure(Γ_out,degree)

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u

m(t,(u,p),(v,q)) = ∫( ∂t(u)⋅v )dΩ
a(t,(u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ

res(t,(u,p),(v,q)) = m(t,(u,p),(v,q)) + a(t,(u,p),(v,q)) + c(u,v)
op = TransientFEOperator(res,X,Y)

using LineSearches: BackTracking
nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())

Δt = 0.01
θ  = 0.5
ode_solver = ThetaMethod(nls,Δt,θ)

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
