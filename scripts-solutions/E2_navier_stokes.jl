using Gridap, GridapGmsh
using DrWatson

msh_file = projectdir("meshes/perforated_plate_tiny.msh")
model = GmshDiscreteModel(msh_file)

writevtk(model,datadir("perforated_plate"))

D = 2
k = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},k)
V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["inlet","walls","cylinder"])
reffeₚ = ReferenceFE(lagrangian,Float64,k-1)
Q = TestFESpace(model,reffeₚ,conformity=:C0)

const Uₘ = 0.3
const H  = 0.41
u_in(x) = VectorValue( 4 * Uₘ * x[2] * (H-x[2]) / (H^2), 0.0 )
u_w(x)  = VectorValue(0.0,0.0)
u_c(x)  = VectorValue(0.0,0.0)

U = TrialFESpace(V,[u_in,u_w,u_c])
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

degree = k
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)

Γ_out = BoundaryTriangulation(model,tags="outlet")
n_Γout = get_normal_vector(Γ_out)
dΓ_out = Measure(Γ_out,degree)

const Re = 20
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ

c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v)
jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)

op = FEOperator(res,jac,X,Y)

using LineSearches: BackTracking
solver = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())

uh, ph = solve(solver,op)

out_file = datadir("ins")
writevtk(Ω,out_file,cellfields=["uh"=>uh,"ph"=>ph])
