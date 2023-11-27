using Gridap
using DrWatson
model = DiscreteModelFromFile("meshes/poisson.json")

writevtk(model,"model")

labels = get_face_labeling(model)

add_tag_from_tags!(labels,"diri0",["sides", "sides_c"])

add_tag_from_tags!(labels,"dirig",
  ["circle","circle_c", "triangle", "triangle_c", "square", "square_c"])

reffe = ReferenceFE(lagrangian,Float64,1)
V0 = TestFESpace(model,reffe,conformity=:H1,labels=labels,dirichlet_tags=["diri0", "dirig"])

g = 1
Ug = TrialFESpace(V0,[0,g])

degree=2
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

const p = 3
flux(∇u) = norm(∇u)^(p-2) * ∇u
f(x) = 1
res(u,v) = ∫( ∇(v)⊙(flux∘∇(u)) - v*f )*dΩ

dflux(∇du,∇u) = (p-2)*norm(∇u)^(p-4)*(∇u⊙∇du)*∇u+norm(∇u)^(p-2)*∇du
jac(u,du,v) = ∫( ∇(v)⊙(dflux∘(∇(du),∇(u))) )*dΩ

op = FEOperator(res,jac,Ug,V0)

op_AD = FEOperator(res,Ug,V0)

using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

import Random
Random.seed!(1234)
x = rand(Float64,num_free_dofs(Ug))
uh0 = FEFunction(Ug,x)
uh, = solve!(uh0,solver,op)

writevtk(Ω,datadir("p_laplacian"),cellfields=["uh"=>uh])
