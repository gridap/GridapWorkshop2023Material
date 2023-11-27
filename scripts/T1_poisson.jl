using Gridap
using DrWatson

u₀(x)  = cos(x[1])*sin(x[2]+π)
∇u₀(x) = VectorValue(-sin(x[1])*sin(x[2]+π),cos(x[1])*cos(x[2]+π),0.0)
Δu₀(x) = -2.0*cos(x[1])*sin(x[2]+π)

f(x) = -Δu₀(x)
g(x) = u₀(x)
h(x) = 0.0      # ∇u₀ ⋅ n_Γ = ∇u₀ ⋅ ± e₃ = 0

domain = (-π,π,-π/2,π/2,0,1)
nC     = (100,40,5)
model  = CartesianDiscreteModel(domain,nC)

labels = get_face_labeling(model)

add_tag_from_tags!(labels,"neumann",["tag_21","tag_22"])
add_tag_from_tags!(labels,"dirichlet",["tag_23","tag_24","tag_25","tag_26"])

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="dirichlet")

U = TrialFESpace(V,g)

degree = order*2
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)

Γ   = BoundaryTriangulation(model,tags="neumann")
dΓ  = Measure(Γ,degree)

a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ
l(v)   = ∫( v*f )*dΩ + ∫( v*h )*dΓ

op = AffineFEOperator(a,l,U,V)

A = get_matrix(op)
b = get_vector(op)

ls = LUSolver()
solver = LinearFESolver(ls)

uh = solve(solver,op)

writevtk(Ω,datadir("poisson"),cellfields=["uh"=>uh])

dΩe  = Measure(Ω,degree*2)
e = uh - u₀
l2_error = sqrt(sum(∫(e⋅e)*dΩe))

function driver(n,order)
  domain = (-π,π,-π/2,π/2,0,1)
  nC     = (n,n,1)
  model  = CartesianDiscreteModel(domain,nC)
  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet",["tag_21","tag_22"])
  add_tag_from_tags!(labels,"neumann",["tag_23","tag_24","tag_25","tag_26"])

  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="dirichlet")

  U = TrialFESpace(V,g)
  degree = order*2+1
  Ω   = Triangulation(model)
  dΩ  = Measure(Ω,degree)
  Γ   = BoundaryTriangulation(model,tags="neumann")
  dΓ  = Measure(Γ,degree)

  a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ
  l(v)   = ∫( v*f )*dΩ + ∫( v*h )*dΓ
  op     = AffineFEOperator(a,l,U,V)
  ls     = LUSolver()
  solver = LinearFESolver(ls)
  uh = solve(solver,op)

  dΩe  = Measure(Ω,degree*2)
  e = uh - u₀
  return sqrt(sum(∫(e⋅e)*dΩe))
end

order_vec = [1,2]
n_vec = [10,20,40,80]
h_vec = map(n -> 1/n, n_vec)

error = zeros((length(order_vec),length(n_vec)))
for (i,order) in enumerate(order_vec)
  for (j,n) in enumerate(n_vec)
    error[i,j] = driver(n,order)
  end
end

using Plots
plt = plot(xlabel="log10(h)",ylabel="log10(L2 error)",grid=true)
for (i,e) in enumerate(eachrow(error))
  order = order_vec[i]
  dx = log10(h_vec[1]) - log10(h_vec[end])
  dy = log10(e[1]) - log10(e[end])
  slope = string(dy/dx)[1:4]
  plot!(plt,log10.(h_vec),log10.(e),label="p = $(order), slope = $(slope)")
end
@show plt
