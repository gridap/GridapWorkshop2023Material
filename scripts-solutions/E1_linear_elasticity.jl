using Gridap, GridapGmsh
using DrWatson

msh_file_gmsh = projectdir("meshes/elasticity.msh")
model = GmshDiscreteModel(msh_file_gmsh)

msh_file_json = projectdir("meshes/elasticity.json")
to_json_file(model,msh_file_json)

model = DiscreteModelFromFile(msh_file_json)

writevtk(model,datadir("elasticity_model"))

dirichlet_tags = ["surface_1","surface_2"]

dirichlet_masks=[(true,false,false), (true,true,true)]

order = 1
reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
V0    = TestFESpace(model,reffe;
                    conformity=:H1,
                    dirichlet_tags=dirichlet_tags,
                    dirichlet_masks=dirichlet_masks)

g1(x) = VectorValue(0.005,0.0,0.0)
g2(x) = VectorValue(0.0,0.0,0.0)

U = TrialFESpace(V0,[g1,g2])

Γ = BoundaryTriangulation(model,tags=dirichlet_tags)

vh = zero(U)

writevtk(Γ,"check-trial-fe-space",cellfields=["vh"=>vh])

const E = 70.0e9
const ν = 0.33
const λ = (E*ν)/((1+ν)*(1-2*ν))
const μ = E/(2*(1+ν))
σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε

degree = 2*order
Ω      = Triangulation(model)
dΩ     = Measure(Ω,degree)

a(u,v) = ∫( (σ∘ε(u)) ⊙ ε(v) )dΩ
l(v)   = 0

op = AffineFEOperator(a,l,U,V0)
uh = solve(op)

out_file = datadir("elasticity_sol")
writevtk(Ω,out_file,cellfields=["uh"=>uh,"epsi"=>ε(uh),"sigma"=>σ∘ε(uh)])
