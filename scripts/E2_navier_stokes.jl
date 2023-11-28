using Gridap, GridapGmsh
using DrWatson

msh_file = projectdir("meshes/perforated_plate_tiny.msh")
model = GmshDiscreteModel(msh_file)

writevtk(model,datadir("perforated_plate"))

D = 2
k = 2

#U =
#P =

#Y =
#X =

degree = k
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)

Γ_out = BoundaryTriangulation(model,tags="outlet")
n_Γout = get_normal_vector(Γ_out)
dΓ_out = Measure(Γ_out,degree)

const Re = 20
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

#a((u,p),(v,q)) =

#c(u,v) =
#dc(u,du,v) =

#res((u,p),(v,q)) =
#jac((u,p),(du,dp),(v,q)) =

op = FEOperator(res,jac,X,Y)

using LineSearches: BackTracking
