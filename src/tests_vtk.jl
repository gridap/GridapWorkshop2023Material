
using Gridap, GridapDistributed, GridapGmsh, PartitionedArrays
using DrWatson

np = 2
ranks = with_debug() do distribute
  distribute(LinearIndices((np,)))
end

msh_file = projectdir("meshes/perforated_plate.msh")
model = GmshDiscreteModel(ranks,msh_file)

reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},2)
V = FESpace(model,reffe)

Ω = Triangulation(model)
uh = zero(V)

out_dir = datadir("test")
!isdir(out_dir) && mkdir(out_dir)
for i in 1:3
  out_file = joinpath(out_dir,"test_$(i).vtu")
  createvtk(Ω,out_file,cellfields=["u"=>uh])
end
