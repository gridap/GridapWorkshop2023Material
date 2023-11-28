using Gridap
using GridapDistributed
using PartitionedArrays

nprocs = (2,1)
ranks  = with_debug() do distribute
  distribute(LinearIndices((prod(nprocs),)))
end

domain = (0,1,0,1)
ncells = (4,2)
serial_model = CartesianDiscreteModel(domain,ncells)

model = CartesianDiscreteModel(ranks,nprocs,domain,ncells)

local_models = local_views(model)
display(local_models)

global_ncells = num_cells(model)
local_ncells  = map(num_cells,local_views(model))

cell_gids = get_cell_gids(model)
local_cell_to_global_cell = map(local_to_global,partition(cell_gids))
local_cell_to_owner       = map(local_to_owner,partition(cell_gids))
owned_cell_to_local_cell  = map(own_to_local,partition(cell_gids))
ghost_cell_to_local_cell  = map(ghost_to_local,partition(cell_gids))

feorder = 1
reffe = ReferenceFE(lagrangian,Float64,feorder)
V = FESpace(model,reffe)
U = TrialFESpace(V)

dof_gids = V.gids
local_dofs_to_global_dof = map(local_to_global,partition(dof_gids))
local_dofs_to_owner      = map(local_to_owner,partition(dof_gids))
owned_dofs_to_local_dof  = map(own_to_local,partition(dof_gids))
ghost_dofs_to_local_dof  = map(ghost_to_local,partition(dof_gids))

degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

f(x)   = cos(x[1])
a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ
l(v)   = ∫( v*f )*dΩ

op = AffineFEOperator(a,l,V,U)
uh = solve(op)

A  = get_matrix(op)
b  = get_vector(op)

local_mats = partition(A)
owned_mats = own_values(A)

local_vectors = partition(b)
owned_vectors = own_values(b)

rows = axes(A,1)
cols = axes(A,2)

owned_dofs_to_local_dof  = map(own_to_local,partition(dof_gids))
ghost_dofs_to_local_dof  = map(ghost_to_local,partition(dof_gids))

owned_rows_to_local_row  = map(own_to_local,partition(rows))
ghost_rows_to_local_row  = map(ghost_to_local,partition(rows))

x = get_free_dof_values(uh) # DoF layout
A * x # Error!
A * b # Error!

x_r = pfill(0.0,partition(axes(A,1))) # Row layout
x_c = pfill(0.0,partition(axes(A,2))) # Col layout
x_r .= A * x_c # OK

vh = FEFunction(V,x_c)
