# In this tutorial, we will use the MPI-emulated environment provided by `DebugArray`
# to interactively have a look at some key aspects of how `GridapDistributed` works. 

# We will be using three packages:
#   - `Gridap`, which provides the local FE framework
#   - `PartitionedArrays`, which provides a generic library for MPI-distributed linear algebra
#   - `GridapDistributed`, which provides the distributed layer on top of `Gridap`
using Gridap
using GridapDistributed
using PartitionedArrays

# We start by creating our distributed rank indexes, which plays the role of the more 
# traditional communicator (such as MPI.COMM_WORLD). 
nprocs = (2,1)
ranks  = with_debug() do distribute
  distribute(LinearIndices((prod(nprocs),)))
end

# We can create a distributed Cartesian model by passing the newly-created ranks to the 
# serial constructor. Thanks to the magic of multiple-dispatch, this is pretty much one of 
# the few changes we will have to do to convert a serial Gridap code into a GridapDistributed code. 
domain = (0,1,0,1)
ncells = (4,2)
serial_model = CartesianDiscreteModel(domain,ncells)

model = CartesianDiscreteModel(ranks,nprocs,domain,ncells)

# The created `DistributedDiscreteModel` is just a wrapper around a distributed array of 
# serial models. One can get access to the local models with the method `local_views`, which 
# is defined for most distributed structures in `GridapDistributed`. 

local_models = local_views(model)
display(local_models)

# A key aspect in parallel programming is the concept of owned & ghost ids. If we compare 
# the number of cells of the distributed model to the number of cells of the local model portions, 
# we can observe that the local models overlap: 
global_ncells = num_cells(model)
local_ncells  = map(num_cells,local_views(model))

# This overlap is due to ghost cells. The Owned/Ghost layout of the current distributed model 
# can be seen in the next figure:
# <div>
# <img src="../figures/distributed/gids_cells.png" width="400"/>
# </div>

# This information is also stored on the distributed model, and can be accessed in the following way:
cell_gids = get_cell_gids(model)
local_cell_to_global_cell = map(local_to_global,partition(cell_gids))
local_cell_to_owner       = map(local_to_owner,partition(cell_gids))
owned_cell_to_local_cell  = map(own_to_local,partition(cell_gids))
ghost_cell_to_local_cell  = map(ghost_to_local,partition(cell_gids))

# For each processor, we have
#   - `local_to_global` - map from local cell ids to global cell ids
#   - `local_to_owner`  - map from local cell ids to the processor id of the owner 
#   - `own_to_local`    - list of owned local cells
#   - `ghost_to_local`  - list of ghost local cells
# More documentation on this can be found in PartitionedArrays.jl

# We can then continue by creating a `DistributedFESpace`, which (like `DistributedDiscreteModel`)
# is a structure that contains: 
#   - A distributed array of serial (overlapped) `FESpace`s
#   - The Owned/Ghost layout for the DoFs
# In this specific example here, and just for clarity, we do not impose any Dirichlet boundary conditions. Since the Poisson problem is defined up to a constant, and therefore is not uniquely defined without any dirichlet boundary conditions, the problem will not be well-posed. However, this is not a problem for the purpose of this tutorial.

feorder = 1
reffe = ReferenceFE(lagrangian,Float64,feorder)
V = FESpace(model,reffe)
U = TrialFESpace(V)

# The DoF layout can be seen in the following figure
# <div>
# <img src="../figures/distributed/gids_dofs.png" width="400"/>
# </div>
# and can be accessed as follows:

dof_gids = V.gids
local_dofs_to_global_dof = map(local_to_global,partition(dof_gids))
local_dofs_to_owner      = map(local_to_owner,partition(dof_gids))
owned_dofs_to_local_dof  = map(own_to_local,partition(dof_gids))
ghost_dofs_to_local_dof  = map(ghost_to_local,partition(dof_gids))

# In order to ensure unicity of the solution, we need to impose Dirichlet boundary conditions. Therefore we will overwrite the `FESpaces` as follows:

g(x) = 0.0
V = FESpace(model,reffe;dirichlet_tags=["boundary"])
U = TrialFESpace(V,g)

# We can now define the weak form and integrate as usual: 

degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

f(x)   = cos(x[1])
a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ
l(v)   = ∫( v*f )*dΩ

op = AffineFEOperator(a,l,V,U)
uh = solve(op)

# Let's take a moment to have a look at the distributed linear system: As usual, one can access the system matrix and rhs as

A  = get_matrix(op)
b  = get_vector(op)

# We observe $A$ and $b$ are now of type `PSparseMatrix` and `PVector`, respectively. These represent distributed equivalents of the serial `Matrix` and `Vector` types. Each object holds their local part of the array (both owned and ghost indices) and information on the Owned/Ghost layout for their rows (and columns). 
#
# The local sub-matrices and sub-vectors can be accessed as follows:
#

local_mats = partition(A)
owned_mats = own_values(A)

local_vectors = partition(b)
owned_vectors = own_values(b)

# Here, `local_mats` and `local_vectors` contain the sub-matrices and sub-vectors for all the local DoFs (i.e with overlapping), while `owned_mats` and `owned_vectors` contain the sub-matrices and sub-vectors for the owned DoFs only (no overlapping).

