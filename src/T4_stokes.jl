# ## Problem statement 
#
# The goal is to solve a linear multi-field PDE with saddle point structure. As a model problem, we consider a well known benchmark in computational fluid dynamics, the lid-driven cavity benchmark for the incompressible Stokes equations at low Reynolds numbers.
# We will be using a mixed finite-element with a well know inf-sup stable element pair $Q_{k}/P_{k-1}^-$ for velocity/pressure.
#
# Formally, the PDE we want to solve is: find the velocity vector $u$ and the pressure $p$ such that
#
# $$
# \left\lbrace
# \begin{aligned}
# - \mathit{Re}\ \Delta u + \nabla p = 0 &\text{ in }\Omega,\\
# \nabla\cdot u = 0 &\text{ in } \Omega,\\
# u = g &\text{ on } \partial\Omega,
# \end{aligned}
# \right.
# $$
#
# where the computational domain is the unit square $\Omega \doteq (0,1)^d$, $d=2$, and $\mathit{Re}$ is the Reynolds number. In this example, the driving force is the Dirichlet boundary velocity $g$, which is a non-zero horizontal velocity with a value of $g = (1,0)^t$ on the top side of the cavity, namely the boundary $(0,1)\times\{1\}$, and $g=0$ elsewhere on $\partial\Omega$. Since we impose Dirichlet boundary conditions on the entire boundary $\partial\Omega$, the mean value of the pressure is constrained to zero in order have a well posed problem,

using Gridap
using DrWatson

# ## Geometry
#
# Discrete model
n = 100
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)

# Define Dirichlet boundaries
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri1",[6,])
add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

# ## FE spaces 
#
# We define reference finite-element pair $Q_{k}/P_{k-1}^-$

order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)

# We then define the test spaces for velocity and pressure: 
# Note that the pressure space is defined as a L2-conforming space with zero mean constraint.

V = TestFESpace(model,reffeᵤ,labels=labels,dirichlet_tags=["diri0","diri1"],conformity=:H1)
Q = TestFESpace(model,reffeₚ,conformity=:L2,constraint=:zeromean)

# The trial spaces are then defined in the usual way: 

u0 = VectorValue(0,0)
u1 = VectorValue(1,0)
U = TrialFESpace(V,[u0,u1])
P = TrialFESpace(Q)

# With all these ingredients we create the FE spaces representing the Cartesian product of the velocity and pressure FE spaces, $(X,Y) = (U \times P ,V \times Q)$, which is where we are seeking the solution the problem. 

Y = MultiFieldFESpace([V,Q])
X = MultiFieldFESpace([U,P])

# ## Integration
#
# From the discrete model we can define the triangulation and integration measure

degree = order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)

# As usual, we define bilinear and linear forms for our problem. 
# Note that, since we are using a Cartesian product FE space, it's elements are tuples. Here we use `(u,p)` and `(v,q)` to denote the trial and test functions.

f = VectorValue(0.0,0.0)
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
l((v,q)) = ∫( v⋅f )dΩ

# We can then build the AffineFEOperator as usual and solve using our solver of choice. 
# Note that the solver now retuns a solution, `xh`, which behaves like a tuple of solutions, one for each field in the space. We can then easily unpack the solution into the velocity and pressure components.

op = AffineFEOperator(a,l,X,Y)
xh = solve(op)
uh, ph = xh

# Finally, we export the results to vtk
writevtk(Ωₕ,datadir("stokes"),order=2,cellfields=["uh"=>uh,"ph"=>ph])
