# ## Problem statement
#
# The goal is to solve a nonlinear multi-field PDE. As a model problem, we consider a well known benchmark in computational fluid dynamics, the laminar flow around a cyclinder for the incompressible Navier-Stokes equations. We will solve this problem by building on the concepts seen in the previous tutorials.
#
# The computational domain $\Omega$ is a 2-dimensional channel. The fluid enters the channel from the left boundary (inlet) and exits through the right boundary (outlet). The channel has a cylindrical obstacle near the inlet. The domain can be seen in the following figure:
# <div>
# <img src="../figures/ins/perforated_plate.png" width="400"/>
# </div>
#
# We define $\partial \Omega = \Gamma_{w} \cup \Gamma_{c} \cup \Gamma_{in} \cup \Gamma_{out}$ with $\Gamma_{w}$ the top and bottom channel walls, $\Gamma_{c}$ the cylinder walls, $\Gamma_{in}$ the inlet and $\Gamma_{out}$ the outlet.
#
# Formally, the PDE we want to solve is: find the velocity vector $u$ and the pressure $p$ such that
#
# $$
# \left\lbrace
# \begin{aligned}
# -\Delta u + \mathit{Re}\ (u\cdot \nabla)\ u + \nabla p = 0 &\text{ in }\Omega,\\
# \nabla\cdot u = 0 &\text{ in } \Omega,\\
# u = u_{in} &\text{ on } \Gamma_{in},\\
# u = 0 &\text{ on } \Gamma_{w} \cup \Gamma_{c},\\
# n_\Gamma \cdot \sigma = 0 &\text{ on } \Gamma_{out},\\
# \end{aligned}
# \right.
# $$
#
# where $d=2$ , and $\mathit{Re}$ is the Reynolds number.
#
# The inflow condition is given by 
#
# $$
# u_{in}(0,y) = \left( 4 U_{m} \frac{y(H-y)}{H^2}, 0 \right),
# $$
#
# with $U_{m}=0.3 \ m/s$ the maximum velocity and $H = 0.41 \ m$ the height of the channel.
#
# ## Numerical Scheme
#
# In order to approximate this problem we chose a formulation based on inf-sub stable $P_{k}/P_{k-1}$ triangular elements with continuous velocities and pressures. The interpolation spaces are defined as follows. The velocity interpolation space is
#
# $$
# V \doteq \{ v \in [H^1(\Omega)]^d:\ v|_T\in [P_k(T)]^d \text{ for all } T\in\mathcal{T} \},
# $$
#
# where $T$ denotes an arbitrary cell of the FE mesh $\mathcal{T}$, and $P_k(T)$ is the usual Lagrangian FE space of order $k$ defined on a mesh of triangles or tetrahedra.
# On the other hand, the space for the pressure is given by
#
# $$
# Q \doteq \{ q \in C^0(\Omega):\ q|_T\in P_{k-1}(T) \text{ for all } T\in\mathcal{T}\}.
# $$
#
# The weak form associated to these interpolation spaces reads: find $(u,p)\in U_g \times Q$ such that $[r(u,p)](v,q)=0$ for all $(v,q)\in V_0 \times Q$
# where $U_g$ and $V_0$ are the set of functions in $V$ fulfilling the Dirichlet boundary conditions and the homogeneous Dirichlet boundary conditions respetively. The weak residual $r$ evaluated at a given pair $(u,p)$ is the linear form defined as
#
# $$
# [r(u,p)](v,q) \doteq a((u,p),(v,q))+ [c(u)](v),
# $$
# with
# $$
# \begin{aligned}
# a((u,p),(v,q)) &\doteq \int_{\Omega} \nabla v \cdot \nabla u \ {\rm d}\Omega - \int_{\Omega} (\nabla\cdot v) \ p \ {\rm d}\Omega + \int_{\Omega} q \ (\nabla \cdot u) \ {\rm d}\Omega,\\
# [c(u)](v) &\doteq \int_{\Omega} v 	\cdot \left( (u\cdot\nabla)\ u \right)\ {\rm d}\Omega.\\
# \end{aligned}
# $$
# Note that the bilinear form $a$ is associated with the linear part of the PDE, whereas $c$ is the contribution to the residual resulting from the convective term.
#
# In order to solve this nonlinear weak equation with a Newton-Raphson method, one needs to compute the Jacobian associated with the residual $r$. In this case, the Jacobian $j$ evaluated at a pair $(u,p)$ is the bilinear form defined as
#
# $$
# [j(u,p)]((\delta u, \delta p),(v,q)) \doteq a((\delta u,\delta p),(v,q))  + [{\rm d}c(u)](\delta u,v),
# $$
# where ${\rm d}c$ results from the linearization of the convective term, namely
# $$
# [{\rm d}c(u)](\delta u,v) \doteq \int_{\Omega} v \cdot \left( (u\cdot\nabla)\ \delta u \right) \ {\rm d}\Omega + \int_{\Omega} v \cdot \left( (\delta u\cdot\nabla)\ u \right)  \ {\rm d}\Omega.
# $$
#
# ## Geometry
# 
# We start by importing the packages and loading the provided mesh: 
#

using Gridap
using DrWatson

msh_file = projectdir("meshes/perforated_plate.json")
model = DiscreteModelFromFile(msh_file)

# ### Exercise 1
#
# _Open the resulting files with Paraview. Visualize the faces of the model and color them by each of the available fields. Identify the tag names representing the boundaries $\Gamma_{w}$ (top and bottom channel walls), $\Gamma_{c}$ (cylinder walls), $\Gamma_{in}$ (inlet) and $\Gamma_{out}$ (outlet)._

writevtk(model,datadir("perforated_plate"))

# ## FE Spaces
#
# ### Exercise 2
#
# _Define the test `FESpaces` for velocity and pressure, as described above. The velocity space should be a vector-valued lagrangian space of order `k` with appropriate boundary conditions. The pressure space should be an unconstrained lagrangian space of order `k-1`._

D = 2
k = 2
#sol=reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},k)
#sol=V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["inlet","walls","cylinder"])
#sol=reffeₚ = ReferenceFE(lagrangian,Float64,k-1)
#sol=Q = TestFESpace(model,reffeₚ,conformity=:C0)

# ### Exercise 3
# _Define the boundary conditions for velocity. You should define three functions `u_in`, `u_w` and `u_c` representing the prescribed dirichlet values at $\Gamma_{in}$, $\Gamma_w$ and $\Gamma_c$ respectively._
# 

#sol=const Uₘ = 0.3
#sol=const H  = 0.41
#sol=u_in(x) = VectorValue( 4 * Uₘ * x[2] * (H-x[2]) / (H^2), 0.0 )
#sol=u_w(x)  = VectorValue(0.0,0.0)
#sol=u_c(x)  = VectorValue(0.0,0.0)

# ### Exercise 4
# _Define the trial and test spaces for the velocity and pressure fields, as well as the corresponding multi-field spaces._

#hint=U = 
#hint=P = 
#hint=
#hint=Y = 
#hint=X = 

#sol=U = TrialFESpace(V,[u_in,u_w,u_c])
#sol=P = TrialFESpace(Q)
#sol=
#sol=Y = MultiFieldFESpace([V, Q])
#sol=X = MultiFieldFESpace([U, P])

# ## Nonlinear weak form and FE operator
#
# As usual, we start by defining the triangulations and measures we will need to define the weak form. In this case, we need to define the measure associate with the bulk $d\Omega$, as well as the measure associated with the outlet boundary $\Gamma_{out}$.

degree = k
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)

Γ_out = BoundaryTriangulation(model,tags="outlet")
n_Γout = get_normal_vector(Γ_out)
dΓ_out = Measure(Γ_out,degree)

# We also define the Reynolds number and functions to represent the convective term and its linearization.

const Re = 20
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

# ### Exercise 5
# _Define the weak form for our problem. You should start by defining the bilinear forms $a$ and $c$ and the trilinear form $dc$. Then use these components to build the residual $r$ and the jacobian $j$._

#hint=a((u,p),(v,q)) = 
#hint=
#hint=c(u,v) = 
#hint=dc(u,du,v) = 
#hint=
#hint=res((u,p),(v,q)) = 
#hint=jac((u,p),(du,dp),(v,q)) = 

#sol=a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
#sol=
#sol=c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
#sol=dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ
#sol=
#sol=res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v)
#sol=jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)

# We can finally define the `FEOperator` as usual

op = FEOperator(res,jac,X,Y)

# ## Solver and solution
#
# ### Exercise 6
#
# _Create a nonlinear Newton-Raphson solver and solve the problem. Print the solutions to a `.vtk` file and examine the obtained solution._
#

using LineSearches: BackTracking
#sol=solver = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())
#sol=
#sol=uh, ph = solve(solver,op)
#sol=
#sol=out_file = datadir("ins")
#sol=writevtk(Ω,out_file,cellfields=["uh"=>uh,"ph"=>ph])

# ## References
# 
# This tutorial follows test case 2D-1 from (Schafer,1996)[https://link.springer.com/chapter/10.1007/978-3-322-89849-4_39]
