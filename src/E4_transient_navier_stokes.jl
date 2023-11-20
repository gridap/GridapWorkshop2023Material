# In this tutorial, we will learn
#
#  - How to solve nonlinear multi-field PDEs in Gridap
#
# ## Problem statement
#
# The goal is to solve a nonlinear multi-field PDE. As a model problem, we consider a well known benchmark in computational fluid dynamics, the lid-driven cavity for the incompressible Navier-Stokes equations.
#
# Formally, the PDE we want to solve is: find the velocity vector $u$ and the pressure $p$ such that
#
# $$
# \left\lbrace
# \begin{aligned}
# \frac{\partial u(t)}{\partial t} - \Delta u + \mathit{Re}\ (u\cdot \nabla)\ u + \nabla p = 0 &\text{ in }\Omega,\\
# \nabla\cdot u = 0 &\text{ in } \Omega,\\
# u = g &\text{ on } \partial\Omega,
# \end{aligned}
# \right.
# $$
#
# where the computational domain is the unit square $\Omega \doteq (0,1)^d$, $d=2$, and $\mathit{Re}$ is the Reynolds number. In this example, the driving force is the Dirichlet boundary velocity $g$, which is a non-zero horizontal velocity with a value of $g = (1,0)^t$ on the top side of the cavity, namely the boundary $(0,1)\times\{1\}$, and $g=0$ elsewhere on $\partial\Omega$. Since we impose Dirichlet boundary conditions on the entire boundary $\partial\Omega$, the mean value of the pressure is constrained to zero in order have a well posed problem,
#
# $$
# \int_\Omega q \ {\rm d}\Omega = 0.
# $$
#
# ## Numerical Scheme
#
# In order to approximate this problem we chose a formulation based on inf-sub stable $Q_k/P_{k-1}$ elements with continuous velocities and discontinuous pressures. The interpolation spaces are defined as follows. The velocity interpolation space is
#
# $$
# V \doteq \{ v \in [C^0(\Omega)]^d:\ v|_T\in [Q_k(T)]^d \text{ for all } T\in\mathcal{T} \},
# $$
# where $T$ denotes an arbitrary cell of the FE mesh $\mathcal{T}$, and $Q_k(T)$ is the local polynomial space in cell $T$ defined as the multi-variate polynomials in $T$ of order less or equal to $k$ in each spatial coordinate. Note that, this is the usual continuous vector-valued Lagrangian FE space of order $k$ defined on a mesh of quadrilaterals or hexahedra.  On the other hand, the space for the pressure is
#
# $$
# \begin{aligned}
# Q_0 &\doteq \{ q \in Q: \  \int_\Omega q \ {\rm d}\Omega = 0\}, \text{ with}\\
# Q &\doteq \{ q \in L^2(\Omega):\ q|_T\in P_{k-1}(T) \text{ for all } T\in\mathcal{T}\},
# \end{aligned}
# $$
# where $P_{k-1}(T)$ is the polynomial space of multi-variate polynomials in $T$ of degree less or equal to $k-1$. Note that functions in $Q_0$ are strongly constrained to have zero mean value. This is achieved in the code by removing one degree of freedom from the (unconstrained) interpolation space $Q$ and  adding a constant to the computed pressure so that the resulting function has zero mean value.
#
# The weak form associated to these interpolation spaces reads: find $(u,p)\in U_g \times Q_0$ such that $[r(u,p)](v,q)=0$ for all $(v,q)\in V_0 \times Q_0$
# where $U_g$ and $V_0$ are the set of functions in $V$ fulfilling the Dirichlet boundary condition $g$ and $0$  on $\partial\Omega$ respectively. The weak residual $r$ evaluated at a given pair $(u,p)$ is the linear form defined as
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
# The implementation of this numerical scheme is done in Gridap by combining the concepts previously seen for single-field nonlinear PDEs  and linear multi-field problems.
#
# ## Creating the discrete model
#
# We start with the discretization of the computational domain. We consider a $100\times100$ Cartesian mesh of the unit square.
#
# ### Exercise 1
# 
# _Create a discrete model of a $100\times100$ Cartesian mesh of the unit square. Then create two new boundary tags,  namely `"diri1"` and `"diri0"`, one for the top side of the square (where the velocity is non-zero), and another for the rest of the boundary (where the velocity is zero)._
#hint= **Hint:** Use the function `add_tag_from_tags!` to construct new boundary tags from pre-existing cartesian tags. Remember that the cartesian tags are in lexicographic order.
#hint= **Hint:** You can also print the model to `.vtk` and visualize the tags with Paraview.

using Gridap
using GridapGmsh

msh_file = projectdir("meshes/perforated_plate.json")
model = DiscreteModelFromFile(msh_file)

# ## Setting up multifield FE spaces
#
# For the velocities, we need to create a conventional vector-valued continuous Lagrangian FE space.
#
# ### Exercise 2
#
# _Create a standard vector-valued continuous Lagrangian test FE space of second order that is constrained at the `diri0` and `diri1` regions._

#hint=# Solution of exercise 2
D = 2
order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["inlet","noSlip","hole"])

# For the pressure, we instantiate a linear discontinuous FE space of functions strongly constrained to have zero mean value.

reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
Q = TestFESpace(model,reffeₚ,conformity=:L2,constraint=:zeromean)

# With the options `:Lagrangian`, `space=:P`, `valuetype=Float64`, and `order=order-1`, we select the local polynomial space $P_{k-1}(T)$ on the cells $T\in\mathcal{T}$. With the symbol `space=:P` we specifically chose a local Lagrangian interpolation of type "P". Without using `space=:P`, would lead to a local Lagrangian of type "Q" since this is the default for quadrilateral or hexahedral elements. On the other hand, `conformity=:L2,constraint=:zeromean` leads to a FE space, whose functions are constrained to have mean value equal to zero, which is just what we need for the pressure space.
#
# ### Exercise 3
#
# _Create the trial velocity and pressure FE spaces._
#
#hint= **Hint:** Remember to create the functions prescribing the Dirichlet values at the `diri0` and `diri1` regions.

#hint=# Solution of exercise 3

const Uₘ = 1.0
const H  = 0.41
u_in(x,t::Real) = VectorValue( 1.5 * Uₘ * x[2] * ( H - x[2] ) / ( (H/2)^2 ), 0.0 )
u_0(x,t::Real)  = VectorValue(0,0)
u_in(t::Real)   = x -> u_in(x,t)
u_0(t::Real)    = x -> u_0(x,t)

U = TransientTrialFESpace(V,[u_in,u_0,u_0])
P = TrialFESpace(Q)

# With all these ingredients we create the FE spaces representing the Cartesian product of the velocity and pressure FE spaces, which is none other than the multifield FE space where we are seeking the solution the problem.

Y = MultiFieldFESpace([V, Q])
X = TransientMultiFieldFESpace([U, P])

# ## Triangulation and integration quadrature
#
# From the discrete model we can define the triangulation and integration measure

degree = order
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)

Γ_out = BoundaryTriangulation(model,tags="outlet")
n_Γout = get_normal_vector(Γ_out)
dΓ_out = Measure(Γ_out,degree)

# ## Nonlinear weak form and FE operator
#
# The different terms of the nonlinear weak form for this example are defined using the notation for multi-field problems.

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

# The bilinear form reads
hN(x) = VectorValue( 0.0, 0.0 )
l_out(v) = ∫( v⋅hN )dΓ_out

m((ut,p),(v,q)) = ∫( ut⋅v )dΩ
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ

# Note that, since we are using a Cartesian product FE space, it's elements are tuples. Here we use `(u,p)` and `(v,q)` to denote the trial and test functions.
#
# The nonlinear term and its Jacobian are given by

c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

# The first argument of function `dc` stands for function $u\in U_g$, where the Jacobian is evaluated. The second argument represents an arbitrary direction $\delta u \in V_0$, and the third argument is a test function $v\in V_0$.
#
# Note that we can still define forms in terms of one of the FE spaces, as done with the convection term with the velocities and combine them later into the residual and Jacobian multifield forms.
#
# Finally, the Navier-Stokes weak form residual and Jacobian can be defined as

res(t,(u,p),(v,q)) = m((u,p),(v,q)) + a((u,p),(v,q)) + c(u,v) - l_out(v)
jac(t,(u,p),(du,dp),(v,q)) = m((du,dp),(v,q)) + a((du,dp),(v,q)) + dc(u,du,v)
jac_t(t,(u,p),(dut,dpt),(v,q)) = m((dut,dpt),(v,q))

op = TransientFEOperator(res,X,Y)

# Here, we have constructed an instance of `FEOperator`, which is the type that represents a general nonlinear FE problem in Gridap. The constructor takes the functions representing the weak residual and Jacobian, and the test and trial spaces. If only the function for the residual is provided, the Jacobian is computed internally with automatic differentiation.

# ## Nonlinear solver phase
#
# We have already built the nonlinear FE problem. Now, the remaining step is to solve it. In Gridap, nonlinear (and also linear) FE problems can be solved with instances of the type `FESolver`.
#
# We construct an instance of `FESolver` as follows:

using LineSearches: BackTracking
nls = NLSolver(show_trace=true, method=:newton, linesearch=BackTracking())

Δt = 0.05
θ = 0.5
ode_solver = ThetaMethod(LUSolver(),Δt,θ)

u₀ = zero(X(0.0))
t₀ = 0.0
T = 10.0
xₕₜ = solve(ode_solver,op,u₀,t₀,T)

# Finally, we write the results for visualization (see next figure).

dir = datadir("ins_stokes_transient")
!isdir(dir) && mkdir(dir)
createpvd(dir) do pvd
  for (xₕ,t) in xₕₜ
    uₕ,pₕ = xₕ
    file = dir*"/solution_$t"*".vtu"
    pvd[t] = createvtk(Ω,file,cellfields=["u"=>uₕ,"p"=>pₕ])
  end
end
