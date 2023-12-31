# In this **exercise**, we will learn
#
#  - How to solve transient multi-field PDEs in Gridap
#
# ## Problem statement
#
# We consider now as model problem the transient Stokes equations. We will assume the solution of the problem is given by the velocity $u(x,t) = t(x_1,x_2)^T$ and pressure $p(x,t) = t(x_1-x_2)$. Thus, the PDE we want to solve is: find the velocity vector $u$ and the pressure $p$ such that
#
# $$
# \left\lbrace
# \begin{aligned}
# \frac{\partial u(t)}{\partial t} - \Delta u(t) + \nabla p &= f(t) & \text{ in }\Omega, \\
# \nabla \cdot u(t) &= g(t) & \text{ in } \Omega, \\
# u(x,t) &= t(x_1,x_2)^T & \text{ on } \partial\Omega, \\
# u(x,0) &= 0.0 & \text{ in } \Omega, \\
# p(x,0) &= 0.0 & \text{ in } \Omega,
# \end{aligned}
# \right.
# $$
#
# where the computational domain is the unit square $\Omega \doteq (0,1)^d$, $d=2$, and $f$ and $g$ are the source terms that can be easily computed from the expressions of $u$ and $p$.
#
# We impose Dirichlet boundary conditions for the velocity on the entire boundary $\partial\Omega$ and constrain the mean value of the pressure to zero in order have a well-posed problem.
#
# ## Numerical scheme
#
# In order to approximate this problem **in space** we choose a formulation based on inf-sup stable $\boldsymbol{Q}_k/Q_{k-1}$ elements with continuous velocities and pressure pairs, the so called Taylor-Hood FEs. The interpolation spaces are defined as follows. The velocity interpolation space is
#
# $$
# V \doteq \{ v \in [C^0(\Omega)]^d:\ v|_T\in [Q_k(T)]^d \text{ for all } T\in\mathcal{T} \},
# $$
# where $T$ denotes an arbitrary cell of the FE mesh $\mathcal{T}$, and $Q_k(T)$ is the local polynomial space in cell $T$ defined as the multi-variate polynomials in $T$ of order less or equal to $k$ in each spatial coordinate. This is the usual continuous vector-valued Lagrangian FE space of order $k$ defined on a mesh of quadrilaterals or hexahedra. Likewise, the space for the pressure is
#
# $$
# \begin{aligned}
# Q_0 &\doteq \{ q \in Q: \  \int_\Omega q \ {\rm d}\Omega = 0\}, \text{ with}\\
# Q &\doteq \{ q \in C^0(\Omega) :\ q|_T\in Q_{k-1}(T) \text{ for all } T\in\mathcal{T}\},
# \end{aligned}
# $$
# where functions in $Q_0$ are strongly constrained to have zero mean value.
#
# The weak form of the problem reads: find $(u,p) \in U_g(t) \times Q_0$ such that
#
# $$
#   m(t,(u,p),(v,q)) + a(t,(u,p),(v,q)) = b(t,(v,q)) \quad \forall (v,q) \in \ V_0 \times Q_0
# $$
#
# where $U_g(t)$ and $V_0$ are the set of functions in $V$ fulfilling the Dirichlet boundary condition $g(t)$ and $0$ on $\partial\Omega$ respectively. Here, $U_g(t)$ is a transient FE space, in the sense that the Dirichlet boundary value of functions in $U_g$ changes in time. The definition of $m(t,(u,p),(v,q))$, $a(t,(u,p),(v,q))$ and $b(t,(v,q))$ is as follows.
#
# $$
# \begin{aligned}
# m(t,(u,p),(v,q)) &= \int_\Omega \frac{\partial u}{\partial t} v \ d\Omega, \\
# a(t,(u,p),(v,q)) &= \int_{\Omega} \nabla u \cdot \nabla v \ {\rm d}\Omega - \int_{\Omega} (\nabla\cdot v) \ p \ {\rm d}\Omega + \int_{\Omega} q \ (\nabla \cdot u) \ {\rm d}\Omega, \\
# b(t,(v,q)) &= \int_\Omega f(t) \cdot v \ d\Omega + \int_\Omega g(t) \ q \ d\Omega
# \end{aligned}
# $$
#
# ## Creating the discrete model
#
# We start with the discretization of the computational domain. We consider a $50\times50$ Cartesian grid of the unit square.
#
# ### Exercise 1
#
# _Load Gridap and create a $50\times50$ Cartesian grid of the unit square._

#hint=# Solution of exercise 1
#sol= using Gridap
#sol= n = 50
#sol= domain = (0,1,0,1)
#sol= partition = (n,n)
#sol= model = CartesianDiscreteModel(domain,partition)

# ## Setting up multifield FE spaces
#
# ### Exercise 2
#
# _Create the test FE spaces of the problem._
#
# _For the velocities, we need to create the standard vector-valued continuous Lagrangian test FE space of order $k$. For the pressures, the standard scalar-valued continuous Lagrangian test FE space of order $k-1$ with zero mean value. We choose k = 2._
#
#hint= **Hints:** 
#hint= - The spaces of test functions are constant in time and are defined as in steady problems.
#hint= - Use the tag `boundary` to set up Dirichlet BCs for the velocity everywhere on the boundary $\partial \Omega$.

#hint=# Solution of exercise 2
#sol=D = 2
#sol=order = 2
#sol=reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
#sol=V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags="boundary")
#sol=reffeₚ = ReferenceFE(lagrangian,Float64,order-1)
#sol=Q = TestFESpace(model,reffeₚ,conformity=:H1,constraint=:zeromean)

# The trial space of the velocities is now a `TransientTrialFESpace`, which is constructed from a `TestFESpace` and a time-dependent function for the Dirichlet boundary condition.

u(x,t::Real) = t*VectorValue(x[1],x[2])
u(t::Real) = x -> u(x,t)

U = TransientTrialFESpace(V,u)

# We need to provide $u$ overloaded with two methods: (1) A function that evaluates $u$ at a given $(x,t)$ pair (needed, e.g., to output the solution) and (2) $u$ with the time argument only, such that it returns the space-only function for a given $t$ (needed, e.g., to compute the time derivative of $u$).
#
# Meanwhile, there is no time derivative operator acting on the pressure. Therefore, the pressure space is constant in time, and thus, defined as in steady problems.

P = TrialFESpace(Q)

# With all these ingredients we create the FE spaces representing the Cartesian product of the velocity and pressure FE spaces, i.e. the multifield FE space where we are seeking the solution the problem. The trial multifield FE space must be a transient one, since `U` is a transient FE space.

Y = MultiFieldFESpace([V, Q])
X = TransientMultiFieldFESpace([U, P])

# ## Triangulation and integration quadrature
#
# We define the triangulation and integration measure from the discrete model as usual:

degree = 2*order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)

# ## Defining the source terms of the problem
#
# Before writing down the weak form of the problem, we need to construct the time-dependent functions representing the source terms `f(t)` and `g(t)`, corresponding to the right-hand sides of the first and second equation of the problem.
# 
# Using the fact that the solution of the problem is $u(x,t) = t(x_1,x_2)^T$ and the pressure is $p(x,t) = t(x_1-x_2)$, we have that
#
# ```math
# f(x,t) = \frac{\partial u(x,t)}{\partial t} - \Delta u(x,t) + \nabla p = (x_1+t,x_2-t)^T \quad \text{and} \quad g(x,t) = 2t,
# ```
#
# where we recall that the pressure is a constant. We will define `f(x,t)` in two equivalent ways. First, we proceed in a conventional manner:
# 
# ### Exercise 3
#
# _Write the expressions for the vector-valued function `f` and the scalar function `g` as written above. Like `u` before, they must be time-dependent functions that return a space-only function._
#

#hint=# Solution of exercise 3
#sol=f(t::Real) = x -> VectorValue(x[1]+t,x[2]-t)
#sol=g(t::Real) = x -> 2*t

# Alternatively, we can use automatic differentiation to get directly `f` and `g`. In order to do that, we must define `p` before.

p(t::Real) = x -> t*(x[1]-x[2])

f_AD(t::Real) = x -> ∂t(u)(t)(x)-Δ(u(t))(x)+ ∇(p(t))(x)
g_AD(t::Real) = x -> (∇⋅u(t))(x)

# We can do some quick (non-exhaustive) checks to compare both alternatives:

@assert f(1.0)(Point(0.5,0.5)) == f_AD(1.0)(Point(0.5,0.5)) == VectorValue(1.5,-0.5)
@assert g(1.0)(Point(0.5,0.5)) == g_AD(1.0)(Point(0.5,0.5)) == 2.0

# ## Writing down the weak form and the FE operator of the problem
#
# The weak form of the problem follows the same structure as other `Gridap` tutorials, where we define the bilinear and linear forms to define the FE operator. In the most general case, we need to deal with time-dependent quantities and with the presence of time derivatives. Here, we exploit the fact that the problem is linear and use the transient Affine FE operator signature `TransientAffineFEOperator`. In that case, we handle time-dependent quantities by passing the time, $t$, as an additional argument to the form, i.e. $a(t,(u,p),(v,q))$. Meanwhile, we take care of the time derivative by defining $m$ as a mass contribution.
#
# ### Exercise 4
#
# _Write the bilinear forms `m`, `a` and the time-dependent linear form `b`. Recall that $m$ is expressed as a mass contribution._
#
#hint= **Hint:** The only variables needing an explicit time dependency are the source terms `f` and `g`.

#hint=# Solution of exercise 4
#sol=m(t,(ut,p),(v,q)) = ∫( ut⋅v )dΩ
#sol=a(t,(u,p),(v,q))  = ∫( ∇(u)⊙∇(v) - (∇⋅v)*p + q*(∇⋅u) )dΩ
#sol=b(t,(v,q))        = ∫( f(t)⋅v )dΩ + ∫( g(t)*q )dΩ

# With all these ingredients we can instantiate the `TransientAffineFEOperator` as:

op = TransientAffineFEOperator(m,a,b,X,Y)

# ## Setting up the transient FE solver
#
# We have already built the transient FE problem. Now, the remaining step is to solve it. First, we define a linear solver to be used at each time step. Here we use the `LUSolver`, but other choices are possible.

ls = LUSolver()

# Then, we define the ODE solver. That is, the scheme that will be used for the time integration. In this tutorial we use the 2nd order `ThetaMethod` ($\theta = 0.5$).

# ### Exercise 5
#
# _Define the ODE solver. Use a `ThetaMethod` with `ls` as the solver, $dt = 0.05$ and $\theta = 0.5$._
#
#hint= **Hint:** Use `methods(ThetaMethod)` to get the signature of the `ThetaMethod` constructor. Note that the `nls` variable in the constructor is a `NonlinearSolver` and `LinearSolver <: NonlinearSolver`.

#hint=# Solution of exercise 5
#sol=dt = 0.1
#sol=θ = 0.5
#sol=ode_solver = ThetaMethod(ls,dt,θ)

# Finally, we define the solution using the `solve` function, giving the ODE solver, the FE operator, an initial solution, an initial time and a final time. To construct the initial condition we interpolate the initial velocity and pressure into the FE space $X(t) = U(t) \times P$ at $t = 0.0$.

u₀ = interpolate_everywhere(u(0.0),U(0.0))
p₀ = interpolate_everywhere(p(0.0),P)

x₀ = interpolate_everywhere([u₀,p₀],X(0.0))
t₀ = 0.0
T = 1.0
xₕₜ = solve(ode_solver,op,x₀,t₀,T)

# ## Postprocessing

# We should highlight that `xₕₜ` is just an _iterable_ function and the results at each time steps are only computed when iterating over it, i.e., lazily. We can post-process the results and generate the corresponding `vtk` files using the `createpvd` and `createvtk` functions. The former will create a `.pvd` file with the collection of `.vtu` files saved at each time step by `createvtk`. The computation of the problem solutions will be triggered in the following loop:

using DrWatson
dir = datadir("transient_stokes")
!isdir(dir) && mkdir(dir)
createpvd(dir) do pvd
  for (xₕ,t) in xₕₜ
    (uₕ,pₕ) = xₕ
    file = dir*"/solution_$t"*".vtu"
    pvd[t] = createvtk(Ω,file,cellfields=["uh"=>uₕ,"ph"=>pₕ])
  end
end

# And visualise them in ParaView.
#
# ## References
#
# [Gridap Tutorial 17: Transient Poisson Equation](https://gridap.github.io/Tutorials/dev/pages/t017_poisson_transient/#Tutorial-17:-Transient-Poisson-equation-1)
#
# ### Bonus exercises
#
# 1. _The solution of this problem belongs to the FE space, since it is linear in space and time. Hence the FE solutions `uₕ` and `pₕ` should coincide with `u` and `p` at every time step. Compute the l2 norm of the errors for `uₕ` and `pₕ` inside the iteration over `xₕₜ` and check exacteness (up to arithmetical precision errors)._
# 2. _Write down the residual of the problem and solve it defining the operator as `op = TransientFEOperator(res,X,Y)`, i.e. with the Jacobian computed with automatic differentiation. You might find help on how to do this in [Gridap Tutorial 17: Transient Poisson Equation](https://gridap.github.io/Tutorials/dev/pages/t017_poisson_transient/#Tutorial-17:-Transient-Poisson-equation-1)._
#
# **Tutorial done!**