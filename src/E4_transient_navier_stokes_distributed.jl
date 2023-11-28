
# The aim of this exercise will be to modify the Transient Incompressible Navier-Stokes driver to run in parallel, while doing so interactively using the debug-mode provided by `PartitionedArrays` and `GridapDistributed`. 

# We start by importing the libraries we will need:

using Gridap, GridapDistributed, GridapGmsh, PartitionedArrays
using DrWatson

# Next, we will create our distributed processor ranks, in debug mode: 

np = 2
ranks = with_debug() do distribute
  distribute(LinearIndices((np,)))
end

# ### Exercise 1
# 
# _By using the code in the previous exercise, load the mesh from the file `perforated_plate_tiny.msh`. Distribute the mesh between different processors by passing `ranks` as an additional input to the constructor, i.e_
# ```julia
#   model = GmshDiscreteModel(ranks,filename)
# ```

#hint=# model = 
#sol=msh_file = projectdir("meshes/perforated_plate_tiny.msh")
#sol=model = GmshDiscreteModel(ranks,msh_file)

# ### Exercise 2
# 
# _By using the code in the previous exercise, define the boundary conditions and the trial and test finite-element spaces. The same code that worked in serial should work here as well_
#

#sol=k = 2
#hint=# reffeᵤ = 
#hint=# reffeₚ = 
#sol=reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
#sol=reffeₚ = ReferenceFE(lagrangian,Float64,k-1)

#hint=# V = 
#hint=# Q = 
#sol=V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["inlet","walls","cylinder"])
#sol=Q = TestFESpace(model,reffeₚ,conformity=:C0)
#sol=
const Tth = 1
const Uₘ = 1.5
const H  = 0.41
ξ(t) = (t <= Tth) ? sin(π*t/(2*Tth)) : 1.0
u_in(x,t::Real) = VectorValue( 4 * Uₘ * x[2] * (H-x[2]) / (H^2) * ξ(t), 0.0 )
u_w(x,t::Real)  = VectorValue(0.0,0.0)
u_c(x,t::Real)  = VectorValue(0.0,0.0)
u_in(t::Real)   = x -> u_in(x,t)
u_w(t::Real)    = x -> u_w(x,t)
u_c(t::Real)    = x -> u_c(x,t)

#hint=# U = 
#hint=# P = 
#hint=# Y = 
#hint=# X = 
#sol=
#sol=U = TransientTrialFESpace(V,[u_in,u_w,u_c])
#sol=P = TrialFESpace(Q)
#sol=
#sol=Y = MultiFieldFESpace([V, Q])
#sol=X = TransientMultiFieldFESpace([U, P])

# As usual, we define the triangulations and measures that we need to integrate:

degree = 2*k
Ω  = Triangulation(model)
dΩ = Measure(Ω,degree)

Γ_out = BoundaryTriangulation(model,tags="outlet")
n_Γout = get_normal_vector(Γ_out)
dΓ_out = Measure(Γ_out,degree)

# We can then proceed to implement our weak form: As you can see, it is done in the same way as in serial. Unfortunately, automatic differentiation is (yet) working on parallel. This means we will have to define the jacobians explicitly: 

const Re = 100.0
conv(u,∇u) = Re*(∇u')⋅u

m((u,p),(v,q)) = ∫( u⋅v )dΩ
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(conv∘(u,∇(du)) + conv∘(du,∇(u))) )dΩ

res(t,(u,p),(v,q)) = m((∂t(u),p),(v,q)) + a((u,p),(v,q)) + c(u,v)
jac(t,(u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)
jac_t(t,(u,p),(dut,dpt),(v,q)) = m((dut,dpt),(v,q))
op = TransientFEOperator(res,jac,jac_t,X,Y)

# We now define the nonlinear solver. Unfortunately, the solvers in `NLSolvers.jl` are not designed for distributed programming. We will therefore have to rely on Gridap's implementation of the Newton-Raphson solver:

using Gridap.Algebra
nls = NewtonRaphsonSolver(LUSolver(),1.e-6,10)

# ### Exercise 3
#
# _By using the code from the previous exercise, create the transient solver. In this exercise you should use the `ThetaMethod` with $\theta = 0.5$ and a time step size $\Delta t = 0.01$._

#hint=# Δt = 
#hint=# θ  = 
#hint=# ode_solver = 

#sol=Δt = 0.01
#sol=θ  = 0.5
#sol=ode_solver = ThetaMethod(nls,Δt,θ)

# We can finally solve the problem and print the solutions: 

x₀ = interpolate_everywhere([VectorValue(0.0,0.0),0.0],X(0.0))
t₀ = 0.0
T  = Tth
xₕₜ = solve(ode_solver,op,x₀,t₀,T)

dir = datadir("ins_distributed")
i_am_main(ranks) && !isdir(dir) && mkdir(dir)
for (xₕ,t) in xₕₜ
  println(" > Computing solution at time $t")
  uₕ,pₕ = xₕ
  file = dir*"/solution_$t"*".vtu"
  writevtk(Ω,file,cellfields=["u"=>uₕ,"p"=>pₕ])
end
