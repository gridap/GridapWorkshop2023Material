
function main_ins(;nprocs,                      # Number of processors
                   mesh="perforated_plate.msh", # Mesh file
                   options=OPTIONS_MUMPS,       # PETSc solver options
                   T=2.0                        # Final time
                  )
  nls_options = "-snes_monitor -ksp_error_if_not_converged true "
  options = string(nls_options,options)
  with_mpi() do distribute
    main_ins(distribute,nprocs,mesh,options,T)
  end
end

function ξ(t::Real,Tth::Real)
  (t <= Tth) ? sin(π*t/(2*Tth)) : 1.0
end

function inflow(x::VectorValue{2},t::Real,H,Tth)
  Uₘ = 1.5
  VectorValue(4*Uₘ*x[2]*(H-x[2])/(H^2) * ξ(t,Tth), 0.0 )
end

function inflow(x::VectorValue{3},t::Real,H,Tth)
  Uₘ = 2.25
  VectorValue(16*Uₘ*x[2]*x[3]*(H-x[2])*(H-x[3])/(H^4) * ξ(t,Tth), 0.0, 0.0)
end

function main_ins(distribute,nprocs,mesh,options,T)
  ranks = distribute(LinearIndices((prod(nprocs),)))

  msh_file = projectdir("meshes/$mesh")
  model = GmshDiscreteModel(ranks,msh_file)
  D = num_cell_dims(model)

  k = 2
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},k)
  reffeₚ = ReferenceFE(lagrangian,Float64,k-1)

  V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["inlet","walls","cylinder"])
  Q = TestFESpace(model,reffeₚ,conformity=:C0)

  H  = 0.41
  u_in(x,t::Real) = inflow(x,t,H,T)
  u_w(x,t::Real)  = zero(VectorValue{D,Float64})
  u_c(x,t::Real)  = zero(VectorValue{D,Float64})
  u_in(t::Real)   = x -> u_in(x,t)
  u_w(t::Real)    = x -> u_w(x,t)
  u_c(t::Real)    = x -> u_c(x,t)

  U = TransientTrialFESpace(V,[u_in,u_w,u_c])
  P = TrialFESpace(Q)

  Y = MultiFieldFESpace([V, Q])
  X = TransientMultiFieldFESpace([U, P])

  degree = 2*k
  Ω  = Triangulation(model)
  dΩ = Measure(Ω,degree)

  Re = 100.0
  conv(u,∇u) = Re*(∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

  m((u,p),(v,q)) = ∫( u⋅v )dΩ
  a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
  c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
  dc(u,du,v) = ∫( v⊙(conv∘(u,∇(du)) + conv∘(du,∇(u))) )dΩ

  res(t,(u,p),(v,q)) = m((∂t(u),p),(v,q)) + a((u,p),(v,q)) + c(u,v)
  jac(t,(u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)
  jac_t(t,(u,p),(dut,dpt),(v,q)) = m((dut,dpt),(v,q))
  op = TransientFEOperator(res,jac,jac_t,X,Y)

  GridapPETSc.with(args=split(options)) do
    nls = PETScNonlinearSolver()

    Δt = 0.01
    θ  = 0.5
    ode_solver = ThetaMethod(nls,Δt,θ)

    x₀ = interpolate_everywhere([zero(VectorValue{D,Float64}),0.0],X(0.0))
    t₀ = 0.0
    xₕₜ = solve(ode_solver,op,x₀,t₀,T)

    out_dir = datadir("ins")
    (i_am_main(ranks) && !isdir(out_dir)) && mkdir(out_dir)
    for (xₕ,t) in xₕₜ
      i_am_main(ranks) && println(" > Computing solution at time $t")
      uₕ, pₕ = xₕ
      out_file = joinpath(out_dir,"ins$(t)")
      writevtk(Ω,out_file,cellfields=["u"=>uₕ,"p"=>pₕ])
    end
  end
end
