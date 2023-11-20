using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays

# This is a distributed version of Poisson

function main(distribute,np)
  ranks  = distribute(LinearIndices((prod(np),)))

  options = "-ksp_type cg -pc_type gamg -ksp_monitor"
  GridapPETSc.with(args=split(options)) do
    domain = (0,1,0,1)
    mesh_partition = (20,20)
    model = CartesianDiscreteModel(ranks,np,domain,mesh_partition)

    order = 2
    u((x,y)) = (x+y)^order
    f(x) = -Δ(u,x)
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe,dirichlet_tags="boundary")
    U = TrialFESpace(u,V)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)
    a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
    l(v) = ∫( v*f )dΩ
    op = AffineFEOperator(a,l,U,V)

    solver = PETScLinearSolver()
    #uh = solve(solver,op)
    ss = symbolic_setup(solver,get_matrix(op))
    ns = numerical_setup(ss,get_matrix(op))
    x = pfill(0.0,partition(axes(get_matrix(op),2)))
    solve!(x,ns,get_vector(op))
    #writevtk(Ω,"results",cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
  end
end

with_mpi() do distribute
  main(distribute,(2,2))
end
