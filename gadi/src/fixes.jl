
# GridapDistributed transient fixes

Gridap.Arrays.evaluate!(cache,k::Operation,a::GridapDistributed.TransientDistributedCellField,b::GridapDistributed.DistributedCellField) = evaluate!(cache,k,a.cellfield,b)
Gridap.Arrays.evaluate!(cache,k::Operation,a::GridapDistributed.DistributedCellField,b::GridapDistributed.TransientDistributedCellField) = evaluate!(cache,k,a,b.cellfield)

Base.:(∘)(f::Function,g::Tuple{GridapDistributed.TransientDistributedCellField,GridapDistributed.DistributedCellField}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{GridapDistributed.DistributedCellField,GridapDistributed.TransientDistributedCellField}) = Operation(f)(g[1],g[2])

# GridapDistributed visualization fixes 

function GridapDistributed._prepare_fdata(trians,a)
  _fdata(v,trians) = local_views(v)
  _fdata(v::AbstractArray,trians) = v
  #_fdata(v,trians) = map(ti->v,trians)
  if length(a) == 0
    return map(trians) do t
      Dict()
    end
  end
  ks = []
  vs = []
  for (k,v) in a
    push!(ks,k)
    push!(vs,_fdata(v,trians))
  end
  map(vs...) do vs...
    b = []
    for i in 1:length(vs)
      push!(b,ks[i]=>vs[i])
    end
    b
  end
end

# GridapPETSc fixes

function Gridap.Algebra.solve!(x::T,
                               nls::PETScNonlinearSolver,
                               op::Gridap.Algebra.NonlinearOperator,
                               cache::GridapPETSc.PETScNonlinearSolverCache{<:T}) where T <: AbstractVector
  @check_error_code GridapPETSc.PETSC.SNESSolve(cache.snes[],C_NULL,cache.x_petsc.vec[])
  copy!(x,cache.x_petsc)
  cache
end
