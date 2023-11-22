
# GridapDistributed transient fixes

Gridap.Arrays.evaluate!(cache,k::Operation,a::GridapDistributed.TransientDistributedCellField,b::GridapDistributed.DistributedCellField) = evaluate!(cache,k,a.cellfield,b)
Gridap.Arrays.evaluate!(cache,k::Operation,a::GridapDistributed.DistributedCellField,b::GridapDistributed.TransientDistributedCellField) = evaluate!(cache,k,a,b.cellfield)

Base.:(∘)(f::Function,g::Tuple{GridapDistributed.TransientDistributedCellField,GridapDistributed.DistributedCellField}) = Operation(f)(g[1],g[2])
Base.:(∘)(f::Function,g::Tuple{GridapDistributed.DistributedCellField,GridapDistributed.TransientDistributedCellField}) = Operation(f)(g[1],g[2])

# GridapPETSc fixes

function Gridap.Algebra.solve!(x::T,
                               nls::PETScNonlinearSolver,
                               op::Gridap.Algebra.NonlinearOperator,
                               cache::GridapPETSc.PETScNonlinearSolverCache{<:T}) where T <: AbstractVector
  @check_error_code GridapPETSc.PETSC.SNESSolve(cache.snes[],C_NULL,cache.x_petsc.vec[])
  copy!(x,cache.x_petsc)
  cache
end
