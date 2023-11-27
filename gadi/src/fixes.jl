# GridapPETSc fixes

function Gridap.Algebra.solve!(x::T,
                               nls::PETScNonlinearSolver,
                               op::Gridap.Algebra.NonlinearOperator,
                               cache::GridapPETSc.PETScNonlinearSolverCache{<:T}) where T <: AbstractVector
  @check_error_code GridapPETSc.PETSC.SNESSolve(cache.snes[],C_NULL,cache.x_petsc.vec[])
  copy!(x,cache.x_petsc)
  cache
end
