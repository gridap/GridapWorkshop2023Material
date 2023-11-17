module GadiTutorial

using DrWatson
using Gridap, GridapDistributed, GridapPETSc
using PartitionedArrays

export OPTIONS_CG_JACOBI, OPTIONS_CG_AMG, OPTIONS_MUMPS
export main_poisson

const OPTIONS_CG_JACOBI = "-pc_type jacobi -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_CG_AMG = "-pc_type gamg -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_MUMPS = "-pc_type lu -ksp_type preonly -ksp_converged_reason -pc_factor_mat_solver_type mumps"

include("poisson_driver.jl")

end # module GadiTutorial
