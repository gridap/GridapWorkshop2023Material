module GadiTutorial

using DrWatson
using Gridap, GridapDistributed, GridapPETSc, GridapGmsh
using PartitionedArrays

export OPTIONS_CG_JACOBI, OPTIONS_CG_AMG, OPTIONS_MUMPS, OPTIONS_NEUTON_MUMPS
export main_poisson, main_ins

const OPTIONS_CG_JACOBI = "-pc_type jacobi -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_CG_AMG = "-pc_type gamg -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_MUMPS = "-pc_type lu -ksp_type preonly -ksp_converged_reason -pc_factor_mat_solver_type mumps"

include("fixes.jl")
include("poisson_driver.jl")
include("ins_driver.jl")

end # module GadiTutorial
