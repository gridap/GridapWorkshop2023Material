
using DrWatson
using PackageCompiler

pkgs = ["Gridap","PartitionedArrays","GridapDistributed","GridapPETSc"]

# Precompilation with -O3 (in serial)
pdir = projectdir()
cmd = `julia --project=. -O3 -e 'using GridapPETSc'`
run(cmd)

# Compilation with -O3 (might be parallel)
create_sysimage(
  sysimage_path=projectdir("GridapWorkshop.so"),
  precompile_execution_file=projectdir("compile/warmup.jl")
)
