module purge
module load pbs

module load intel-compiler-llvm/2023.2.0
module load intel-mpi/2021.10.0
module load intel-mkl/2023.2.0

export MPI_VERSION="intel-$INTEL_MPI_VERSION"
export JULIA_MPI_PATH=$INTEL_MPI_ROOT

# Julia setup
export PROJECT="vp91"

export PATH=/scratch/$PROJECT/gridap-workshop/julia/julia-1.8.5/bin:$PATH
SCRATCH="/scratch/$PROJECT/$USER"
export JULIA_DEPOT_PATH="$SCRATCH/.julia"

PETSC_VERSION='3.19.5'
P4EST_VERSION='2.2'
INSTALL_ROOT="/scratch/$PROJECT/gridap-workshop/"
export JULIA_PETSC_LIBRARY="$INSTALL_ROOT/petsc/$PETSC_VERSION-$MPI_VERSION/lib/libpetsc"
export P4EST_ROOT_DIR="$INSTALL_ROOT/p4est/$P4EST_VERSION-$MPI_VERSION"
