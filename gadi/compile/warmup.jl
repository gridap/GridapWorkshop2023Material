
#cmd = `
#  mpiexec -n 4 julia --project=. -O3 -e'using GadiTutorial; main_poisson((2,2))'
#`
#run(cmd)

using GadiTutorial
main_poisson(;nprocs=(1,1))
