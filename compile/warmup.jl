using DrWatson

pdir   = projectdir()
driver = projectdir("src/poisson_dist.jl")

cmd = `
  mpiexec -n 4 julia --project=$pdir -O3 $driver
`
run(cmd)
