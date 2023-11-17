using PackageCompiler

create_sysimage([:GadiTutorial],
  sysimage_path=joinpath(@__DIR__,"..","GadiTutorial.so"),
  precompile_execution_file=joinpath(@__DIR__,"warmup.jl"))
