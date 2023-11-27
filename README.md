+++
title = "Software install instructions and material"
+++
 
# GridapWorkshop2023Material

*Latest update: 23/11/2023*

The practical part of the workshop will consist of instructors' guided hands-on tutorials and exercises.
This will be carried out  either
on the attendees laptops (mostly first day) or the [Gadi supercomputer](https://opus.nci.org.au/display/Help/Gadi+User+Guide) at 
NCI (mostly second day). You will find below the instructions to set up the software environment required in both cases.
For the Gadi supercomputer, you will get an invitation from the instructors to create an account prior to the 
workshop.

**NOTE**: In the case you have any issues while following these instructions, please fill an issue [here](https://github.com/gridap/GridapWorkshop2023Material/issues) 
and we will try to help you out. This will also help other participants that may have the same issue.

## Required software

Before being able to work on the workshop material, you will need to install the following software on your laptop:

- Install [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), the version control system we will use. **For Windows users, we strongly recommend installing [git for Windows](https://gitforwindows.org/). This will also install a bash shell, which will allow you to follow the rest of the installation instructions verbatim.**
- Download and install Julia based on the platform you are using from the [Julia](https://julialang.org/downloads/platform/) home page.
- Download and install VSCode based on the platform you are using from the [VSCode](https://code.visualstudio.com/download) home page. 
- [Install](https://www.julia-vscode.org/docs/dev/gettingstarted/#Installing-the-Julia-extension) and [configure](https://www.julia-vscode.org/docs/dev/gettingstarted/#Configuring-the-Julia-extension) the [Julia extension for VSCode](https://code.visualstudio.com/docs/languages/julia). Some interesting features of the Julia extension for VSCode are covered in the following [YouTube](https://www.youtube.com/watch?v=IdhnP00Y1Ks&t=125s) video.
- Install [ParaView](https://www.paraview.org/download/) post-processing software. We will use the basic features of ParaView. In any case, if you are 
  keen on learning more,  there is a whole [YouTube channel](https://www.youtube.com/playlist?list=PLvkU6i2iQ2fpcVsqaKXJT5Wjb9_ttRLK-) on ParaView that will give you many more ideas. 
- Finally, you will need an **ssh** client to connect to Gadi. Generally, every modern OS should have one installed by default. To check if you have one, open a terminal and type `ssh`. If a message like `usage: ssh ...` appears, you are good to go.

## Getting the workshop material

The workshop material is available as a git repository [here](https://github.com/gridap/GridapWorkshop2023Material). You can either download it as a zip file or clone the repository using git. We strongly recommend the latter as you will be able to automatically pull the most up-to-date changes as per required.

If your system has an ssh client, you can clone the repository using the following command

```bash
git clone git@github.com:gridap/GridapWorkshop2023Material.git
```

from the terminal. In order this command to be successful, you will need to generate a pair of public/private SSH keys, and then associate the public key to your GitHub account. You may find some instructions on how to do this [here](https://github.com/MonashMath/SCI1022/blob/master/Git.md#182-connecting-to-github-with-ssh-keys). If, for whatever reason, you are not able to succeed, you may also try:

```bash
git clone https://github.com/gridap/GridapWorkshop2023Material.git
```

which does not require the aforementioned pair of public/private SSH keys.

Alternative methods to clone the repository can be found [here](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

Once you have cloned the repository, you can pull the most up-to-date changes with the following command executed from the root directory of the repository:

```bash
git pull origin
```

You may expect changes in the tutorials and exercises till the very last minute. These instructions are mostly definitive, though.

## Setting up the environment on your local computer from the terminal

Move into the newly cloned repository and start Julia from the terminal by typing

```bash
julia --project=.
```

Then, press `]` to enter the package manager and run

```julia
(GridapWorkshop) pkg> instantiate
(GridapWorkshop) pkg> update
(GridapWorkshop) pkg> build
```

to install and precompile all the packages needed for the workshop. This may take a while.

To render Jypyter notebooks interactively, you also need to run the following:

```bash
  julia -e'using Pkg; Pkg.add("IJulia")'
```

More information on Julia Environments can be found [here](https://pkgdocs.julialang.org/v1/environments/).

## Setting up the environment on your local computer with VSCode

1. Open VSCode. Then, on the top menu, select `File->Open Folder`, and select the workshop's material folder you just cloned.
2. Ensure that the Julia environment in the bottom status bar of VSCode is `GridapWorkshop`. Click [here](https://www.julia-vscode.org/docs/dev/userguide/env/#Julia-Environments) for instructions on how to do that.
3. Open the Julia REPL in VSCode. To this end, open the command palette with the keyboard key combination `Crtl+Shift+P`.
4. On the command palette, type `"julia"`. You should get a drop-down list with different options. Select `Julia: Start REPL` option. This should open the Julia REPL on the VSCode's terminal window at the bottom.
5. Run the `instantiate` package manager command as described in the previous section. 

## Setting up the environment on Gadi

First, we will need to log into Gadi. You will receive an email at some point with an invitation to create an account. At the end of the process, you will get a username and password (which you should change). If your username is `aaa777`, you can connect to Gadi by typing

```bash
ssh -X aaa777@gadi.nci.org.au
```

When prompted, enter your password. You should now be logged into Gadi and located in your home directory. We would also recommend setting up passwordless ssh access to Gadi (but it is not required).

In addition to your home directory, you have access to a scratch space, a project-wide shared filesystem that is optimized for parallel access. This is where we will be working during the workshop. Start by creating a personal folder within the project's scratch space and linking it back to your home directory:

```bash
mkdir /scratch/vp91/$USER
ln -s /scratch/vp91/$USER $HOME/scratch
```

Move into your newly created scratch directory and clone the workshop repository as described above. Once the workshop repository is cloned, move into the `/gadi` subdirectory. This directory contains the distributed codes we will be using. Load the Gadi environment by running

```bash
source modules.sh
```

The script `modules.sh` has several purposes, and needs to be sourced every time you log in. It loads the Julia module, the Intel-MPI modules, and sets up some environment variables we will need.

Next, repeat the steps described in the previous section to setup the Julia environment in serial.
In addition, you will need to setup the environment for parallel. To do so, run the following commands on the terminal:

```bash
julia -e'using Pkg; Pkg.add("MPIPreferences")'
julia --project=. -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
julia --project=. -e 'using Pkg; Pkg.build()'
```

The first line sets up `MPI.jl` to work with the Intel-MPI binaries installed on Gadi instead of the Julia-specific artifacts. The second line does the same for `GridapPETSc.jl` and `GridapP4est.jl`, which are the Gridap wrappers for PETSc and P4est, respectively.

## Accessing to Gadi using ARE (graphical interface on the web browser)

For those users which are more comfortable with graphical interfaces, the instructions in the previous section can also be followed on a terminal opened in your web browser using the so-called [ARE-Australian Research Environment](https://opus.nci.org.au/display/Help/ARE+User+Guide). ARE will also allow you to open a Virtual Desktop on the web browser, and run ParaView on Gadi, so that you can visualize the results of your simulations without having to download the data files to your local computer (which might be heavy for large scale problems). You can find ARE usage instructions [here](https://opus.nci.org.au/display/Help/ARE+User+Guide).

## Creating a system image

Unfortunately, there is no parallel distributed version of the Julia REPL. This means running MPI codes interactively is not possible. Moreover, Julia notoriously suffers from long TTFX (Time To First eXecution) times due to Just-In-Time (JIT) compilation. Although this issue is being the focus of the latest releases, it can still be tedious to work within an edit-run-debug cycle.

To alleviate this problem, we will create a system image that contains all the packages we will need during the workshop. This will allow us to start Julia with the system image preloaded, and thus avoid the long TTFX times.

The `/gadi/compilation` directory contains files that allow us to do just that using [`PackageCompiler.jl`](https://julialang.github.io/PackageCompiler.jl/stable/). Although we will be creating a sysimage (to be used with Julia), this package also has options to create dynamic libraries (to be used with C/C++/Fortran) and executables.

First, we will need to install `PackageCompiler` package. It does not need to be installed within the project, so just run

```bash
julia -e 'using Pkg; Pkg.add("PackageCompiler")'
```

Next, run the following commands from the `/gadi` directory to launch a remote job that creates the system image

```bash
qsub compilation/compile.sh
```

This will take a while (around 30 mins), and will create a system image `GadiTutorial.so` in the `/gadi` directory. You can see the status of the job by running `qstat`.

You can then test this sysimage by running

```bash
julia --project=. -JGadiTutorial.so compilation/warmup.jl
```
