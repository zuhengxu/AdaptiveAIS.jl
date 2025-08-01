module AdaptiveAIS

using Base: @kwdef
using Base.Threads: @threads

using Random, Distributions, LinearAlgebra, LogExpFunctions, StatsBase, Statistics
using LogDensityProblems
using ProgressMeter
using SimpleUnPack

using Pigeons
using Pigeons: @auto


include("SplitRandom.jl")
export SplitRandomArray
include("logdensity.jl")

include("path.jl")
export LinearPath, AbstractPath

using Functors
include("aisproblem.jl")
export AISProblem, AIS_output
export log_density_ratio, log_density_ratio!, log_density_reference, log_density_target
export log_annealed_density, log_annealed_gradient, log_annealed_density_and_gradient
export iid_sample, iid_sample_reference

include("schedule.jl")
export AbstractScheduler, FixedSchedule, schedule_length
export OnlineScheduling, DebiasOnlineScheduling


include("utils.jl")


include("barriers.jl")
export find_optimal_schedule

include("particles.jl")
export Particles

include("report.jl")
export report, process_result

# transition kernels
include("kernels/Kernels.jl")
export TransitionKernel, step!, step, mutate!, mutate
export RWMH_sweep, CoordSliceSampler

# ais algorithms with different schedule adapting schemes
include("ais_fixsched.jl")
export ais

# round based tuning using Saif's method: "Optimised Annealed Sequential Monte Carlo Samplers"
include("sais.jl")
export SAIS

#############################
# online temperature selection 
############################

include("ais_online.jl")
# 1. line search for temperature selection: "Toward automatic model comparison: An adaptive sequential Monte Carlo approach"
# 2. connection with mirror descent: "A connection between Tempering and Entropic Mirror Descent"
# 3. constant divergence rate progression:  "Adaptive Annealed Importance Sampling with Constant Rate Progress"
include("online_scheduling.jl")
export MirrorDescent, LineSearch, ConstantRateProgress

end
