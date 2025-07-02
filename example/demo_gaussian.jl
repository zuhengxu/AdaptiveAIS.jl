using AdaptiveAIS
using LinearAlgebra
using Distributions, Random, LogDensityProblems, LogDensityProblemsAD
using Mooncake, DifferentiationInterface, ADTypes

const AIS = AdaptiveAIS

include("logdensityprobs.jl")


############################
# set up the problem
############################
dim = 10
L = LinearPath()
μ = 10
sigma = 0.2
p0 = MvNormal(zeros(dim), I)
p1 = MvNormal(μ*ones(dim), sigma*I)
prob = AISProblem(p0, p1, L)

##################### 
# testing sais
#####################
# FS = FixedSchedule(T)
# ais_fs = ais(prob, FS; N = N, compute_barriers = true, transition_kernel = CoordSliceSampler())
# β_linear = ais_fs.schedule
#
# nrounds = 20
# sais = SAIS(T, nrounds)
#
# ais_sais = ais(prob, sais; N = N)
# β_sais = ais_sais.schedule

##################### 
# testing online schedule selection
#####################
# MD = MirrorDescent(stepsize = 0.1, max_Δ = 0.5, max_T = Inf)
# ais_md = ais(prob, MD; N = N, transition_kernel = CoordSliceSampler())
# β_md = ais_md.schedule
#
# LS = LineSearch(divergence = 0.001, max_T = Inf)
# ais_ls = ais(prob, LS; N = N, transition_kernel = CoordSliceSampler())
#
# CRP = ConstantRateProgress(stepsize = 0.1, max_Δ = 0.5, max_T = Inf)
# ais_crp = ais(prob, CRP; N = N, transition_kernel = CoordSliceSampler())
# β_crp = ais_crp.schedule

#####################
# testing all methods
#####################

nptls = 2^12

T = 2048
nrounds = 3
S = SAIS(T, nrounds)

a_res = ais(prob, S; N = nptls, transition_kernel = CoordSliceSampler(), show_report = true)

# δs = [0.01, 0.1, 0.2]

# for δ in δs
#     MD = MirrorDescent(stepsize = δ, max_Δ = 0.5, max_T = Inf)
#     a_res = ais(prob, MD; N = nptls, show_report = true)
# end

# for δ in δs
#     CRP = ConstantRateProgress(stepsize = δ, max_Δ = 1, max_T = Inf)
#     a_res = ais(prob, CRP; N = nptls, show_report = true)
# end

# divs = [0.01, 0.1, 0.5]
# for div in divs
#     LS = LineSearch(divergence = div, max_T = Inf)
#     a_res = ais(prob, LS; N = nptls, show_report = true)
# end

