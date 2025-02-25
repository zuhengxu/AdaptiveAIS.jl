include("demo_utils.jl")

struct GaussianTarget{E<:Real} <: AbstractTarget
    dist::MvNormal{E}
end

function GaussianTarget(mu::AbstractVector{E}, σ::E = 0.2) where {E}
    GaussianTarget(MvNormal(mu, σ*one(E)))
end

function LogDensityProblems.capabilities(::GaussianTarget)
    LogDensityProblems.LogDensityOrder{1}()
end

LogDensityProblems.dimension(πT::GaussianTarget) = length(πT.dist)
function LogDensityProblems.logdensity(πT::GaussianTarget, x::AbstractVector)
    return logpdf(πT.dist, x)
end
function LogDensityProblems.logdensity_and_gradient(πT::GaussianTarget, x::AbstractVector)
    l = logpdf(πT.dist, x)
    ∇l = ForwardDiff.gradient(y -> logpdf(πT.dist, y), x)
    return l, ∇l
end


###############################
# testing slice sampler
##############################
# N = 1000
# s = zeros(2, N)
# rngs = SplitRandomArray(N)
# iid_sample!(rngs, p0, s)
#
# T = 20
# sched = FixedSchedule(T)
# L = LinearPath
# prob = AISProblem(p0, p1, L)
# iid_sample_reference!(rngs, prob, s)
#
#
# logpβ(β, s) = log_annealed_density(prob, β, s) 
# # current_temp(prob)
#
# nstep = 10
# d = dimension(prob)
# T = length(sched)
# state = zeros(d, nstep)
# buffer = copy(state)
#
# # checking slice sampler
# rng = SplittableRandom(1)
# h = CoordSliceSampler()
# state = zeros(2)
# n = 1000
# states = Vector{typeof(state)}(undef, n)
# cached_lp = -Inf
#
# for i in 1:n
#     replica = Replica(state, 1, rng, (;), 1)
#     step!(h, prob, 0.5, state)
#     states[i] = copy(state)
# end
# states

############################
# set up the problem
############################
dim = 10
N = 2000
T = 40
L = LinearPath
μ = 10
sigma = 0.2
p0 = GaussianReference(zeros(dim))
p1 = GaussianTarget(μ*ones(dim), sigma)
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
#
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
#
#####################
# testing checking bias
#####################
# _, _, diffs = check_bias(prob, MD; show_report = false);

δs = [0.001, 0.01, 0.1, 0.2, 0.5, 1]

for δ in δs
    MD = MirrorDescent(stepsize = δ, max_Δ = 0.1, max_T = Inf)
    check_bias(prob, MD; N = N);
end

# for δ in δs
#     CRP = ConstantRateProgress(stepsize = δ, max_Δ = 1, max_T = Inf)
#     check_bias(prob, CRP; N = N);
# end

divs = [0.001, 0.01, 0.1, 0.2, 0.5, 1]
for div in divs
    LS = LineSearch(divergence = div, max_T = Inf)
    check_bias(prob, LS; N = N);
end

