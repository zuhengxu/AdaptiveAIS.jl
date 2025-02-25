using LogDensityProblems
using Distributions, Random
using DifferentiationInterface, Zygote, ForwardDiff
using Optimisers
using LogExpFunctions
using PDMats
using Statistics
using StatsBase
using StatsFuns
using JLD2
using Base.Threads: @threads
using LinearAlgebra
include("../logdensity_ad_wrap.jl")
include("../gaussian_reference.jl")

struct BrownianMotion{Y, Idx}
    y       :: Y
    obs_idx :: Idx
end

function LogDensityProblems.capabilities(::Type{<:BrownianMotion})
    return LogDensityProblems.LogDensityOrder{2}()
end
function LogDensityProblems.logdensity(prob::BrownianMotion, θ)
    (; y, obs_idx) = prob
    x     = @view(θ[1:30])
    α_inn = softplus(θ[31])
    α_obs = softplus(θ[32])

    ℓjac_α_inn = loglogistic(α_inn)
    ℓjac_α_obs = loglogistic(α_obs)

    ℓp_α_inn = logpdf(LogNormal(0, 2), α_inn)
    ℓp_α_obs = logpdf(LogNormal(0, 2), α_obs)
    ℓp_x1    = logpdf(Normal(0, α_inn), x[1])
    ℓp_x     = logpdf(MvNormal(@view(x[1:end-1]), α_inn), @view(x[2:end]))
    ℓp_y     = logpdf(MvNormal(@view(x[obs_idx]), α_obs), y)

    ℓp_y + ℓp_x1 + ℓp_x + ℓp_α_inn + ℓp_α_obs + ℓjac_α_inn + ℓjac_α_obs
end

LogDensityProblems.dimension(prob::BrownianMotion) = 32

function BrownianMotion()
    y = [
        0.21592641,
        0.118771404,
        -0.07945447,
        0.037677474,
        -0.27885845,
        -0.1484156,
        -0.3250906,
        -0.22957903,
        -0.44110894,
        -0.09830782,       
        #
        -0.8786016,
        -0.83736074,
        -0.7384849,
        -0.8939254,
        -0.7774566,
        -0.70238715,
        -0.87771565,
        -0.51853573,
        -0.6948214,
        -0.6202789,
    ]
    obs_idx = vcat(1:10, 21:30)
    BrownianMotion(y, obs_idx)
end

D = 32
p0 = GaussianReference(zeros(D))
p1 = ADHessian(BrownianMotion(), AutoForwardDiff(); x = randn(D))
L = DifferentiableAIS.LinearPath

prob = AISProblem(p0, p1, L)
adbackend = AutoZygote()




# T = 64
# ulas = [ULA(D, -6ones(D)) for _ in 1:T]
# S = FixedSchedule(T)
# BK2 = TimeCorrectedReverse()
# BK1 = DefaultReverse()
#
# ps = (
#     D = D,
#     damping = 0.1,
#     stepsize = 0.05*ones(D),
#     share_damping = false,
# )
# uhas = instantiate_kernels(UHA, ps, T)
#
#
# bs = 20
# A, ns = ais(prob, S, BK2, ulas, 1024; update_schedule = true)
# ula_t, _, _, _ = dais(
#     prob, S, BK2, ulas, bs, adbackend;
#     max_iters = 3000,
#     optimiser = Optimisers.Adam(1e-2),
# )
# ais(prob, S, BK2, ula_t, 256)
#
#
# ais(prob, S, BK1, uhas, 1024)
# uha_t, _, _, _ = dais(
#     prob, S, BK1, uhas, bs, adbackend;
#     max_iters = 3000,
#     optimiser = Optimisers.Adam(1e-2),
# )
# ais(prob, S, BK1, uha_t, 256)
#
#
#
# smc(prob, S, BK2, ulas, 512, 1024)
# ula_tt, _, _, _ = dsmc(
#     prob, S, BK2, ulas, 32, 64, adbackend;
#     max_iters = 5000,
#     optimiser = Optimisers.Adam(1e-2),
# )
# smc(prob, S, BK2, ula_tt, 128, 256)

# run_dais(1, UHA, DefaultReverse(); init_T = 16, max_T = 128, lr = 1e-4, batchsize = 64, max_iters = 3_000, share_damping = true)
