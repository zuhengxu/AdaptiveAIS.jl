using Base.Iterators
using DelimitedFiles
using Distributions
using FillArrays
using LogDensityProblems
using LogExpFunctions
using PDMats
using Statistics
using StatsFuns
using Tullio
using Zygote
include("../gaussian_reference.jl")
include("../logdensity_ad_wrap.jl")

neg_sigmoid(x) = -1.0/(1.0 + exp(-x))

struct LogisticRegression{XT, YT}
    X::XT
    y::YT
end

function LogDensityProblems.logdensity(prob::LogisticRegression, θ)
    (; X, y) = prob

    ℓp_θ   = mapreduce(normlogpdf, +, θ)

    logits = X*θ
    ℓp_y   = sum(@. logpdf(BernoulliLogit(logits), y))

    ℓp_θ + ℓp_y
end
Zygote.@adjoint function LogDensityProblems.logdensity(prob::LogisticRegression, θ)
      lr_logpdf_pullback(x̄) = (nothing, _mygrad(prob, θ) * x̄)
    return LogDensityProblems.logdensity(prob, θ), lr_logpdf_pullback
end


function LogDensityProblems.logdensity_and_gradient(prob::LogisticRegression, θ)
    (; X, y) = prob
    D = size(X, 2)
    logits = X*θ

    ℓp_θ   = mapreduce(normlogpdf, +, θ)
    ℓp_y   = sum(@. logpdf(BernoulliLogit(logits), y))
    ℓ =  ℓp_θ + ℓp_y

    p = neg_sigmoid.(logits)
    @tullio M[j] := X[n,j]*(p[n] + y[n])
    ∇ℓ = -θ .+ M
    return ℓ, ∇ℓ
end

function LogDensityProblems.logdensity_gradient_and_hessian(prob::LogisticRegression, θ)
    (; X, y) = prob
    D = size(X, 2)
    logits = X*θ

    ℓp_θ   = mapreduce(normlogpdf, +, θ)
    ℓp_y   = sum(@. logpdf(BernoulliLogit(logits), y))
    ℓ =  ℓp_θ + ℓp_y

    p = neg_sigmoid.(logits)
    @tullio M[j] := X[n,j]*(p[n] + y[n])
    ∇ℓ = -θ .+ M
    
    H = ForwardDiff.jacobian(x -> _mygrad(prob, x), θ)
    return ℓ, ∇ℓ, H
end


function _mygrad(prob::LogisticRegression, θ)
    (; X, y) = prob
    D = size(X, 2)

    logits = X*θ
    p = neg_sigmoid.(logits)
    @tullio M[j] := X[n,j]*(p[n] + y[n])
    return -θ .+ M
end

function LogDensityProblems.capabilities(::Type{<:LogisticRegression})
    return LogDensityProblems.LogDensityOrder{2}()
end

LogDensityProblems.dimension(prob::LogisticRegression) = size(prob.X, 2)

function preprocess_features(X::AbstractMatrix)
    μ = mean(X, dims=1)
    σ = std(X, dims=1)
    σ[σ .<= eps(Float64)] .= 1.0
    X = (X .- μ) ./ σ
    hcat(X, ones(size(X, 1), 1))
end

function LogisticRegressionSonar()
    data   = readdlm("sonar.csv", ',', Any, '\n')
    X      = convert(Matrix{Float64}, data[:, 1:end-1])
    y      = data[:, end] .== "R"
    X_proc = preprocess_features(X)
    LogisticRegression(X_proc, y)
end

Zygote.refresh()

p1 = LogisticRegressionSonar()
D = LogDensityProblems.dimension(p1)

p0 = GaussianReference(zeros(D))
# p1 = ADHessian(pt, AutoZygote(); x = randn(D))


L = DifferentiableAIS.LinearPath
prob = AISProblem(p0, p1, L)
adbackend = AutoZygote()








#
#
#
#
# T = 16
# ulas = [ULA(D, -4ones(D)) for _ in 1:T]
# S = FixedSchedule(T)
# BK2 = TimeCorrectedReverse()
# BK1 = DefaultReverse()
#
# ps = (
#     D = D,
#     damping = 0.1,
#     stepsize = 0.1*ones(D),
#     share_damping = false,
# )
# uhas = instantiate_kernels(UHA, ps, T)
#
#
# bs = 20
# A, ns = ais(prob, S, BK2, ulas, 256; update_schedule = true)
# ula_t, _, _, _ = dais(
#     prob, S, BK2, ulas, bs, adbackend;
#     max_iters = 3000,
#     optimiser = Optimisers.Adam(1e-2),
# )
# ais(prob, S, BK2, ula_t, 256)
#
#
# smc(prob, S, BK2, ulas, 512, 1024)
# ula_tt, _, _, _ = dsmc(
#     prob, S, BK2, ulas, 32, 64, AutoForwardDiff();
#     max_iters = 5000,
#     optimiser = Optimisers.Adam(1e-2),
# )
# smc(prob, S, BK2, ula_tt, 128, 256)
#
# ais(prob, S, BK1, uhas, 256)
# uha_t, _, _, _ = dais(
#     prob, S, BK1, uhas, bs, adbackend;
#     max_iters = 3000,
#     optimiser = Optimisers.Adam(1e-2),
# )
# ais(prob, S, BK1, uha_t, 256)
#
# smc(prob, S, BK1, uhas, 128, 256)
# uhatt, _, _, _ = dsmc(
#     prob, S, BK1, uhas, 32, 64, adbackend;
#     max_iters = 5000,
#     optimiser = Optimisers.Adam(1e-2),
# )
# smc(prob, S, BK1, uhatt, 128, 256)
#
