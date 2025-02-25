using LogDensityProblems
using LogExpFunctions
using DifferentiableAIS
using Distributions, Random
using DifferentiationInterface, Zygote, ForwardDiff
using Optimisers
using JLD2
using Base.Threads: @threads
using LinearAlgebra
include("../synthetic/funnel.jl")

### wrap gaussian reference into logdensityproblems interface
struct GaussianReference{E<:Real} 
    dist::MvNormal{E}
end

function GaussianReference(mu::AbstractVector{E}) where {E}
    GaussianReference(MvNormal(mu, one(E)))
end

# need to extend this function
DifferentiableAIS.iid_sample(π0::GaussianReference, N) = rand(π0.dist, N)
DifferentiableAIS.iid_sample(π0::GaussianReference) = rand(π0.dist)

function LogDensityProblems.capabilities(::GaussianReference)
    LogDensityProblems.LogDensityOrder{2}()
end

LogDensityProblems.dimension(π0::GaussianReference) = length(π0.dist)

function LogDensityProblems.logdensity(π0::GaussianReference, x::AbstractVector)
    return logpdf(π0.dist, x)
end

function LogDensityProblems.logdensity_and_gradient(
    π0::GaussianReference, x::AbstractVector
)
    l = logpdf(π0.dist, x)
    ∇l = π0.dist.μ .- x
    return l, ∇l
end

function LogDensityProblems.logdensity_gradient_and_hessian(
    π0::GaussianReference, x::AbstractVector
)
    l = logpdf(π0.dist, x)
    g = π0.dist.μ .- x
    H = -I
    return l, g, H
end

struct FunnelTarget{G<:Real} 
    dist::Funnel{G}
end
FunnelTarget(dim::Int) = FunnelTarget(Funnel(dim, 3.0))

function LogDensityProblems.capabilities(::FunnelTarget)
    LogDensityProblems.LogDensityOrder{2}()
end

LogDensityProblems.dimension(πT::FunnelTarget) = length(πT.dist)

function LogDensityProblems.logdensity(πT::FunnelTarget, x::AbstractVector)
    return logpdf(πT.dist, x)
end

function LogDensityProblems.logdensity_and_gradient(
    πT::FunnelTarget, x::AbstractVector
)
    l = logpdf(πT.dist, x)
    g = score(πT.dist, x)
    return l, g
end

function LogDensityProblems.logdensity_gradient_and_hessian(
    πT::FunnelTarget, x::AbstractVector
)
    l = logpdf(πT.dist, x)
    g = score(πT.dist, x)
    H = ForwardDiff.hessian(y -> logpdf(πT.dist, y), x)
    return l, g, H
end

Zygote.refresh()


D  = 10
p0 = GaussianReference(zeros(D))
p1 = FunnelTarget(D)
L = DifferentiableAIS.LinearPath
prob = AISProblem(p0, p1, L)

