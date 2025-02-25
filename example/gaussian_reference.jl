using LogDensityProblems
using AdaptiveAIS
using Distributions, Random
using LinearAlgebra

### wrap gaussian reference into logdensityproblems interface
struct GaussianReference{E<:Real} 
    dist::MvNormal{E}
end

function GaussianReference(mu::AbstractVector{E}) where {E}
    GaussianReference(MvNormal(mu, one(E)))
end

# need to extend this function
iid_sample(rng, π0::GaussianReference, N) = rand(π0.dist, N)
iid_sample(rng, π0::GaussianReference) = rand(π0.dist)

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

