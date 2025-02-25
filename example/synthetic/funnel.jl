using Distributions, Random

struct Funnel{T<:Real} <: ContinuousMultivariateDistribution
    "Dimension of the distribution, must be >= 2"
    dim::Int
    "Mean of the first dimension"
    μ::T
    "Standard deviation of the first dimension, must be > 0"
    σ::T
    function Funnel{T}(dim::Int, μ::T, σ::T) where {T<:Real}
        dim >= 2 || error("dim must be >= 2")
        σ > 0 || error("σ must be > 0")
        return new{T}(dim, μ, σ)
    end
end
Funnel(dim::Int, μ::T, σ::T) where {T<:Real} = Funnel{T}(dim, μ, σ)
Funnel(dim::Int, σ::T) where {T<:Real} = Funnel{T}(dim, zero(T), σ)
Funnel(dim::Int) = Funnel(dim, 0.0, 9.0)

Base.length(p::Funnel) = p.dim
Base.eltype(p::Funnel{T}) where {T<:Real} = T

function Distributions._rand!(rng::AbstractRNG, p::Funnel, x::AbstractVecOrMat)
    T = eltype(x)
    d, μ, σ = p.dim, p.μ, p.σ
    d == size(x, 1) || error("Dimension mismatch")
    x[1, :] .= randn(rng, T, size(x, 2)) .* σ .+ μ
    x[2:end, :] .= randn(rng, T, d - 1, size(x, 2)) .* exp.(@view(x[1, :]) ./ 2)'
    return x
end

function Distributions._logpdf(p::Funnel{T}, x::AbstractVector) where {T}
    d, μ, σ = p.dim, p.μ, p.σ
    lpdf1 = logpdf(Normal(μ, σ), x[1])
    lpdfs = logpdf(MvNormal(zeros(T, d - 1), exp(x[1] / 2)), @view(x[2:end]))
    return lpdf1 + lpdfs 
end


function score(p::Funnel{T}, x::AbstractVector) where {T}
    d, μ, σ = p.dim, p.μ, p.σ
    x1 = x[1]
    x_2_d = x[2:end]
    a = expm1(-x1) + 1

    ∇lpdf1 = (μ - x1)/σ^2 - (d-1)/2 + a*sum(abs2, x_2_d)/2
    ∇lpdfs = -a*x_2_d
    return vcat(∇lpdf1, ∇lpdfs)
end


