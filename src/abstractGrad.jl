using Random, Distributions, LinearAlgebra

abstract type AbstractGradScheme end


"""
    TwoPointZeroOrderSmooth(smoother::Distributions.MultivariateDistribution; repsilon::Float64=1e-6)

Two point zero order gradient estimation for a differentiable function. Section 2.1 of Duchi et al. (2014)
Convergence rate is O(1/sqrt(t)) where t is the number of iterations, with stepsize O(1/sqrt(t)), 
and smoothing bw ut = O(1/t).

# Parameters
- `dim::Int`: Dimension of the gradient
- `smoother::Distributions.MultivariateDistribution`: Smoother distribution that is used to find a random direction
- `u::Real`: a bandwidth parameter for the smoother
"""

struct TwoPointZeroOrderSmooth{D<:Int, Z<:Distributions.MultivariateDistribution, T<:Real} <: AbstractGradScheme
    dim::D
    smoother::Z
    u::T
end

TwoPointZeroOrderSmooth(dim) = TwoPointZeroOrderSmooth(dim, MvNormal(zeros(dim), I), 1.0)

_get_ut(G::TwoPointZeroOrderSmooth, t::Int) = G.u / t
_ensure_stable(x) = isfinite(x) ? x : 0.0

function get_gradient(rng, G::TwoPointZeroOrderSmooth, f::Function, x::AbstractVector, t::Int, fargs...)
    fx = f(x, fargs...)
    ut = _get_ut(G, t)
    Z = rand(rng, G.smoother)
    x_pert = x .+ ut .* Z
    fx_pert = f(x_pert, fargs...) 

    gt = _ensure_stable((fx_pert - fx) / ut) .* Z
    return gt
end
