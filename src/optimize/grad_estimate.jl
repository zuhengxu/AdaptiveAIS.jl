# using Random, Distributions, LinearAlgebra

abstract type AbstractGradEst end

function get_gradient end
function update_state! end
function init_state! end
function value_and_grad end

_ensure_stable(x) = isfinite(x) ? x : 0.0

"""
    grad_and_update_state!(rng, G::AbstractGradEst, args...)

get gradient estimate based on G and update the state of G
"""
function grad_and_update_state!(rng, G::AbstractGradEst, args...)
    gt = get_gradient(rng, G, args...)
    update_state!(G)
    return gt
end
function value_grad_and_update_state!(rng, G::AbstractGradEst, args...)
    fx, gt = value_and_grad(rng, G, args...)
    update_state!(G)
    return fx, gt
end


get_gradient(G::AbstractGradEst, args...) = get_gradient(Random.default_rng(), G, args...)
value_and_grad(G::AbstractGradEst, args...) =
    value_and_grad(Random.default_rng(), G, args...)

grad_and_update_state!(G::AbstractGradEst, args...) =
    grad_and_update_state!(Random.default_rng(), G, args...)

value_grad_and_update_state!(G::AbstractGradEst, args...) =
    value_grad_and_update_state!(Random.default_rng(), G, args...)


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

struct TwoPointZeroOrderSmooth{T<:Real} <: AbstractGradEst
    u::T
    t::Base.RefValue{Int} # mutable state carried to track the current iter 
end

TwoPointZeroOrderSmooth() = TwoPointZeroOrderSmooth(1.0, Ref(1))

init_state!(G::TwoPointZeroOrderSmooth, t) = G.t[] = t
update_state!(G::TwoPointZeroOrderSmooth) = G.t[] += 1
_get_ut(G::TwoPointZeroOrderSmooth) = G.u / G.t[]

function get_gradient(rng, G::TwoPointZeroOrderSmooth, f::Function, x, fargs...)
    # sample gaussian noise for smoothing
    Z = randn(rng, size(x))

    # perturb x with noise
    ut = _get_ut(G) # lets not update the state when just calling get_gradient
    x_pert = x .+ ut .* Z

    fx = f(x, fargs...)
    fx_pert = f(x_pert, fargs...) 
    gt = _ensure_stable((fx_pert - fx) / ut) .* Z
    return gt
end

function value_and_grad(rng, G::TwoPointZeroOrderSmooth, f::Function, x, fargs...)
    # sample gaussian noise for smoothing
    Z = randn(rng, size(x))

    # perturb x with noise
    ut = _get_ut(G) # lets not update the state when just calling get_gradient
    x_pert = x .+ ut .* Z

    fx = f(x, fargs...)
    fx_pert = f(x_pert, fargs...) 
    gt = _ensure_stable((fx_pert - fx) / ut) .* Z
    return fx, gt
end



struct CondBernoulli <: AbstractGradEst end

struct CondBernoulliCV <: AbstractGradEst end

struct StochasticAD <: AbstractGradEst end
