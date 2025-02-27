struct RWMH_sweep <: TransitionKernel
    ϵs::AbstractVector{<:Real}
    sqrtΣ::Union{AbstractMatrix{<:Real}, UniformScaling{Bool}}
    n_passes::Int
end

RWMH_sweep() = RWMH_sweep([10^p for p in -5:0.1:1], I, 1)

function rwmh!(rng, x, ℓ, ϵ, logπ::Function, sqrtΣ)
    u_norm = randn(rng, size(x))
	xp = x + ϵ*(sqrtΣ*u_norm)
	ℓp = logπ(xp)
	if log(rand(rng)) <= ℓp - ℓ
        x .= xp
        return ℓp, true
    end
    return ℓ, false
end

function step!(rng, kernel::RWMH_sweep, logπ::Function, state)
    ℓ = logπ(state)
    for _ in 1:kernel.n_passes
        for ϵ in kernel.ϵs
            ℓ, _ = rwmh!(rng, state, ℓ, ϵ, logπ, kernel.sqrtΣ)
        end
    end
end



_mh_select(xp::T, x::T, acc::Bool) where T = acc * xp + (1 - acc) * x
_mh_logα(logα::T, acc::Bool) where T = acc * logα + (1 - acc) * log1mexp(logα)

function _accept_reject(rng, x::AbstractMatrix, ℓ::AbstractVector, xp::AbstractMatrix, ℓp::AbstractVector)
    logα = min.(ℓp .- ℓ, 0) # size N
    acc = (log.(rand(rng, size(logα))) .<= logα) # size N
    return _mh_select.(xp, x, acc), _mh_select.(ℓp, ℓ, acc), _mh_logα.(logα, acc), acc
end

function rwmh_batch(rng, xs::AbstractMatrix, ℓs::AbstractVector, ϵ, logπ::Function, sqrtΣ)
    u_norm = randn(rng, size(xs))
    xp = xs + ϵ * (sqrtΣ * u_norm)
    ℓp = logπ.(eachcol(xp))
    return _accept_reject(rng, xs, ℓs, xp, ℓp)
end


# function rwmh(rng, x::AbstractVector, ℓ, ϵ, logπ::Function, sqrtΣ)
#     u_norm = randn(rng, size(x))
# 	xp = x + ϵ*(sqrtΣ*u_norm)
# 	ℓp = logπ(xp)
#     logα = min(ℓp - ℓ, 0)
# 	if log(rand(rng)) <= logα
# 		return xp, ℓp, logα, true
# 	else
#         return x, ℓ, log1mexp(logα), false
# 	end
# end

# function step(rng, kernel::RWMH_sweep, logπ::Function, state::AbstractVector)
#     logA = 0
#     ℓ = logπ(state)
#     for _ in 1:kernel.n_passes
#         for ϵ in kernel.ϵs
#             state, ℓ, logα, _ = rwmh(rng, state, ℓ, ϵ, logπ, kernel.sqrtΣ)
#             logA += logα
#         end
#     end        
#     return state, logA
# end

function step(rng, kernel::RWMH_sweep, logπ::Function, states::AbstractMatrix)
    logAs = zeros(eltype(states), size(states, 2))
    ℓs = logπ.(eachcol(states))
    for _ in 1:kernel.n_passes
        for ϵ in kernel.ϵs
            states, ℓs, logαs, _ = rwmh_batch(rng, states, ℓs, ϵ, logπ, kernel.sqrtΣ)
            logAs = logAs .+ logαs
        end
    end
    return states, logAs
end

mutate(rng, kernel::RWMH_sweep, prob::AISProblem, next_beta, states::AbstractMatrix) =
    step(rng, kernel, prob, next_beta, states)
