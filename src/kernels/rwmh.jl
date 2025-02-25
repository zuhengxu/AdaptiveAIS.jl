struct RWMH_sweep <: TransitionKernel
    ϵs::AbstractVector{<:Real}
    sqrtΣ::Union{AbstractMatrix{<:Real}, UniformScaling{Bool}}
    n_passes::Int
end

RWMH_sweep() = RWMH_sweep([10^p for p in -5:0.1:1], I, 1)

function rwmh(rng, x, ℓ, ϵ, logπ::Function, sqrtΣ)
    dim = size(x, 1)
	xp = x + ϵ*(sqrtΣ*randn(dim))
	ℓp = logπ(xp)
    logα = min(ℓp - ℓ, 0)
	if log(rand(rng)) <= logα
		return xp, ℓp, logα, true
	else
        return x, ℓ, log1mexp(logα), false
	end
end

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

function step(rng, kernel::RWMH_sweep, logπ::Function, state)
    logA = 0
    ℓ = logπ(state)
    for _ in 1:kernel.n_passes
        for ϵ in kernel.ϵs
            state, ℓ, logα, _ = rwmh(rng, state, ℓ, ϵ, logπ, kernel.sqrtΣ)
            logA += logα
        end
    end        
    return state, logA
end
