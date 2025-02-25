using Roots

fisher_information(
    prob::AISProblem{R,T,LinearPath}, log_ratios::AbstractVector,
) where {R,T} = var(log_ratios)

@kwdef struct MirrorDescent <: OnlineScheduling
    """ODE discretization stepsize"""
    stepsize::Real

    """maximal jump in temperature"""
    max_Δ::Real

    """maximal number of annealings, can be Inf for no limit"""
    max_T::Real

    function MirrorDescent(stepsize::Real, max_Δ::Real, max_T::Real)
        if stepsize <= 0
            throw(ArgumentError("stepsize must be positive"))
        end
        if max_Δ <= 0 && max_Δ >= 1
            throw(ArgumentError("max_Δ must be in (0, 1)"))
        end
        if max_T <= 0
            throw(ArgumentError("max_T must be positive"))
        end
        return new(stepsize, max_Δ, max_T)
    end
end

@kwdef struct ConstantRateProgress <: OnlineScheduling
    """ODE discretization stepsize"""
    stepsize::Real

    """maximal jump in temperature"""
    max_Δ::Real

    """maximal number of annealings, can be Inf for no limit"""
    max_T::Real

    function ConstantRateProgress(stepsize::Real, max_Δ::Real, max_T::Real)
        if stepsize <= 0
            throw(ArgumentError("stepsize must be positive"))
        end
        if max_Δ <= 0 && max_Δ >= 1
            throw(ArgumentError("max_Δ must be in (0, 1)"))
        end
        if max_T <= 0
            throw(ArgumentError("max_T must be positive"))
        end
        return new(stepsize, max_Δ, max_T)
    end
end

"""Eq(16) in Chopin et al. 2024"""
function find_delta(
    prob::AISProblem{R,T,LinearPath},
    sched::MirrorDescent,
    log_weights,
    log_ratios::AbstractVector,
    cur_beta,
    ntrans_now::Int,
) where {R,T}
    mx = 1 - cur_beta
    if ntrans_now ≥ sched.max_T
        return mx
    end
    I_hat = fisher_information(prob, log_ratios)
    G = 1 / √(I_hat)
    Δ = sched.stepsize * G
    return min(Δ, mx, sched.max_Δ)
end

"""Eq(20) in Chopin et al. 2024"""
function find_delta(
    prob::AISProblem,
    sched::ConstantRateProgress,
    log_weights,
    log_ratios::AbstractVector,
    cur_beta,
    ntrans_now::Int,
)
    mx = 1 - cur_beta
    if ntrans_now ≥ sched.max_T || mx < 1e-6
        return mx
    end
    I_hat = fisher_information(prob, log_ratios)
    logmax = log1p(-cur_beta)
    G = exp(-logsumexp(log(I_hat), logmax))
    Δ = sched.stepsize * G
    return min(Δ, mx, sched.max_Δ)
end


@kwdef struct LineSearch <: OnlineScheduling
    """
    target log-CESS for each line search so that it remains constant; see Algorithm 4 in the paper

    formula at equi-divergence (see section 5.1 of sais paper) 
    #   Λ = T √div => T = Λ / √div
    """
    divergence::Real

    """maximal number of annealings, can be Inf for no limit"""
    max_T::Real
end

function find_delta(prob, sched::LineSearch, log_weights, log_ratios, cur_beta, ntrans_now::Int) 
    max_Δ = 1 - cur_beta
    divergence = sched.divergence
    f(x) = objective(log_weights, log_ratios, x) - divergence
    if ntrans_now ≥ sched.max_T || f(max_Δ) ≤ 0
        return max_Δ
    else
        Roots.find_zero(f, (0, max_Δ))
    end
end

"""line search to find next_temp - cur_temp based on conditional ESS (see Eq 3.16 in the paper)"""
objective(log_weights, log_ratios, delta) =
    compute_log_g(log_weights, log_ratios, delta, 2) - 2 * 
    compute_log_g(log_weights, log_ratios, delta, 1)

function compute_log_g(log_weights, log_ratios, delta, exponent::Int)
    log_sum = log_weights .+ (exponent * delta) .* log_ratios 
    return logsumexp(log_sum) .- logsumexp(log_weights)
end

