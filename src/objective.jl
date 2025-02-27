using ChainRulesCore: ignore_derivatives

function mutate_and_weigh(
    rngs,
    prob::AISProblem{R,F,LinearPath},
    transition_kernel::TransitionKernel,
    next_beta,
    states::AbstractMatrix,
    log_weights::AbstractVector,
    Δβ, 
) where {R, F}
    log_ratios = log_density_ratio(prob, states)
    states, logAs = mutate(rngs, transition_kernel, prob, next_beta, states)
    return states, logAs, log_weights + Δβ.* log_ratios
end

function propogate_and_weigh(
    rng,
    sched::FixedSchedule,
    prob::AISProblem,
    transition_kernel::TransitionKernel,
    states,                 # D x N
    log_weights,
    logAs,  
)
    T = schedule_length(sched)
    βs = sched.schedule

    for t in 2:T
        cur_β, prev_β = βs[t], βs[t - 1]
        Δβ = cur_β - prev_β
        states, ΔlogAs, log_weights = mutate_and_weigh(
            rng, prob, transition_kernel, cur_β, states, log_weights, Δβ
        )
        logAs = logAs .+ ΔlogAs
    end
    return states, log_weights, logAs
end

function _compute_log_weights(
    rngs,
    prob::AISProblem,
    sched::FixedSchedule, 
    states::AbstractMatrix;
    N::Int=2^5,
    transition_kernel=RWMH_sweep(),
)
    log_weights = zeros(N)
    logAs = zeros(N)

    _, log_weights, logAs = propogate_and_weigh(
        rngs, sched, prob, transition_kernel, states, log_weights, logAs
    )
    return log_weights, logAs, sched.schedule
end

function _compute_log_weights(
    rngs,
    prob::AISProblem,
    sched::OnlineScheduling, 
    states::AbstractMatrix;
    N::Int=2^5,
    transition_kernel=RWMH_sweep(),
)
    log_weights = zeros(N)
    logAs = zeros(N)
    
    schedule = Float64[0]
    cur_t = Ref(0) # keeping track of the number of annealings so far

    while schedule[end] < 1
        log_ratios = log_density_ratio(prob, states)
        
        # find delta:= next_β - cur_β
        cur_beta = schedule[end]

        # WARNING: Do we want to drop grad for this function?
        Δ = find_delta(
            prob, sched, log_weights, log_ratios, cur_beta, cur_t[]
        )
        next_beta = cur_beta + Δ
        push!(schedule, next_beta)
        cur_t[] += 1

        # mutate and weigh particles
        states, ΔlogAs, log_weights = mutate_and_weigh(
            rngs, prob, transition_kernel, next_beta, states, log_weights, Δ
        )
        logAs = logAs .+ ΔlogAs
    end

    return log_weights, logAs, schedule
end

function _compute_log_weights(
    rngs,
    prob::AISProblem,
    sched::DebiasOnlineScheduling, 
    states::AbstractMatrix;
    N::Int=2^5,
    transition_kernel=RWMH_sweep(),
)
    _, _, β = _compute_log_weights(
        rngs, prob, sched.OnlineSched, states; N=N, transition_kernel=transition_kernel
    )

    # WARNING: okay to drop grad for this β generation?
    FS = FixedSchedule(ignore_derivatives(β))
    return _compute_log_weights(rngs, prob, FS, states; N=N, transition_kernel=transition_kernel)
end

# now we can define the objective function based on those computed quantities
_elbo(log_weights::T, log_As::T) where {T <: AbstractVector} = mean(log_weights)
function _elbo_mcvae(log_weights::T, log_As::T) where {T <: AbstractVector}
    W = mean(log_weights)
    return W + ignore_derivatives(W) * mean(log_As)
end
function _elbo_mcvae_cv(log_weights::T, log_As::T) where {T <: AbstractVector}
    N = size(log_weights, 1)
    SW = sum(log_weights)

    # eq(26) in Thin et.al. 21
    cv = log_weights .+ (log_weights .- SW)./(N-1)
    return mean(log_weights) + mean(ignore_derivatives(cv) .* log_As)
end
_elbo(::TwoPointZeroOrderSmooth, log_weights, log_As) = _elbo(log_weights, log_As)
_elbo(::CondBernoulli, log_weights, log_As) = _elbo_mcvae(log_weights, log_As)
_elbo(::CondBernoulliCV, log_weights, log_As) = _elbo_mcvae_cv(log_weights, log_As)

function kl_objective(
    GE::AbstractGradEst,
    rngs,
    prob::AISProblem,
    sched::AbstractScheduler; 
    N::Int=2^5,
    transition_kernel=RWMH_sweep(),
)
    states = iid_sample_reference(rngs, prob, N)
    log_weights, log_As, _ = _compute_log_weights(rngs, prob, sched, states; N=N, transition_kernel=transition_kernel)
    return -_elbo(GE, log_weights, log_As)
end



