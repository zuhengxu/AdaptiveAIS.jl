using ChainRulesCore: ignore_derivatives

function mutate_and_weigh(
    rng,
    prob::AISProblem{R,F,LinearPath},
    transition_kernel::TransitionKernel,
    next_beta,
    states::AbstractMatrix,
    log_weights::AbstractVector,
    Δβ, 
) where {R, F}
    log_ratios = log_density_ratio(prob, states) 
    states, logAs = mutate(rng, transition_kernel, prob, next_beta, states)
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
    rng,
    prob::AISProblem,
    sched::FixedSchedule, 
    states::AbstractMatrix;
    N::Int=2^5,
    transition_kernel=RWMH_sweep(),
)
    log_weights = zeros(N)
    logAs = zeros(N)

    _, log_weights, logAs = propogate_and_weigh(
        rng, sched, prob, transition_kernel, states, log_weights, logAs
    )
    return log_weights, logAs, sched.schedule
end

function _compute_log_weights(
    rng,
    prob::AISProblem,
    sched::OnlineScheduling,
    states::AbstractMatrix;
    N::Int = 2^5,
    transition_kernel = RWMH_sweep(),
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
        Δ = find_delta(prob, sched, log_weights, log_ratios, cur_beta, cur_t[])
        next_beta = cur_beta + Δ

        ignore_derivatives() do
            push!(schedule, next_beta)
            cur_t[] += 1
        end

        # mutate and weigh particles
        states, ΔlogAs, log_weights = mutate_and_weigh(
            rng,
            prob,
            transition_kernel,
            next_beta,
            states,
            log_weights,
            Δ,
        )
        logAs = logAs .+ ΔlogAs
    end

    return log_weights, logAs, schedule
end

function _compute_log_weights(
    rng,
    prob::AISProblem,
    sched::DebiasOnlineScheduling, 
    states::AbstractMatrix;
    N::Int=2^5,
    transition_kernel=RWMH_sweep(),
)
    _, _, β = _compute_log_weights(
        rng, prob, sched.OnlineSched, states; N=N, transition_kernel=transition_kernel
    )

    # WARNING: okay to drop grad for this β generation?
    FS = FixedSchedule(ignore_derivatives(β))
    return _compute_log_weights(rng, prob, FS, states; N=N, transition_kernel=transition_kernel)
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
    rng::AbstractRNG,
    prob::AISProblem,
    GE::AbstractGradEst,
    sched::AbstractScheduler,
    N::Int,
    transition_kernel::TransitionKernel,
)
    states = iid_sample_reference(rng, prob, N)
    log_weights, log_As, _ = _compute_log_weights(rng, prob, sched, states; N=N, transition_kernel=transition_kernel)
    return -_elbo(GE, log_weights, log_As)
end

kl_objective(θ_flat, re, GE::AbstractGradEst, args...) = kl_objective(re(θ_flat), GE, args...)

# function dais(
#     rng::AbstractRNG,
#     prob::AISProblem,
#     args...;
#     max_iters::Int=1000,
#     optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
#     ADbackend,
#     kwargs...
# )
#     θ_flat, re = destructure(prob)
#     loss(θ, rng, args...) = kl_objective(rng, re(θ), args...)

#     # Normalizing flow training loop 
#     θ_flat_trained, opt_stats, st, time_elapsed = optimize(
#         ADbackend,
#         loss,
#         θ_flat,
#         re,
#         (rng, args...)...;
#         max_iters=max_iters,
#         optimiser=optimiser,
#         kwargs...,
#     )

#     prob_trained = re(θ_flat_trained)
#     return prob_trained, opt_stats, st, time_elapsed
# end

