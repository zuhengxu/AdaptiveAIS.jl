function elbo(
    rngs,
    prob::AISProblem,
    sched::FixedSchedule;
    N::Int=2^5,
    transition_kernel=RWMH_sweep(),
)
    T = schedule_length(sched)

    # initialization: iid sampling from reference
    states = iid_sample_reference(rngs, prob, N)

    # parallel propagation 
    log_weights = zeros(N)
    log_increments = zeros(T, N)

    propogate_and_weigh!(
        rngs,
        sched,
        prob,
        transition_kernel,
        states,
        nothing,
        log_weights,
        log_increments,
    )

    return mean(log_weights), sched.schedule
end

function elbo(
    rngs,
    prob::AISProblem,
    sched::OnlineScheduling;
    N::Int=2^5,
    transition_kernel=RWMH_sweep(),
)
    # initialization: iid sampling from reference
    states = iid_sample_reference(rngs, prob, N)
    # sample_trajectory = save_trajectory ? [] : nothing
    # push_states!(sample_trajectory, states)

    # parallel propagation
    log_weights = zeros(N)
    log_ratios = zeros(N)

    schedule = Float64[0]
    cur_t = Ref(0)

    while schedule[end] < 1
        log_density_ratio!(prob, states, log_ratios)

        # find delta:= next_β - cur_β
        cur_beta = schedule[end]
        Δ = find_delta(prob, sched, log_weights, log_ratios, cur_beta, cur_t[])
        next_beta = cur_beta + Δ
        push!(schedule, next_beta)
        cur_t[] += 1

        # propagate and weigh particles
        mutate_and_weigh!(
            rngs,
            sched,
            prob,
            transition_kernel,
            next_beta,
            states,
            log_weights,
            log_ratios,
            Δ,
        )
    end

    return mean(log_weights), schedule
end

function elbo(
    rngs,
    prob::AISProblem,
    sched::DebiasOnlineScheduling;
    N::Int=2^5,
    transition_kernel=RWMH_sweep(),
)
    _, β = elbo(rngs, prob, sched.OnlineSched; N = N, transition_kernel = transition_kernel)
    FS = FixedSchedule(β)
    return elbo(rngs, prob, FS; N = N, transition_kernel = transition_kernel), β
end

elbo(rngs, θ_flat, re, sched::AbstractScheduler; kargs...) =
    elbo(rngs, re(θ_flat), sched; kargs...)

elbo(θ_flat, re, sched::AbstractScheduler; kargs...) =
    elbo(Random.default_rng(), θ_flat, re, sched; kargs...)

