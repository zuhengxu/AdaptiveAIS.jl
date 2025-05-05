using ProgressMeter

####################
## general AIS for FixedSchedule
####################
function ais(
    prob::AISProblem,
    sched::FixedSchedule;
    N::Int=2^5,
    seed=1,
    save_trajectory=false,
    compute_barriers=false,
    transition_kernel=RWMH_sweep(),
    show_report = true, 
)
    full_timing = @timed begin
        T = schedule_length(sched)
        rngs = SplitRandomArray(N; seed)

        # rngs = [SplittableRandom(seed + i) for i in 1:N]
        D = dimension(prob)

        # initialization: iid sampling from reference
        states = iid_sample_reference(rngs, prob, N)
        
        sample_trajectory = save_trajectory ? zeros(D, T, N) : nothing
        if sample_trajectory !== nothing
            @threads for n in 1:N
                state = @view(states[:, n])
                @view(sample_trajectory[:, 1, n]) .= copy(state)
            end
        end


        # parallel propagation 
        log_weights = zeros(N)
        log_increments = zeros(T, N)

        timing = @timed begin
            propogate_and_weigh!(
                rngs,
                sched,
                prob,
                transition_kernel,
                states,
                sample_trajectory,
                log_weights,
                log_increments,
            )
            nothing
        end

        # collect particles
        particles = Particles(states, log_weights)
        intensity_vector = intensity(log_increments)
        schedule = sched.schedule
        barriers = if compute_barriers
            Pigeons.communication_barriers(intensity_vector, schedule)
        else
            nothing
        end
        nothing
    end

    output = AIS_output(particles, sample_trajectory, timing, full_timing, schedule, intensity_vector, barriers)
    report(output, show_report)
    return output
end

"""
    propogate_and_weigh!(
        rngs,
        sched::FixedSchedule,
        prob::AISProblem,
        transition_kernel,
        states,         # D x N
        sample_trajectory, # D x T x N
        log_weights,    # N 
        log_increments, # T x N or nothing
    )

Particle transitions and weight update for AIS with fixed schedule.
"""
function propogate_and_weigh!(
    rngs,
    sched::FixedSchedule,
    prob::AISProblem,
    transition_kernel,
    states,         # D x N
    sample_trajectory, # D x T x N
    log_weights,    # N 
    log_increments, # T x N or nothing
)
    D, N = size(states)

    # println("D: $(D), N: $(N)")

    T = schedule_length(sched)
    if sample_trajectory !== nothing
        @assert size(sample_trajectory) == (D, T, N)
    end

    βs = sched.schedule

    @threads for n in 1:N
        rng = rngs[n]
        for t in 2:T
            cur_β, prev_β = βs[t], βs[t - 1]
            state = @view(states[:, n])

            if sample_trajectory !== nothing
                @view(sample_trajectory[:, t, n]) .= copy(state)
            end

            log_increment = log_incremental_weight(prob, state, cur_β, prev_β)
            update_log_increment!(log_increments, t, n, log_increment)
            log_weights[n] += log_increment

            step!(rng, transition_kernel, prob, cur_β, state)
        end
    end
end

# define the log incremental weight for linear path
# need to dispatch for different path
function log_incremental_weight(
    prob::AISProblem{R, T,LinearPath},
    x::AbstractVecOrMat{E},
    cur_beta::E,
    prev_beta::E,
) where {R,T,E<:Real}
    @assert cur_beta > prev_beta
    Δβ = cur_beta - prev_beta
    ldr = log_density_ratio(prob, x)
    return Δβ * ldr
end

# Only record those when adaptation needs it
function update_log_increment!(log_increments, t, n, log_increment)
    return log_increments[t, n] = log_increment
end
update_log_increment!(::Nothing, _, _, _) = nothing
