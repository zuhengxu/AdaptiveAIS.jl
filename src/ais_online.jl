function ais(
    prob::AISProblem{R, T, LinearPath},
    sched::OnlineScheduling;
    N::Int=2^5,
    seed=1,
    save_trajectory=false,
    transition_kernel=RWMH_sweep(),
    show_report=true,
) where {R, T}
    full_timing = @timed begin
        rngs = SplitRandomArray(N; seed)
        D = dimension(prob)

        # initialization: iid sampling from reference
        states = iid_sample_reference(rngs, prob, N)
        sample_trajectory = save_trajectory ? [] : nothing
        push_states!(sample_trajectory, states)

        # parallel propagation 
        log_weights = zeros(N)
        log_ratios = zeros(N)
        println("Starting AIS with $(string(typeof(sched))) with $(N) particles, and maximum $(sched.max_T) temperatures")
        
        schedule = Float64[0]
        cur_t = Ref(0) # keeping track of the number of annealings so far

        timing = @timed begin
            while schedule[end] < 1
                log_density_ratio!(prob, states, log_ratios)
                
                # find delta:= next_β - cur_β
                cur_beta = schedule[end]
                Δ = find_delta(
                    prob, sched, log_weights, log_ratios, cur_beta, cur_t[]
                )
                next_beta = cur_beta + Δ
                push!(schedule, next_beta)
                cur_t[] += 1

                # mutate and weigh particles
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
                push_states!(sample_trajectory, states)                 
            end
            nothing
        end

        # collect particles
        particles = Particles(states, log_weights)
        nothing
    end

    output = AIS_output(particles, sample_trajectory, timing, full_timing, schedule, nothing, nothing)
    report(output, show_report)
    return output
end


function mutate_and_weigh!(
    rngs,
    ::OnlineScheduling,
    prob::AISProblem{R,F,LinearPath},
    transition_kernel,
    next_beta,
    states,
    log_weights,
    log_ratios,
    Δβ, 
) where {R, F}
    mutate!(rngs, transition_kernel, prob, next_beta, states)
    log_weights .+=  Δβ.* log_ratios
end

push_states!(::Nothing, _) = nothing
push_states!(trajectory, states) = push!(trajectory, copy(states))


function ais(prob, sched::DebiasOnlineScheduling; kargs...)
    a = ais(prob, sched.OnlineSched; kargs...)
    β = a.schedule
    FS = FixedSchedule(β)
    return ais(prob, FS; kargs...)
end
