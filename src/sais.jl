@kwdef struct SAIS
    schedule::FixedSchedule
    n_rounds::Int
end
function SAIS(T::Int, nrounds::Int)
    return SAIS(FixedSchedule(T), nrounds)
end
tuning_rounds(sais::SAIS) = sais.n_rounds

function schedule_adaptation!(
    prob::AISProblem, sais::SAIS;
    show_report=true, kwargs...,
)
    kwargs = (save_trajectory=false, compute_barriers=true, show_report = false, kwargs...)
    println("Starting schedule adaptation with $(sais.n_rounds) rounds")
    for r in 1:(sais.n_rounds)
        a = ais(prob, sais.schedule; kwargs...)
        if show_report
            report(a, r == 1, r == sais.n_rounds)
        end
        if r == sais.n_rounds
            return a
        else
            schedule = a.schedule
            T = length(schedule)
            schedule = Pigeons.optimal_schedule(a.intensity, schedule, T)
            update_schedule!(sais.schedule, schedule)
        end
    end
end
ais(prob::AISProblem, sais::SAIS; kwargs...) = schedule_adaptation!(prob, sais; kwargs...)

# get intensity from the particles
function sais_optimal_schedule(intensity, current_schedule, ntemps)
    schedule = Pigeons.optimal_schedule(intensity, current_schedule, ntemps)
    return schedule
end

