# schedule setter
abstract type AbstractScheduler end

struct FixedSchedule{G<:Real} <: AbstractScheduler
    """ temperature schedule """
    schedule::AbstractVector{G}
end

function FixedSchedule(n::Int)
    temp = collect(range(0, 1; length=n))
    return FixedSchedule(temp)
end
Base.length(S::FixedSchedule) = length(S.schedule)

# the new schedule has to be of the same length as the old one
update_schedule!(S::FixedSchedule, new_sched) = (S.schedule .= new_sched)
schedule_length(S::FixedSchedule) = length(S.schedule)

abstract type OnlineScheduling <: AbstractScheduler end

struct DebiasOnlineScheduling <: AbstractScheduler
    OnlineSched::OnlineScheduling
end


abstract type StochasticScheduling <: AbstractScheduler end
