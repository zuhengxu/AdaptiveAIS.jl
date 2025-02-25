"""
Normalize a vector of non-negative weights stored in 
log-scale. Returns the log normalization. 
Works on both CPU and GPU. 

I.e., given input `log_weights` (an array), 
perform in-place, numerically stable implementation of: 
```
prs = exp.(log_weights)
log_weights .= prs/sum(prs)
```

Moreover, returns a numerically stable version of 
```
log(sum(prs))
```
"""
# code taken from https://github.com/alexandrebouchard/sais-gpu/blob/main/utils.jl

function exp_normalize!(log_weights)
    m = maximum(log_weights)
    log_weights .= exp.(log_weights .- m) 
    return m + log(normalize!(log_weights))
end 

function normalize!(weights) 
    s = sum(weights)
    weights .= weights ./ s 
    return s
end


function log_normalize_weights(log_weights::AbstractVector)
    ℓW = log_weights .- LogExpFunctions.logsumexp(log_weights)
    return ℓW
end

exp_normalize(log_weights::AbstractVector) = exp.(log_normalize_weights(log_weights))

"""
compute sum_i exp(lw_i) fi
"""
function expectation_from_logweights(ℓw_norm::AbstractVector, fs::AbstractVector)
    N = length(ℓw_norm)
    return dot(exp.(ℓw_norm), fs)    
end
function expectation_from_weights(w_norm::AbstractVector, fs::AbstractVector)
    N = length(w_norm)
    return dot(w_norm, fs)    
end


# vectorized log_sum_exp
function log_sum_exp(log_weights; dims...) 
    m = maximum(log_weights; dims...)
    m .= m .+ log.(sum(exp.(log_weights .- m); dims...))
    return m
end


# tracking computation
mutable struct TrackedLogDensityProblem{Prob} 
    n_density_evals  :: Int
    n_gradient_evals :: Int
    n_hessian_evals  :: Int
    prob             :: Prob
end

function TrackedLogDensityProblem(prob)
    TrackedLogDensityProblem{typeof(prob)}(0, 0, 0, prob)
end

function LogDensityProblems.capabilities(::Type{TrackedLogDensityProblem{Prob}}) where {Prob}
    return LogDensityProblems.capabilities(Prob)
end

LogDensityProblems.dimension(prob::TrackedLogDensityProblem) = LogDensityProblems.dimension(prob.prob)

function LogDensityProblems.logdensity(prob::TrackedLogDensityProblem, x)
    prob.n_density_evals += 1
    return LogDensityProblems.logdensity(prob.prob, x)
end

function LogDensityProblems.logdensity_and_gradient(prob::TrackedLogDensityProblem, x)
    prob.n_gradient_evals += 1
    return LogDensityProblems.logdensity_and_gradient(prob.prob, x)
end

# function LogDensityProblems.logdensity_gradient_and_hessian(prob::TrackedLogDensityProblem, x)
#     prob.n_hessian_evals += 1
#     return LogDensityProblems.logdensity_gradient_and_hessian(prob.prob, x)
# end


# check if the AIS problem is tracked
function is_tracked(prob::AISProblem)
    return isa(prob.target, TrackedLogDensityProblem)
end

function TrackAISProblem(prob::AISProblem)
    if is_tracked(prob)
        println("The problem is already tracked")
        return prob
    else
        return AISProblem(prob.reference, TrackedLogDensityProblem(prob.target), prob.path)
    end
end

function number_of_evals(prob::AISProblem)
    _is_tracked = is_tracked(prob)
    if !is_tracked(prob)
        println("The problem is not tracked")
    end
    return (
        n_density_evals = _is_tracked ? prob.target.n_density_evals : 0,
        n_gradient_evals = _is_tracked ? prob.target.n_gradient_evals : 0,
        n_hessian_evals = _is_tracked ? prob.target.n_hessian_evals : 0,
    )
end

mutable struct TrainLog{T}
    logZs :: Vector{T}
    n_density_evals :: Vector{Int}
    n_gradient_evals :: Vector{Int}
    n_hessian_evals :: Vector{Int}
end
TrainLog() = TrainLog(Vector{Float64}(), Vector{Int}(), Vector{Int}(), Vector{Int}())

function _process_logging(train_stats)
    return (
        logZs = [s.logZ for s in train_stats if !isnothing(s.logZ)],
        n_density_evals = [
            s.n_density_evals for s in train_stats if !isnothing(s.n_density_evals)
        ],
        n_gradient_evals = [
            s.n_gradient_evals for s in train_stats if !isnothing(s.n_gradient_evals)
        ],
        n_hessian_evals = [
            s.n_hessian_evals for s in train_stats if !isnothing(s.n_hessian_evals)
        ],
    )
end


function _update_train_log!(train_log::Union{TrainLog,Nothing}, raw_stats)
    if !isnothing(train_log)
        stats = _process_logging(raw_stats)
        train_log.logZs = vcat(train_log.logZs, stats.logZs)
        train_log.n_density_evals = vcat(train_log.n_density_evals, stats.n_density_evals)
        train_log.n_gradient_evals = vcat(train_log.n_gradient_evals, stats.n_gradient_evals)
        train_log.n_hessian_evals = vcat(train_log.n_hessian_evals, stats.n_hessian_evals)
    end
end

