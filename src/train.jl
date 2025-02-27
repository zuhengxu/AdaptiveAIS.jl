using ProgressMeter
using Optimisers
using LinearAlgebra
using ADTypes
import DifferentiationInterface as DI

#######################################################
# training loop for variational objectives 
#######################################################
function pm_next!(pm, stats::NamedTuple)
    return ProgressMeter.next!(pm; showvalues=map(tuple, keys(stats), values(stats)))
end

_wrap_in_DI_context(args) = map(DI.Constant, args)

function _prepare_gradient(loss, adbackend::ADTypes.AbstractADType, θ, args...)
    if isempty(args)
        return DI.prepare_gradient(loss, adbackend, θ)
    end
    return DI.prepare_gradient(loss, adbackend, θ, map(DI.Constant, args)...)
end
function _value_and_gradient(loss, prep, adbackend::ADTypes.AbstractADType, θ, args...)
    if isempty(args)
        return DI.value_and_gradient(loss, prep, adbackend, θ)
    end
    return DI.value_and_gradient(loss, prep, adbackend, θ, map(DI.Constant, args)...)
end

# TODO: find a better way for the interface
_prepare_gradient(loss, ::TwoPointZeroOrderSmooth, θ, args...) = nothing
_value_and_gradient(loss, ::Nothing, GE::TwoPointZeroOrderSmooth, θ, args...) =
    value_grad_and_update_state!(GE, loss, θ, args...)

function optimize(
    adbackend,
    loss,
    θ₀::AbstractVector{<:Real}, 
    reconstruct,
    args...;
    max_iters::Int=10000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    show_progress::Bool=true,
    callback=default_cb,
    hasconverged=(i, stats, re, θ, st) -> false,
    prog=ProgressMeter.Progress(
        max_iters; desc="Training", barlen=31, showspeed=true, enabled=show_progress
    ),
)
    time_elapsed = @elapsed begin 
        opt_stats = []

        # prepare loss and autograd
        θ = copy(θ₀)
        # grad = similar(θ)
        prep = _prepare_gradient(loss, adbackend, θ₀, args...)


        # initialise optimiser state
        st = Optimisers.setup(optimiser, θ)

        # general `hasconverged(...)` approach to allow early termination.
        converged = false
        i = 1
        while (i ≤ max_iters) && !converged
            ls, g = _value_and_gradient(loss, prep, adbackend, θ, args...)

            # Save stats
            stat = (iteration=i, loss=ls, gradient_norm=norm(g))

            # callback
            if callback !== nothing
                new_stat = callback(i, opt_stats, reconstruct, θ, args...)
                stat = new_stat !== nothing ? merge(stat, new_stat) : stat
            end
            push!(opt_stats, stat)

            # update optimiser state and parameters
            st, θ = Optimisers.update!(st, θ, g)

            # check convergence
            i += 1
            converged = hasconverged(i, stat, reconstruct, θ, st)
            pm_next!(prog, stat)
        end
    end
    # return status of the optimiser for potential continuation of training
    return θ, map(identity, opt_stats), st, time_elapsed
end


function dais(
    rng::AbstractRNG,
    prob::AISProblem,
    args...;
    adbackend::Union{ADTypes.AbstractADType, TwoPointZeroOrderSmooth}=AutoZygote(),
    max_iters::Int=1000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    kwargs...
)
    θ_flat, re = destructure(prob)
    loss(θ, rng, args...) = kl_objective(rng, re(θ), args...)

    # Normalizing flow training loop 
    θ_flat_trained, opt_stats, st, time_elapsed = optimize(
        adbackend,
        loss,
        θ_flat,
        re,
        (rng, args...)...;
        max_iters=max_iters,
        optimiser=optimiser,
        kwargs...,
    )

    prob_trained = re(θ_flat_trained)
    return prob_trained, opt_stats, st, time_elapsed
end

# a default Call back function for training evaluation 
function default_cb(i, stats, re, θ, rng, GE, sched, N, kernel)
    if i % 100 == 0
        a = ais(re(θ), sched; seed = i, N = N, transition_kernel = kernel, show_report = false)
        T = length(get_schedule(a))
        el = a.particles.elbo
        logz = a.particles.log_normalization
        return (elbo=el, logZ=logz, sched_length=T)
    else
        return (
            elbo=nothing,
            logz=nothing,
            sched_length=nothing,
        )
    end    
end

function process_logging(train_stats)
    return (
        iteration = [s.iteration for s in train_stats],
        logZs = [s.logZ for s in train_stats if !isnothing(s.logZ)],
        elbos = [s.elbo for s in train_stats if !isnothing(s.elbo)],
        Ts = [s.sched_length for s in train_stats if !isnothing(s.sched_length)],
    )
end
