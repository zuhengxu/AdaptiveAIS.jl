struct AISProblem{F,T,A<:AbstractPath}
    reference::F
    target::T
    path::A
    function AISProblem(reference::F, target::T, path::A) where {F,T,A}
        if LogDensityProblems.dimension(reference) != LogDensityProblems.dimension(target)
            throw(DimensionMismatch("Dimension mismatch between reference and target"))
        end
        new{F,T,A}(reference, target, path)
    end
end
# adding this so that reference can be tunable
@functor AISProblem (reference, )

dimension(prob::AISProblem) = LogDensityProblems.dimension(prob.target)

# function to be dispatched for reference distribution
function iid_sample end

# generate N iid samples from the reference distribution, output shape: D x N
iid_sample_reference(rng, prob::AISProblem, nsample) = iid_sample(rng, prob.reference, nsample)

log_density_reference(prob::AISProblem, x) = LogDensityProblems.logdensity(prob.reference, x)
log_density_target(prob::AISProblem, x) = LogDensityProblems.logdensity(prob.target, x)

# ensure numerical stability, mapping infiniteness to zero
ensure_finite(x::Real) = isfinite(x) ? x : zero(x)
ensure_finite(x::Real, v::AbstractVecOrMat) = isfinite(x) ? v : zeros(size(v))

function log_density_ratio(prob::AISProblem, x::AbstractVector)
    ℓπ0 = log_density_reference(prob, x)
    ℓπT = log_density_target(prob, x)
    return ensure_finite(ℓπT - ℓπ0)
end

function log_density_ratio(prob::AISProblem, xs::AbstractMatrix)
    ℓπ0 = log_density_reference(prob, xs)
    ℓπT = log_density_target(prob, xs)
    return ensure_finite.(ℓπT .- ℓπ0)
end

function log_density_ratio!(prob::AISProblem, states, output)
    N = size(states, 2)
    @threads for i in 1:N
        state = @view(states[:, i])
        output[i] = log_density_ratio(prob, state)
    end
end

function log_annealed_density(prob::AISProblem, beta, x::AbstractVector)
    @unpack reference, target, path = prob
    ℓπ0 = LogDensityProblems.logdensity(reference, x) 
    ℓπT = LogDensityProblems.logdensity(target, x)
    ℓπt = anneal(path, beta, ℓπ0, ℓπT)
    return ensure_finite(ℓπt)
end
log_annealed_density(prob::AISProblem, beta, xs::AbstractMatrix) = map(x -> log_annealed_density(prob, beta, x), eachcol(xs))


# customized adjoint for log_annealed_density
function ChainRulesCore.rrule(::typeof(log_annealed_density), prob::AISProblem, beta::Real, state::AbstractVector)
    y = log_annealed_density(prob, beta, state)
    function log_annealed_density_pullback(Δy)
        prob⁻ = NoTangent()
        beta⁻ = Δy * log_density_ratio(prob, state)
        state⁻ = log_annealed_gradient(prob, beta, state) .* Δy 
        return NoTangent(), prob⁻, beta⁻, state⁻
    end
    return y, log_annealed_density_pullback
end


function log_annealed_gradient(prob::AISProblem, beta, x::AbstractVector)
    @unpack reference, target, path = prob
    ℓπ0, ∇ℓπ0 = LogDensityProblems.logdensity_and_gradient(reference, x)
    ℓπT, ∇ℓπT = LogDensityProblems.logdensity_and_gradient(target, x)
    ℓπt = anneal(path, beta, ℓπ0, ℓπT)
    ∇ℓπt = anneal(path, beta, ∇ℓπ0, ∇ℓπT)
    return ensure_finite(ℓπt, ∇ℓπt)
end
log_annealed_gradient(prob::AISProblem, beta, xs::AbstractMatrix) = mapreduce(x -> log_annealed_gradient(prob, beta, x), hcat, eachcol(xs))

function log_annealed_density_and_gradient(prob::AISProblem, beta, x::AbstractVector)
    @unpack reference, target, path = prob
    ℓπ0, ∇ℓπ0 = LogDensityProblems.logdensity_and_gradient(reference, x)
    ℓπT, ∇ℓπT = LogDensityProblems.logdensity_and_gradient(target, x)
    ℓπt = anneal(path, beta, ℓπ0, ℓπT)
    ∇ℓπt = anneal(path, beta, ∇ℓπ0, ∇ℓπT)
    return ensure_finite(ℓπt), ensure_finite(ℓπt, ∇ℓπt)
end

#######################
# output struct
######################
@auto struct AIS_output 
    particles 
    trajectory
    timing # for the kernel only
    full_timing # for the full function call
    schedule 
    intensity
    barriers 
end

Base.show(io::IO, a::AIS_output) = begin
    if isnothing(a.barriers)
        print(io, "AIS(T=$(length(a.schedule)), N=$(n_particles(a.particles)), time=$(a.timing.time)s, ess=$(ess(a.particles)), lognorm=$(a.particles.log_normalization), elbo=$(a.particles.elbo))")
    else
        print(io, "AIS(T=$(length(a.schedule)), N=$(n_particles(a.particles)), global barriers=$(a.barriers.globalbarrier), time=$(a.timing.time)s, ess=$(ess(a.particles)), lognorm=$(a.particles.log_normalization), elbo=$(a.particles.elbo))")
    end
end

get_schedule(a::AIS_output) = a.schedule
