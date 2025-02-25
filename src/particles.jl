# code taken from https://github.com/alexandrebouchard/sais-gpu/blob/main/Particles.jl

@auto struct Particles
    states
    log_weights
    probabilities
    log_normalization
    elbo
end

ess(p::Particles) = 1 / sum(x -> x^2, p.probabilities)
n_particles(p) = length(p.probabilities)

ess_from_logweights(logw::AbstractVector) = begin
    logw_norm = logw .- LogExpFunctions.logsumexp(logw)
    exp(-LogExpFunctions.logsumexp(2 * logw_norm))
end

function Particles(particles::AbstractArray, log_weights::AbstractVector{E}) where {E}
    logws = copy(log_weights)
    elbo = mean(log_weights)
    log_normalization = exp_normalize!(log_weights) - log(E(length(log_weights)))
    return Particles(particles, logws, log_weights, log_normalization, elbo)
end

integrate(f::Function, p::Particles) =
    sum(1:n_particles(p)) do i 
        state = @view p.states[:, i] 
        f(state) * p.probabilities[i]
    end

∫ = integrate 

Statistics.mean(μ::Particles) = ∫(x -> x, μ) 
function Statistics.var(μ::Particles) 
    m = Statistics.mean(μ)
    return ∫(x -> (x - m).^2, μ)
end
Statistics.std(μ::Particles) = sqrt.(var(μ))
