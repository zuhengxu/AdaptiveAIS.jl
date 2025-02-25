using LogDensityProblems
using Bijectors, Distributions
using AdaptiveAIS

function LogDensityProblems.capabilities(::Bijectors.MultivariateTransformed)
    LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(πT::Bijectors.MultivariateTransformed) = length(πT)

function LogDensityProblems.logdensity(πT::Bijectors.MultivariateTransformed, x)
    return logpdf(πT, x)
end

AdaptiveAIS.iid_sample(rngs, p0::Bijectors.MultivariateTransformed, N) = rand(p0, N)
AdaptiveAIS.iid_sample(rng, p0::Bijectors.MultivariateTransformed) = rand(p0)
