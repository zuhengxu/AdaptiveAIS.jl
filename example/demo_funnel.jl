include("demo_utils.jl")
include("synthetic/funnel.jl")
using AdaptiveAIS

struct FunnelTarget{G<:Real} 
    dist::Funnel{G}
end
FunnelTarget(dim::Int) = FunnelTarget(Funnel(dim))

function LogDensityProblems.capabilities(::FunnelTarget)
    LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(πT::FunnelTarget) = length(πT.dist)
function LogDensityProblems.logdensity(πT::FunnelTarget, x::AbstractVector)
    return logpdf(πT.dist, x)
end

#####################
# setup ais problem
#####################
D = 2
N = 2000
T = 40
L = LinearPath()

p0 = GaussianReference(zeros(D))
p1 = FunnelTarget(D)
prob = AISProblem(p0, p1, L)

# rng = Random.default_rng()
# xs= iid_sample(rng, p0, N)
# lrs = log_density_ratio(prob, xs)

# test run
MD = MirrorDescent(stepsize = 0.01, max_Δ = 0.1, max_T = Inf)
a = ais(prob, MD; N = N, save_trajectory = false, show_report = true, transition_kernel = RWMH_sweep())

LS = LineSearch(divergence = 0.01, max_T = Inf)
a = ais(prob, LS; N = N, save_trajectory = false, show_report = true, transition_kernel = RWMH_sweep())


#####################
# testing checking bias
#####################
δs = [0.001, 0.01, 0.1, 0.2, 0.5, 1]

for δ in δs
    MD = MirrorDescent(stepsize = δ, max_Δ = 0.1, max_T = Inf)
    check_bias(prob, MD; N = N);
end

for δ in δs
    CRP = ConstantRateProgress(stepsize = δ, max_Δ = 0, max_T = Inf)
    check_bias(prob, CRP; N = N);
end

divs = [0.001, 0.01, 0.1, 0.2, 0.5, 1]
for div in divs
    LS = LineSearch(divergence = div, max_T = Inf)
    check_bias(prob, LS; N = N);
end



