include("synthetic/funnel.jl")
include("logdensityprobs.jl")
using AdaptiveAIS
using Functors
using Bijectors, FunctionChains
using LinearAlgebra
using Optimisers

struct FunnelTarget{G<:Real} 
    dist::Funnel{G}
end
FunnelTarget(dim::Int) = FunnelTarget(Funnel(dim))

function LogDensityProblems.capabilities(::FunnelTarget)
    LogDensityProblems.LogDensityOrder{1}()
end

LogDensityProblems.dimension(πT::FunnelTarget) = length(πT.dist)
function LogDensityProblems.logdensity(πT::FunnelTarget, x::AbstractVector)
    return logpdf(πT.dist, x)
end

# function LogDensityProblems.logdensity_and_gradient(πT::FunnelTarget, x::AbstractVector)
#     l = logpdf(πT.dist, x)
#     ∇l = score(πT.dist, x)
#     return l, ∇l
# end


# flow reference distribution
function create_planar_flow(n_layers::Int, q₀)
    d = length(q₀)
    Ls = [PlanarLayer(d) for _ in 1:n_layers]
    ts = fchain(Ls)
    return transformed(q₀, ts)
end

 
#####################
# setup ais problem
#####################
D = 2
L = LinearPath()

# p0 = GaussianReference(zeros(D))
# create a 10-layer planar flow as reference
nlayers = 10
@leaf MvNormal # to prevent MvNormal from being transformed

std_normal = MvNormal(zeros(D), ones(D))
q0 = Bijectors.transformed(std_normal, Bijectors.Shift(zeros(D)) ∘ Bijectors.Scale(Diagonal(ones(D))))

p0 = create_planar_flow(nlayers, q0)
p1 = FunnelTarget(D)
prob = AISProblem(p0, p1, L)

# destrcut the prob
ps, re = Optimisers.destructure(prob)


# test run
N = 32
MD = MirrorDescent(stepsize = 0.1, max_Δ = 0.5, max_T = Inf)
a = ais(prob, MD; N = N, save_trajectory = false, show_report = true, transition_kernel = RWMH_sweep())
FS = FixedSchedule(a.schedule)

rngs = SplitRandomArray(N; seed = 1)
xs = iid_sample_reference(rngs, prob, N)
# mutate!(rngs, RWMH_sweep(), prob, 0.5, xs)
# AdaptiveAIS.mutate_and_weigh!(rngs, MD, prob, RWMH_sweep(), 0.5, xs, zeros(N), zeros(N), 0.1)

# elbo(rngs, prob, FS; N = N, transition_kernel = RWMH_sweep())

# elbo(rngs, re(ps), MD; N = N, transition_kernel = RWMH_sweep())

# DMD = DebiasOnlineScheduling(MD)
# elbo(rngs, prob, DMD; N = N, transition_kernel = RWMH_sweep())



# loss = θ -> -elbo(rngs, re(θ), MD; N = N, transition_kernel = RWMH_sweep())[1]

# GE = TwoPointZeroOrderSmooth()
# gt = get_gradient(GE, loss, ps)


# to use it for actual update loop
# opt = DecayDescent(0.1)
# st = Optimisers.setup(opt, ps)

# st, ps = Optimisers.update!(st, ps, gt)

# gt = grad_and_update_state!(GE, loss, ps)

# ls, gt = value_grad_and_update_state!(GE, loss, ps)


rng = rngs[1]
logws, logas, β= AdaptiveAIS._compute_log_weights(rng, prob, MD, xs; N = N, transition_kernel = RWMH_sweep())

ZO = TwoPointZeroOrderSmooth()
CB = CondBernoulli()
CBCV = CondBernoulliCV()

kl_objective(prob, ZO, MD, rng, N,  RWMH_sweep())
kl_objective(prob, CB, MD, rng, N,  RWMH_sweep())
kl_objective(prob, CBCV, MD, rng, N,  RWMH_sweep())

ps, re = Optimisers.destructure(prob)
lld(θ_) = kl_objective(θ_, re, CB, DMD, rng, N, RWMH_sweep())
lld(ps)


Zygote.refresh()

using Zygote
import DifferentiationInterface as DI
using ADTypes


# test gradient of loss

AD = AutoZygote()

DI.gradient(llls, AD, ps)
DI.value_and_gradient(llls, AD, ps)

Zygote.gradient(lld, ps)
