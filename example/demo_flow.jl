include("synthetic/funnel.jl")
include("logdensityprobs.jl")
using AdaptiveAIS 
using Functors
using Bijectors
using LinearAlgebra
using Optimisers
using Zygote
using ADTypes
using Base.Threads: @threads
using DataFrames, CSV, JLD2

# Zygote.refresh()

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

# flow reference distribution
function create_planar_flow(n_layers::Int, q₀)
    d = length(q₀)
    Ls = reduce(∘, [PlanarLayer(d) for _ in 1:n_layers])
    # ts = fchain(Ls) # fchain seems to be broken for zygote
    return transformed(q₀, Ls)
end

 
#####################
# setup ais problem
#####################
D = 5
L = LinearPath()

nlayers = 10
@leaf MvNormal # to prevent MvNormal from being transformed

std_normal = MvNormal(zeros(D), ones(D))
q0 = Bijectors.transformed(std_normal, Bijectors.Shift(zeros(D)) ∘ Bijectors.Scale(Diagonal(ones(D))))

p0 = create_planar_flow(nlayers, q0)
p1 = FunnelTarget(D)
prob = AISProblem(p0, p1, L)


rng = Random.default_rng()

ZO = TwoPointZeroOrderSmooth()
CB = CondBernoulli()
CBCV = CondBernoulliCV()

# kl_objective(rng, prob, ZO, MD, N, kernel)
# kl_objective(rng, prob, CB, MD, N, kernel)
# kl_objective(rng, prob, CBCV, MD, N, kernel)


# Zygote.gradient(lld, ps)
lr = 1e-3
kernel = RWMH_sweep()
AD = AutoZygote()
# using Mooncake
# AD = AutoMooncake(; config = Mooncake.Config())

MD = MirrorDescent(stepsize = 0.1, max_Δ = 0.5, max_T = Inf)
FS = FixedSchedule(32)
DMD = DebiasOnlineScheduling(MD)
bs = 32
niters = 10_000

scheds = [MD, DMD, FS]


# for sched in scheds
#     @threads for id in 1:5
#         AdaptiveAIS.init_state!(ZO, 1)
#         prob_trained, train_stats, _, _ = dais_train(
#             rng, 
#             prob,
#             ZO, 
#             sched, 
#             bs, 
#             kernel;
#             adbackend = ZO, 
#             max_iters = niters,
#             optimiser = Optimisers.ADAM(lr),
#         )
#         out = process_logging(train_stats)

#         # save csv
#         fpath = "funnel_res/$(sched)_$(id).csv"
#         df = DataFrame(out)
#         CSV.write(fpath, df)

#         # save jld2
#         fp_jld = "funnel_res/$(sched)_$(id).jld2"
#         JLD2.save(fp_jld, "out", out)
#     end
# end

# a = ais(prob, FS; N = bs, save_trajectory = false, show_report = true, transition_kernel = RWMH_sweep())

for sched in scheds
    @threads for id in 1:5
        AdaptiveAIS.init_state!(ZO, 1)
        prob_trained, train_stats, _, _ = dais_train(
            rng, 
            prob,
            ZO, 
            sched, 
            bs, 
            kernel;
            adbackend = ZO, 
            max_iters = niters,
            optimiser = DecayDescent()
        )
        out = process_logging(train_stats)

        # save csv
        fpath = "funnel_res/Decay_$(sched)_$(id).csv"
        df = DataFrame(out)
        CSV.write(fpath, df)

        # save jld2
        fp_jld = "funnel_res/Decay_$(sched)_$(id).jld2"
        JLD2.save(fp_jld, "out", out)
    end
end

# out = process_logging(train_stats)
# # turn it into a DataFrame
# df = DataFrame(out)
# CSV.write("funnel_res/demo_flow.csv", df)

# using DataFrames
# df2 = CSV.read("funnel_res/demo_flow.csv", DataFrame)

# oo = JLD2.load(fp_jld)
