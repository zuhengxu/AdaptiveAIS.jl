using Pigeons: slice_sample!, SliceSampler

struct CoordSliceSampler <: TransitionKernel
    ss::SliceSampler
end

CoordSliceSampler(w::Float64, p::Int, n_passes::Int, max_iter::Int) = 
    CoordSliceSampler(SliceSampler(w, p, n_passes, max_iter))

CoordSliceSampler() = CoordSliceSampler(SliceSampler())

function step!(rng, kernel::CoordSliceSampler, log_potential::Function, state)
    cached_lp = -Inf
    for _ in 1:kernel.ss.n_passes
        # replica = Replica(state, 1, rng, (;), 1)
        # cached_lp = slice_sample!(kernel.ss, state, log_potential, cached_lp, replica)
        replica = (; state, rng, chain = 1, recorders = (;))
        cached_lp = slice_sample!(kernel.ss, state, log_potential, cached_lp, replica)
    end
end



# N = 10
# seed = 1
# h = CoordSliceSampler()
# state = zeros(2)
# n = 1000
# states = Vector{typeof(state)}(undef, n)
# cached_lp = -Inf
# f = Base.Fix1(logpdf, MvNormal(10*ones(2), 0.5))
# rng = Random.default_rng()

# for i in 1:n
#     step!(rng, h, f, state)
#     states[i] = copy(state)
# end

# using Plots
# scatter([s[1] for s in states], [s[2] for s in states], label="Samples", legend=:topleft)
