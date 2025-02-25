using LogDensityProblems
using DifferentiableAIS
using Distributions, Random
using DifferentiationInterface, Zygote, ForwardDiff
using Optimisers
using LogExpFunctions
using PDMats
using Statistics
using StatsFuns
using JLD2
using Base.Threads: @threads
using LinearAlgebra

function run_dsmc(
    prob,
    id, kernel_type, BK;
    init_T = 32, max_T = 128,
    ula_logstepsize::Vector{<:Real} = -6 * ones(LogDensityProblems.dimension(prob.target)),
    uha_damping::Real = 0.1, uha_stepsize::Vector{<:Real} = 0.05 * ones(LogDensityProblems.dimension(prob.target)), share_damping::Bool = true,
    batchsize = 64,
    lr = 1e-4,
    nrounds = 3,
    nparticles_inference = 256,
    max_iters = 10_000,
    logging_frequency = 500,
    adbackend = AutoZygote(),
    save_result = true,
)
    Random.seed!(id)
    D = LogDensityProblems.dimension(prob.target)
    
    # this tracking has to be instantiate inside of this function otherwise when doing parallel computing, it will carry calls from other runs
    prob_tracked = DifferentiableAIS.TrackAISProblem(prob)

    if kernel_type <: UHA
        ps = (
            D = D,
            damping = uha_damping,
            stepsize = uha_stepsize,
            share_damping = share_damping,
        )
        Ms = instantiate_kernels(UHA, ps, init_T)
    else
        ps = (D = D, log_stepsize = ula_logstepsize)
        Ms = instantiate_kernels(Vector{ULA}, ps, init_T)
    end

    argument_string =
        "dsmc_$(kernel_type)_$(BK)/" *
        string(ps) *
        "/$(init_T)_$(max_T)_$(batchsize)_$(nrounds)_$(nparticles_inference)_$(max_iters)_$(lr)/" * 
        "$(id)"

    @info "Running $argument_string"

    β0 = range(0.0, 1.0, length = init_T).^2
    S0 = FixedSchedule(β0)
    
    thresh = ceil(Int, batchsize / 2)
    out, stats_cb = dsmc_with_rescheduling(
        prob_tracked,
        S0,
        BK,
        Ms,
        thresh,
        batchsize,
        adbackend;
        nrounds = nrounds,
        nparticles_inference = nparticles_inference,
        max_sched_length = max_T,
        max_iters = max_iters,
        optimiser = Optimisers.Adam(lr),
        train_logging = true,   # turn on the intrianing logz logging
        logging_frequency = logging_frequency, # logging frequency in training
        prob_untracked = prob,  # has to pass in the untracked problem for intrianing logging
    )
    results = (
        kernels = out.kernels,
        backward_kernel = BK,
        schedule = out.schedule,
        global_barrier = out.global_barrier,
        log_normalizer = stats_cb.logZs,
        n_density_evals = stats_cb.n_density_evals,
        n_gradient_evals = stats_cb.n_gradient_evals,
        n_hessian_evals = stats_cb.n_hessian_evals,
    )

    if save_result
        @info "Saving results for $(argument_string)"
        JLD2.save("results/$(argument_string).jld2", Dict(pairs(results)))
    end
    return results
end


function run_dais(
    prob,
    id, kernel_type, BK;
    init_T = 32, max_T = 128,
    ula_logstepsize::Vector{<:Real} = -6 * ones(LogDensityProblems.dimension(prob.target)),
    uha_damping::Real = 0.1, uha_stepsize::Vector{<:Real} = 0.05 * ones(LogDensityProblems.dimension(prob.target)), share_damping::Bool = true,
    batchsize = 64,
    lr = 1e-4,
    nrounds = 3,
    nparticles_inference = 256,
    max_iters = 10_000,
    logging_frequency = 500,
    adbackend = AutoZygote(),
    save_result = true,
)
    Random.seed!(id)
    D = LogDensityProblems.dimension(prob.target)

    # this tracking has to be instantiate inside of this function otherwise when doing parallel computing, it will carry calls from other runs
    prob_tracked = DifferentiableAIS.TrackAISProblem(prob)

    if kernel_type <: UHA
        ps = (
            D = D,
            damping = uha_damping,
            stepsize = uha_stepsize,
            share_damping = share_damping,
        )
        Ms = instantiate_kernels(UHA, ps, init_T)
    else
        ps = (D = D, log_stepsize = ula_logstepsize)
        Ms = instantiate_kernels(Vector{ULA}, ps, init_T)
    end

    argument_string =
        "dsmc_$(kernel_type)_$(BK)/" *
        string(ps) *
        "/$(init_T)_$(max_T)_$(batchsize)_$(nrounds)_$(nparticles_inference)_$(max_iters)_$(lr)/" * 
        "$(id)"

    @info "Running $argument_string"

    β0 = range(0.0, 1.0, length = init_T).^2
    S0 = FixedSchedule(β0)
    out, stats_cb = dais_with_rescheduling(
        prob_tracked,
        S0,
        BK,
        Ms,
        batchsize,
        adbackend;
        nrounds = nrounds,
        max_sched_length = max_T,
        nparticles_inference = nparticles_inference,
        max_iters = max_iters,
        optimiser = Optimisers.Adam(lr),
        train_logging = true,   # turn on the intrianing logz logging
        logging_frequency = logging_frequency, # logging frequency in training
        prob_untracked = prob,  # has to pass in the untracked problem for intrianing logging
    )
    results = (
        kernels = out.kernels,
        backward_kernel = BK,
        schedule = out.schedule,
        global_barrier = out.global_barrier,
        log_normalizer = stats_cb.logZs,
        n_density_evals = stats_cb.n_density_evals,
        n_gradient_evals = stats_cb.n_gradient_evals,
        n_hessian_evals = stats_cb.n_hessian_evals,
    )
    
    if save_result
        @info "Saving results for $(argument_string)"
        JLD2.save("results/$(argument_string).jld2", Dict(pairs(results)))
    end
    return results
end


function sweepsrun(
    prob::AISProblem,
    run_func = run_dsmc,
    kernel_type = UHA,
    BK::BackwardKernelType = DefaultReverse();
    nrounds = 1, # nrounds = 1 means no sched adapataion
    nreps = 5,
    max_iter = 3000,
    logging_freq = 10,
    batchsizes = [32],
    Ts = [4, 8, 16],
    lrs = [1e-4, 1e-3, 1e-2],
    ula_logstepsize = zeros(D),
    uha_damping = 0.1,
    uha_stepsize = ones(D),
    share_damping = true,
    fname_prefix = nothing,
)
    ndata = Int(floor(Int, max_iter / logging_freq) * nrounds)   # length of logged vectors
    for T in Ts
        for bs in batchsizes

            for lr in lrs
                logZs = zeros(ndata, nreps)
                n_lpdfs = zeros(ndata, nreps)
                n_grads = zeros(ndata, nreps)
                n_hess = zeros(ndata, nreps)

                @threads for id in 1:nreps
                    res = run_func(
                        prob, id, kernel_type, BK,
                        nrounds = nrounds, 
                        ula_logstepsize = ula_logstepsize, uha_damping = uha_damping, uha_stepsize = uha_stepsize, share_damping = share_damping,
                        init_T = T, max_T = 64, lr = lr, batchsize = bs, max_iters = max_iter,
                        logging_frequency = logging_freq,
                        nparticles_inference = 1024,
                        adbackend = AutoZygote(),
                        save_result = false,
                    )
                    logZs[:, id] = res.log_normalizer
                    n_lpdfs[:, id] = res.n_density_evals
                    n_grads[:, id] = res.n_gradient_evals
                    n_hess[:, id] = res.n_hessian_evals
                end
                method_name =
                    string(run_func) * "/" * string(kernel_type) * "_" * string(BK) * "_" * string(share_damping)

                fname_pre = isnothing(fname_prefix) ? "results/sweeps/" * method_name : fname_prefix * "/" * method_name
                fname = fname_pre * "_T_$(T)_bs_$(bs)_lr_$(lr).jld2"

                @info "Saving sweeps for $(method_name) with T = $(T), bs = $(bs), and lr = $(lr)"
                JLD2.save(
                    fname,
                    Dict(
                        "logZs" => logZs,
                        "n_lpdfs" => n_lpdfs,
                        "n_grads" => n_grads,
                        "n_hess" => n_hess,
                        "T" => T,
                        "lr" => lr,
                        "method_name" => method_name,
                        "max_iter" => max_iter,
                        "logging_freq" => logging_freq,
                        "bs" => bs,
                    ),
                )
            end
        end
    end
end

