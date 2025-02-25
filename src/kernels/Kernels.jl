abstract type TransitionKernel end 


include("slice.jl") # somehow the coord slice is not working anymore
include("rwmh.jl")

step!(rng, kernel::TransitionKernel, prob::AISProblem, beta, state) = step!(rng, kernel, s -> log_annealed_density(prob, beta, s), state)  
step!(kernel::TransitionKernel, prob::AISProblem, beta, state) = step!(SplittableRandom(1), kernel, prob, beta, state)

"""
    mutate!(rngs, transition_kernel, prob, next_beta, states)

just parallel execution of step! for each particle in states
"""
function mutate!(rngs, transition_kernel, prob::AISProblem, next_beta, states)
    N = size(states, 2)
    @threads for i in 1:N
        rng = rngs[i]
        state = @view(states[:, i])
        step!(rng, transition_kernel, prob, next_beta, state)
    end
end
