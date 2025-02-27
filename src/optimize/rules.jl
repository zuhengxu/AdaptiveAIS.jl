# using Optimisers


# Define a container to hold any optimiser specific parameters (if any):
Optimisers.@def struct DecayDescent <: Optimisers.AbstractRule
    eta = 1.0
end

# Define an `apply!` rule which encodes how the gradients will be used to
# update the parameters:
function Optimisers.apply!(o::DecayDescent, state, x, x̄)
    T = eltype(x)
    newx̄ = T(o.eta / √state) .* x̄
    nextstate = state + 1
    return nextstate, newx̄
end

# Define the function which sets up the initial state (if any):
# in this case, state is t --- the current number of iters
Optimisers.init(o::DecayDescent, x::AbstractArray) = 1

# # to use it for actual update loop
# opt = DecayDescent(0.1)
# θ = randn(10)
# st = Optimisers.setup(opt, θ)

# g = 100rand(10)

# st, θ = Optimisers.update!(st, θ, g)

