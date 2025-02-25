# annealing path between reference and target
abstract type AbstractPath end

# default annealing path: linear path (or in some literature, called geometric path)
struct LinearPath <: AbstractPath end
anneal(::LinearPath, gamma, x, y) = (1 - gamma) * x + gamma * y
