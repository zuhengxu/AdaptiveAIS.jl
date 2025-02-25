# code modified from  https://github.com/alexandrebouchard/sais-gpu/blob/main/barriers.jl

function intensity(log_increments) 
    log_weights = cumsum(log_increments, dims = 1)
    g1 = compute_log_g(log_weights, log_increments, 1)
    g2 = compute_log_g(log_weights, log_increments, 2)
    return sqrt.(ensure_non_negative.(g2 .- 2 .* g1))
end

# In the paper's notation, computes log_g_exponent - log_g_0
function compute_log_g(log_weights, log_increments, exponent::Int) 
    T, _ = size(log_increments) 
    
    log_sum = 
        @view(log_weights[1:(T-1), :]) + 
        @view(log_increments[2:T, :]) .* exponent 
    result = log_sum_exp(log_sum, dims = 2) 

    weight_log_norms = log_sum_exp(log_weights, dims = 2)
    result .= result .- @view(weight_log_norms[1:(T-1), :])

    return vec(result)
end

# we have to do this because some slightly negative 
# (i.e., larger than -1e4) intensities do pop up 
# because of numerical error
ensure_non_negative(x) = max(x, 0)


# update schedule
function find_optimal_schedule(log_increments, old_schedule::AbstractVector, T::Int)
    intensity_vector = intensity(log_increments)
    schedule = Pigeons.optimal_schedule(intensity_vector, old_schedule, T)
    return schedule
end

