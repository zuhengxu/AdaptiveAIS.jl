using Plots, StatsBase, Statistics

function get_percentiles(dat; p1=25, p2=75)
    n = size(dat,1) 

    plow = zeros(n)
    phigh = zeros(n)

    for i in 1:n
        dat_remove_nan = (dat[i, :])[iszero.(isnan.(dat[i, :]))]
        median_remove_nan = median(dat_remove_nan)
        plow[i] = median_remove_nan - percentile(vec(dat_remove_nan), p1)
        phigh[i] = percentile(vec(dat_remove_nan), p2) - median_remove_nan
    end

    return plow, phigh
end

