using LogDensityProblems

# broadcast over each column of xs
function LogDensityProblems.logdensity(prob, xs::AbstractMatrix)
    return LogDensityProblems.logdensity.(Ref(prob), eachcol(xs))
end

# function ∇logpdf(prob, x::AbstractVector)
#     _, ∇ℓπt = LogDensityProblems.logdensity_and_gradient(prob, x)
#     return ∇ℓπt
# end

# function ∇logpdf(prob, xs::AbstractMatrix)
#     return mapslices(xi -> ∇logpdf(prob, Vector(xi)), xs; dims=1)
# end

# example of manually defining everything

# function LogDensityProblems.capabilities(::Type{<:MvNormal})
#     LogDensityProblems.LogDensityOrder{1}()
# end
#
# LogDensityProblems.dimension(dist::MvNormal) = length(dist)
#
# function LogDensityProblems.logdensity(dist::MvNormal, x::AbstractVector)
#     return logpdf(dist, x)
# end
#
# function LogDensityProblems.logdensity_and_gradient(dist::MvNormal, x::AbstractVector)
#     l = logpdf(dist, x)
#     ∇l = ForwardDiff.gradient(y -> logpdf(dist, y), x)
#     return l, ∇l
# end


