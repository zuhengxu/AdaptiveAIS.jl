using LogDensityProblems
using ADTypes
import DifferentiationInterface as DI

struct DIHessian{B<:ADTypes.AbstractADType,G,H,L}
    backend::B
    prep_grad::G
    prep_hess::H
    ℓ::L
end

function _logdensity_switched(x, ℓ)
    return LogDensityProblems.logdensity(ℓ, x)
end

function ADHessian(ℓ::L, backend::B; x::Union{Nothing,AbstractVector}=nothing) where {B,L}
    if x === nothing
        prep_grad = nothing
        prep_hess = nothing
    else
        prep_grad = DI.prepare_gradient(_logdensity_switched, backend, x, DI.Constant(ℓ))
        prep_hess = DI.prepare_hessian(_logdensity_switched, backend, x, DI.Constant(ℓ))
    end
    return DIHessian(backend, prep_grad, prep_hess, ℓ)
end
LogDensityProblems.dimension(ℓ::DIHessian) = LogDensityProblems.dimension(ℓ.ℓ)

LogDensityProblems.capabilities(::Type{<:DIHessian}) = LogDensityProblems.LogDensityOrder{2}()

LogDensityProblems.logdensity(ℓ::DIHessian, x::AbstractVector) = LogDensityProblems.logdensity(ℓ.ℓ, x)

function LogDensityProblems.logdensity_and_gradient(Hℓ::DIHessian, x::AbstractVector)
    (; backend, prep_grad, prep_hess, ℓ) = Hℓ
    if prep_grad === nothing
        return DI.value_and_gradient(_logdensity_switched, backend, x, DI.Constant(ℓ))
    else
        return DI.value_and_gradient(_logdensity_switched, prep_grad, backend, x, DI.Constant(ℓ))
    end
end

function LogDensityProblems.logdensity_gradient_and_hessian(Hℓ::DIHessian, x::AbstractVector)
    (; backend, prep_grad, prep_hess, ℓ) = Hℓ
    if prep_hess === nothing
        return DI.value_gradient_and_hessian(_logdensity_switched, backend, x, DI.Constant(ℓ))
    else
        return DI.value_gradient_and_hessian(_logdensity_switched, prep_hess, backend, x, DI.Constant(ℓ))
    end
end


# p1_ad = ADHessian(p1, AutoForwardDiff(); x = randn(10))
#
# LogDensityProblems.logdensity_and_gradient(p1_ad, randn(10))
# LogDensityProblems.logdensity_gradient_and_hessian(p1_ad, randn(10))
#
# LogDensityProblems.capabilities(p1_ad)
#
# prob_ad = AISProblem(p0, p1_ad, L)
# log_annealed_gradient(prob_ad, 0.5, randn(10))
# log_annealed_hessian(prob_ad, 0.5, randn(10))
# log_annealed_density(prob_ad, 0.5, randn(10))
