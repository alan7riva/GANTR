using Distributions
import StatsBase.ecdf
#using Plots


function s_weibull( x::Float64, k::Float64=1.0, λ::Float64=1.0)
    return 1.0-cdf( Weibull(k, λ), x)
end

#using PyPlot
# figure()
# plot( collect(0.:1.0/100.:5.),[s_weibull(i, 2.2, 1.75) for i in collect(0.:1.0/100.:5.)], linewidth=1.1, label = "\$ Y\\^{(0)} survival with \$l=1\$"  )
# plot( collect(0.:1.0/100.:5.),[s_weibull(i, 1.2, 2.2) for i in collect(0.:1.0/100.:5.)], linewidth=1.1, label = "\$ Y^{(1)} survival with \$l=0\$"   )
# plot( collect(0.:1.0/100.:5.),[s_weibull(i, 5.3, 1.5) for i in collect(0.:1.0/100.:5.)], linewidth=1.1, label = "\$ Y^{(2)} survival with \$l=0\$" )
# ylabel("\$ S(t) \$")
# xlabel("\$ t \$")
# title("Simulated data true survival functions")
# legend(loc=1)
# savefig("weibulls.png")




c_sw = [5.0,2.2]
c_ws = [1.5,3.4]
c_m = [2.1,2.0]

function s_comprisk_weibull( x::Float64, k₁::Float64=1.0, k₂::Float64=1.0,
                        λ₁::Float64=1.0,  λ₂::Float64=1.0)
    return exp( log( ccdf( Weibull(k₁, λ₁), x) ) + log( ccdf( Weibull(k₂, λ₂), x) ) )
end

# plot( collect(0.:1.0/100.:5.),[s_comprisk_weibull(i, 5.0, 1.2, 1.8, 2.2) for i in collect(0.:1.0/100.:5.)] )
# plot!( collect(0.:1.0/100.:5.),[s_comprisk_weibull(i, 5.0, 4.3, 1.8, 1.5) for i in collect(0.:1.0/100.:5.)] )


function sim_study_sampler_comprisk( n::Int64, δ::Float64, thr::Float64,
        k₀::Float64=1.0, λ₀::Float64=1.0, k₁::Float64=1.0, λ₁::Float64=1.0,
        k₂::Float64=1.0, λ₂::Float64=1.0,
        μ_co::Float64=1.0, σ_co::Float64=0.75, d_co::Float64=0.0, u_co::Float64=2.0)
    samp_Z = zeros( Float64, n)
    samp_δ = zeros( Float64, n)
    samp_Y = zeros( Float64, n)
    for i = 1:n
        samp_Z[i] = rand( TruncatedNormal( μ_co, σ_co, d_co, u_co) )
        if samp_Z[i] < thr
            samp_Y[i] = min( rand( Weibull( k₁, λ₁/(1.0-δ)^(1.0/k₁) ) ),
                                 rand( Weibull( k₀, λ₀/(δ)^(1.0/k₀) ) ) )
        else
            samp_Y[i] = min( rand( Weibull( k₂, λ₂/(1.0-δ)^(1.0/k₂) ) ),
                                 rand( Weibull( k₀, λ₀/(δ)^(1.0/k₀) ) ) )
        end
    end
    return [ samp_Y, samp_Z ]
end

function true_surv_comprisk( x::Float64, z::Float64, δ::Float64, thr::Float64,
    k₀::Float64=1.0, λ₀::Float64=1.0, k₁::Float64=1.0, λ₁::Float64=1.0,
    k₂::Float64=1.0, λ₂::Float64=1.0)
    if z <= thr
        return s_comprisk_weibull( x, k₁, k₀,
        λ₁/(1.0-δ)^(1.0/k₁), λ₀/(δ)^(1.0/k₀) )
    else
        return s_comprisk_weibull( x, k₂, k₀,
        λ₂/(1.0-δ)^(1.0/k₂), λ₀/(δ)^(1.0/k₀) )
    end
end

# Kaplan-Meier estiamtor function
function KMest(tiempo, muerte)
    t_KMest = sort( unique( tiempo ) )
    if t_KMest[1] != 0
        t_KMest = [0; t_KMest]
    end
    n_KMest = [count( i->(i >= j), tiempo) for j in t_KMest]
    d_KMest = [sum( muerte[ findall(j->j==i,tiempo) ] ) for i in t_KMest]
    s_KMest = cumprod( 1 .- d_KMest./n_KMest )
    return t_KMest, s_KMest
end

# plot( collect(0.:1.0/100.:5.), [true_surv_comprisk( i, 0.5, 0.1, 1.0, 5.0, 1.8, 1.2, 2.2, 4.3, 1.5) for i in collect(0.:1.0/100.:5.)])
# plot!( collect(0.:1.0/100.:5.), [true_surv_comprisk( i, 1.5, 0.1, 1.0, 5.0, 1.8, 1.2, 2.2, 4.3, 1.5) for i in collect(0.:1.0/100.:5.)])
