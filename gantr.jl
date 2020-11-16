using Distributions, ProgressMeter, IterTools, LinearAlgebra
import StatsFuns.logistic

struct Data
  T::Array{Float64,1} # observation times
  δ::Array{Int64,1} # censoring indicator, 1 if exact observation
  Z::Array{Array{Float64,1},1} # covariate in dimension 1
end

struct Model
    data::Data
    obs::Array{ Float64, 1} # observed times, sorted and without repetitions
    last_inc::Float64 # to be added to the last observation to set overall time window in model
    incs::Array{ Float64, 1} # array of increments between observations
    E::Array{ Array{ Int64, 1}, 1} # array of indexes of exact observation corresponding to obs
    C::Array{ Array{ Int64, 1}, 1} # array of indexes of censored observation corresponding to obs
    f::Array{Function,1} # array of regression function in covariates corresponding to obs[i] CoRM dimension 1
    Hᵉ::Function # frequencies of exact observations in dimension 1
    Hᶜ::Function # frequencies of censored observations in dimension 1
    e_o::Array{ Int64, 1} # array of indexes of the exact observations in obs
    e_o_f::Function
end

# function to initialize model struct
@inline function Model(data::Data, last_inc::Float64, f::Array{Function,1})
      obs = sort( unique( data.T ) )
      incs = push!( diff( [ 0; obs ] ), last_inc)
      E = Array{Int64,1}[]
      C = Array{Int64,1}[]
      for i in 1:length( obs )
        push!( E, intersect( findall( data.T .== obs[i] ), findall( data.δ .== 1 ) ) )
        push!( C, intersect( findall( data.T .== obs[i] ), findall( data.δ .== 0 ) ) )
      end
      @inbounds hᵉ(k::Int64, β::Array{Float64,1}) = [ mapreduce( j -> f[k]( data.Z[j], β), +, E[i], init = 0.0 ) for i in 1:length(obs)]
      @inbounds hᶜ(k::Int64, β::Array{Float64,1}) = [ mapreduce( j -> f[k]( data.Z[j], β), +, C[i], init = 0.0 ) for i in 1:length(obs)]
      @inbounds Hᵉ(k::Int64, β::Array{Float64,1}) = [ cumsum( hᵉ(k,β)[end:-1:1] )[end:-1:1]; 0]
      @inbounds Hᶜ(k::Int64, β::Array{Float64,1}) = [ cumsum( hᶜ(k,β)[end:-1:1] )[end:-1:1]; 0]
      @inbounds Hᶜ(k::Int64, β::Array{Float64,1}) = [ cumsum( hᶜ(k,β)[end:-1:1] )[end:-1:1]; 0]
      e_o = findall( E .!= [ [] ] )
      e_o_f(s) = intersect( findall( obs .<= s ), e_o)
      return Model( data, obs, last_inc, incs, E, C, f, Hᵉ, Hᶜ, e_o, e_o_f)
end

# Random walk Metropolis-Hastings algorithm in the unit interval at log scale
@inline function metr_hast_chain_unif_a_b_interv_logscale_justend( lf::Function, l::Int64, a::Float64, b::Float64, σ::Float64)
    log_π_σ = lf(σ)
    for i in 1:l
        σ_prop = rand( Uniform(a,b) )
        log_π_σ_prop = lf(σ_prop)
        u = rand(Uniform())
        if log(u) < min( log_π_σ_prop - log_π_σ , 0. ) # symmetric q(x,y) AS IT IS gaussian random walk
            σ = σ_prop
            log_π_σ =log_π_σ_prop
        end
    end
    return σ
end

mi_logistic(x::Real,a::Real,b::Real) = (b + a*exp(-x))*inv(exp(-x) + one(x))
# Random walk Metropolis-Hastings algorithm at log scale
@inline function metr_hast_chain_randwalk_a_b_interv_logscale_justend( lf::Function, l::Int64, var::Float64, a::Float64, b::Float64, σ::Float64)
    log_π_σ = lf(σ)
    for i in 1:l
        σ_prop = mi_logistic( rand( Normal( log(σ-a) - log(b-σ), var) ), a, b)
        log_π_σ_prop = lf(σ_prop)
        log_q_given_σ = - log( σ_prop - a ) - log( b - σ_prop ) # + log(b-a) cancels out
        log_q_given_σ_prop = - log( σ - a ) - log( b - σ ) # + log(b-a) cancels out
        u = rand(Uniform())
        if log(u) < min( log_π_σ_prop - log_π_σ + log_q_given_σ_prop - log_q_given_σ, 0. ) # symmetric q(x,y) AS IT IS gaussian random walk
            σ = σ_prop
            log_π_σ = log_π_σ_prop
        end
    end
    return σ
end

# Random walk Metropolis-Hastings algorithm at log scale
@inline function metr_hast_chain_gauss_randwalk_logscale_justend( lf::Function, l::Int64, var::Float64, σ::Float64=0.0)
    log_π_σ = lf(σ)
    for i in 1:l
        σ_prop = rand(Normal(σ, var))
        log_π_σ_prop = lf(σ_prop)
        u = rand(Uniform())
        if log(u) < min( log_π_σ_prop - log_π_σ , 0. ) # symmetric q(x,y) AS IT IS gaussian random walk
            σ = σ_prop
            log_π_σ = log_π_σ_prop
        end
    end
    return σ
end

mi_LogNormal( m::Float64, v::Float64) = LogNormal( log(m) - 0.5*log( v/m^2 + 1. ), ( log( v/m^2 + 1. ) )^0.5 )

@inline function clayton_corm_lognorm_2dim_sampler(  δ::Float64, σ²::Float64)
    loc_u = rand( Uniform() )
    if loc_u < 1.0/2.0
        return [ rand( mi_LogNormal( 1.0*(1.0-δ) + δ, σ²)), rand( mi_LogNormal( 0.01*(1.0-δ) + δ, σ²)) ]
    else
        return [ rand( mi_LogNormal( 0.01*(1.0-δ) + δ, σ²)), rand( mi_LogNormal( 1.0*(1.0-δ) + δ, σ²)) ]
    end
end

# Uniform (0,1) Metropolis-Hastings algorithm at log scale
@inline function metr_hast_chain_unif_LogNorm_delta_logscale_justend( lf::Function,
        δ::Float64, V::Array{Array{Float64,1},1}, LogNorm_v::Float64, l_mc::Int64, l::Int64)
    log_π_V = lf(V)
    for i in 1:l
        δ_prop = rand( Uniform() )
        V_prop = [ clayton_corm_lognorm_2dim_sampler( δ_prop, LogNorm_v) for j in 1:l_mc ]
        log_π_V_prop = lf(V_prop)
        u = rand(Uniform())
        if log(u) < min( log_π_V_prop - log_π_V , 0.) # symmetric q(x,y) AS IT IS UNIFORM
            δ  = δ_prop
            V = V_prop
            log_π_V = log_π_V_prop
        end
    end
    return δ, V
end

mi_logistic(x::Real,a::Real,b::Real) = (b + a*exp(-x))*inv(exp(-x) + one(x))

# Uniform (0,1) Metropolis-Hastings algorithm at log scale
@inline function metr_hast_chain_randwalk_LogNorm_delta_logscale_justend( lf::Function,
        δ::Float64, V::Array{Array{Float64,1},1}, LogNorm_v::Float64, l_mc::Int64, l::Int64,
        var::Float64)
    log_π_V = lf(V)
    for i in 1:l
        δ_prop =  logistic( rand( Normal( log(δ) - log(1.0-δ), var) ) )
        V_prop = [ clayton_corm_lognorm_2dim_sampler( δ_prop, LogNorm_v) for j in 1:l_mc ]
        log_π_V_prop = lf(V_prop)
        log_q_given_δ = - log( δ_prop ) - log( 1.0 - δ_prop )
        log_q_given_δ_prop = - log( δ ) - log( 1.0 - δ )
        u = rand(Uniform())
        if log(u) < min( log_π_V_prop - log_π_V + log_q_given_δ_prop - log_q_given_δ, 0.)
            δ  = δ_prop
            V = V_prop
            log_π_V = log_π_V_prop
        end
    end
    return δ, V
end

function ψ_gamma( λ::Float64, α::Float64)
    return log(1.0 + λ/α)
end

function ψ( λ::Array{Float64,1}, α::Float64, V::Array{Array{Float64,1},1})
    return mean( [ ψ_gamma( sum( λ.*V[i] ), α) for i in 1:length(V)] )
end

function ψ_dif( λ::Array{Float64,1}, q::Array{Float64,1}, α::Float64, V::Array{Array{Float64,1},1})
    return mean( [ ψ_gamma( sum( (λ+q).*V[i]), α) - ψ_gamma(  sum( λ.*V[i]), α)  for i in 1:length(V) ] )
end

#function to calculate the product similar to the binomial theorem Π_{i ∈ I}(1-aᵢ) that I have in my likelihoods
# allowing for multiplicities 1 and 2
@inline function simil_binom_prod( q::Array{Float64}, i::Int64, β::Array{Float64,1}, α::Float64, v::Array{Float64,1}, model::Model)
    if length( model.E[i]) ==  1
        return log(  ( 1.0 + dot( v./α, [ model.f[j]( model.data.Z[model.E[i][1]], β) for j in 1:length(q) ] + q ) ) / ( 1.0 +
                            dot( v./α, q ) ) )
    elseif length( model.E[i]) ==  2
        q₁ = dot( v./α, [ model.f[j]( model.data.Z[model.E[i][1]], β) for j in 1:length(q) ] )
        q₂ = dot( v./α, [ model.f[j]( model.data.Z[model.E[i][2]], β) for j in 1:length(q) ] )
        q₀ = dot( v./α, q )
        return log( 1.0 + q₁*q₂ / ( 1.0 + q₁ + q₂ + 2.0*q₀ + q₁*q₀ + q₂*q₀ + q₀^2.0) )
    else
        println( length( model.E[i]) )
    end
end

#function to calculate the product similar to the binomial theorem Π_{i ∈ I}(1-aᵢ) that I have in my likelihoods
# allowing for multiplicity 1, length(model.E[i])=1 for all i
# @inline function simil_binom_prod( q::Array{Float64}, i::Int64, β::Array{Float64,1}, α::Float64, v::Array{Float64,1}, model::Model)
#     return log(  ( 1.0 + dot( v./α, [ model.f[j]( model.data.Z[model.E[i][1]], β) for j in 1:length(q) ] + q ) ) / ( 1.0 +
#                             dot( v./α, q ) ) )
# end

@inline function simil_binom_prod( q::Array{Float64}, i::Int64, β::Array{Float64,1}, α::Float64, v::Array{Float64,1}, model::Model)
    return log( 1.0 + ( 1.0/(α + dot(v,q)) ) * dot( v, [ model.f[j]( model.data.Z[model.E[i][1]], β) for j in 1:length(q) ] ) )
end


#function to calculate the product similar to the binomial theorem Π_{i ∈ I}(1-aᵢ) that I have in my likelihoods
# @inline function simil_binom_prod( q::Array{Float64}, i::Int64, β::Array{Float64,1}, α::Float64, v::Array{Float64,1}, model::Model)
#     return mapreduce( S -> (-1)^length(S) * ( ψ_gamma( sum( v.*( q + [ sum( [ model.f[i]( model.data.Z[l], β) for l in S]) for i in 1:length(q) ] ) ), α) - ψ_gamma( sum( v.*q ), α) ),
#               +, collect( subsets(model.E[i]) )[2:end], init = 0.0 )
# end
function e_contrib_func( q::Array{Float64,1}, i::Int64, β::Array{Float64,1}, α::Float64, V::Array{Array{Float64,1},1}, model::Model)
    r = mean( [ simil_binom_prod( q, i, β, α, V[j], model) for j in 1:length(V)])
    if r < 0.0
        println( "r<0!!!")
        global loc_q = q
        global loc_i = i
        global loc_β = β
        global loc_α = α
        global loc_V = V
    end
    return r
end

@inline function model_loglikelihood( β::Array{Float64,1}, A::Float64, α::Float64, V::Array{Array{Float64,1},1}, model::Model)
  obs = [0.; model.obs]
  Hᶜ = [ model.Hᶜ(i,β) for i in 1:length(model.f) ]
  Hᵉ = [ model.Hᵉ(i,β) for i in 1:length(model.f) ]
  return  - A * mapreduce( j -> ( obs[j+1] - obs[j] ) * ψ( [ Hᶜ[i][j] + Hᵉ[i][j] for i in 1:length(model.f) ], α, V), +, 1:length(model.obs), init = 0.0 ) +
          mapreduce( j -> log( e_contrib_func( [ Hᶜ[i][j] + Hᵉ[i][j+1] for i in 1:length(model.f) ], j, β, α, V, model) ), +, model.e_o, init = 0.0 ) +
          length(model.e_o)*log(A)
end

@inline function MCMC_comprisk_uniform( n::Int64, m::Int64, l_mc::Int64, A::Float64, α::Float64, model::Model,
    LogNorm_v::Float64, ch_init::Array{Float64},
    delta_init::Float64, a_interv::Float64, b_interv::Float64, β_run_map::Array{Float64}=Float64[], δ_run_map::Float64=0.0)
    chain_δ = zeros(Float64, n+1)
    chain_β = [ zeros(Float64, 1) for i in 1:(n+1) ]
    M = -Inf
    if ch_init != 1.0 .+ zeros(Float64,1)
        chain_β[1] = ch_init
        chain_δ[1] = delta_init
    else
        chain_β[1] = [ rand( Uniform(a_interv,b_interv) ) ]
        chain_δ[1] = rand( Uniform())
    end
    δ_map = δ_run_map
    β_map = β_run_map
    if δ_run_map == 0.0
        δ_map = chain_δ[1]
        β_map = chain_β[1]
    end
    loc_V = [ clayton_corm_lognorm_2dim_sampler( chain_δ[1], LogNorm_v) for j in 1:l_mc ]
    println("MCMC run has started.")
    prog = Progress( n, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    for i in 1:n
        @inbounds lf_pseud_marg(V) = model_loglikelihood( chain_β[i], A, α, V, model)
        @inbounds chain_δ[i+1], loc_V =  metr_hast_chain_unif_LogNorm_delta_logscale_justend( lf_pseud_marg, chain_δ[i], loc_V, LogNorm_v, l_mc, m)
        @inbounds lf_β_nocox(b) =  model_loglikelihood( [ b ], A, α, loc_V, model) # + logpdf( Beta(1.7,1.7), 0.5*b ) # logpdf( TruncatedNormal( 1.0, 1.0, 0.0, 2.0), b)
        @inbounds chain_β[i+1][1] =  metr_hast_chain_unif_a_b_interv_logscale_justend( lf_β_nocox, m, a_interv, b_interv, chain_β[i][1])
        M_temp = model_loglikelihood( chain_β[i+1], A, α, loc_V, model) # + logpdf( Beta(1.7,1.7), 0.5*chain_β[i+1][1])
        if M < M_temp
            @inbounds β_map = chain_β[i+1]
            @inbounds δ_map = chain_δ[i+1]
            M = M_temp
        end
        next!(prog)
    end
    return chain_δ, chain_β, δ_map, β_map
end

@inline function MCMC_comprisk_randwalk( n::Int64, m::Int64, l_mc::Int64, A::Float64, α::Float64, model::Model,
    LogNorm_v::Float64, δ_var::Float64, β_var::Float64, ch_init::Array{Float64}, delta_init::Float64, a_interv::Float64, b_interv::Float64, β_run_map::Array{Float64}=Float64[], δ_run_map::Float64=0.0)
    chain_δ = zeros(Float64, n+1)
    chain_β = [ zeros(Float64, 1) for i in 1:(n+1) ]
    M = -Inf
    if ch_init != 1.0 .+ zeros(Float64,1)
        chain_β[1] = ch_init
        chain_δ[1] = delta_init
    else
        chain_β[1] = [ rand( Uniform(a_interv,b_interv) ) ]
        chain_δ[1] = rand( Uniform())
    end
    δ_map = δ_run_map
    β_map = β_run_map
    if δ_run_map == 0.0
        δ_map = chain_δ[1]
        β_map = chain_β[1]
    end
    loc_V = [ clayton_corm_lognorm_2dim_sampler( chain_δ[1], LogNorm_v) for j in 1:l_mc ]
    println("MCMC run has started.")
    prog = Progress( n, dt=0.5, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    for i in 1:n
        @inbounds lf_pseud_marg(V) = model_loglikelihood( chain_β[i], A, α, V, model)
        @inbounds chain_δ[i+1], loc_V = metr_hast_chain_randwalk_LogNorm_delta_logscale_justend( lf_pseud_marg, chain_δ[i], loc_V, LogNorm_v, l_mc, m, δ_var)
        @inbounds lf_β_nocox(b) =  model_loglikelihood( [ b ], A, α, loc_V, model) + logpdf( Beta(1.7,1.7), 0.5*b ) # logpdf( TruncatedNormal( 1.0, 1.0, 0.0, 2.0), b)
        @inbounds chain_β[i+1][1] =  metr_hast_chain_randwalk_a_b_interv_logscale_justend( lf_β_nocox, m, β_var, a_interv, b_interv, chain_β[i][1])
        M_temp = model_loglikelihood( chain_β[i+1], A, α, loc_V, model) # + logpdf( Beta(1.7,1.7), 0.5*chain_β[i+1][1])
        if M < M_temp
            @inbounds β_map = chain_β[i+1]
            @inbounds δ_map = chain_δ[i+1]
            M = M_temp
        end
        next!(prog)
    end
    return chain_δ, chain_β, δ_map, β_map
end

@inline function surv( t::Float64, z_new::Array{Float64,1}, β::Array{Float64,1}, A::Float64, α::Float64, V::Array{Array{Float64,1},1}, model::Model)
    obs = [0.; model.obs; Inf]
    Hᶜ = [ model.Hᶜ(i,β) for i in 1:length(model.f) ]
    Hᵉ = [ model.Hᵉ(i,β) for i in 1:length(model.f) ]
    H = [ [ model.Hᶜ(i,β)[j] + model.Hᵉ(i,β)[j] for i in 1:length(model.f) ] for j in 1:(length(model.obs) + 1) ]
    H_shift = [ [ model.Hᶜ(i,β)[j] + model.Hᵉ(i,β)[j+1] for i in 1:length(model.f) ] for j in 1:length(model.obs) ]
    f_new = [ model.f[i]( z_new, β) for i in 1:length(model.f) ]
    return ℯ^( - A * mapreduce( j -> ( min( t, obs[j+1]) - obs[j] ) * ( obs[j] <= t ) *
                                 ψ_dif( H[j], f_new, α, V), +, 1:(length(model.obs) + 1), init = 0.0 ) ) *
          mapreduce( j -> e_contrib_func( f_new + H_shift[j], j, β, α, V, model)/
                          e_contrib_func(  H_shift[j], j, β, α, V, model), *, model.e_o_f(t), init = 1.0 )
end

@inline function surv_vec_delta_prev( z_new::Array{Float64,1}, β::Array{Float64,1}, A::Float64, α::Float64, δ::Float64, LogNorm_v::Float64, l_mc::Int64, t_f::Float64, mesh::Float64, model::Model)
    V = [ clayton_corm_lognorm_2dim_sampler( δ, LogNorm_v) for j in 1:l_mc ]
    return @showprogress [ surv( j, z_new, β, A, α, V, model)  for j in collect(0.:1.0/mesh:t_f) ]
end

function mi_find(v::Array{Float64,1},e::Float64)
    index = 1
    while v[index] <= e
        index += 1
    end
    return index-1
end

@inline function surv_vec_delta( z_new::Array{Float64,1}, β::Array{Float64,1}, A::Float64, α::Float64, δ::Float64, LogNorm_v::Float64, l_mc::Int64, t_f::Float64, mesh::Float64, model::Model)
    t_vec = collect(0.:1.0/mesh:t_f)
    s_vec = zeros(Float64,length(t_vec))
    s_vec[1] = 1.0
    V = [ clayton_corm_lognorm_2dim_sampler( δ, LogNorm_v) for j in 1:l_mc ]
    obs = [model.obs; Inf]
    H = [ [ model.Hᶜ(i,β)[j] + model.Hᵉ(i,β)[j] for i in 1:length(model.f) ] for j in 1:(length(model.obs) + 1) ]
    H_shift = [ [ model.Hᶜ(i,β)[j] + model.Hᵉ(i,β)[j+1] for i in 1:length(model.f) ] for j in 1:length(model.obs) ]
    f_new = [ model.f[i]( z_new, β) for i in 1:length(model.f) ]
    obs_ind_run = 1
    exp_part_run = 1.0
    prod_part_run = 1.0
    for j in 2:length(t_vec)
        obs_ind_t = mi_find( obs[(obs_ind_run+1):end], t_vec[j]) + obs_ind_run
        if obs_ind_t == obs_ind_run
            exp_part_run = exp_part_run*exp( - A * ( t_vec[j] - t_vec[j-1] ) * ψ_dif( H[obs_ind_run], f_new, α, V))
        else
            exp_part_run = exp_part_run*exp( - A * ( obs[obs_ind_run+1] - t_vec[j-1] ) * ψ_dif( H[obs_ind_run], f_new, α, V))
            for i in (obs_ind_run+1):(obs_ind_t-1)
                exp_part_run = exp_part_run*exp( - A * (obs[i+1]-obs[i]) * ψ_dif( H[i], f_new, α, V))
                prod_part_run = prod_part_run*e_contrib_func( f_new + H_shift[i], i, β, α, V, model)/e_contrib_func( H_shift[i], i, β, α, V, model)
            end
            obs_ind_run = obs_ind_t
            exp_part_run = exp_part_run*exp( - A * ( t_vec[j] - obs[obs_ind_run] ) * ψ_dif( H[obs_ind_run], f_new, α, V))
            prod_part_run = prod_part_run*e_contrib_func( f_new + H_shift[obs_ind_run], obs_ind_run, β, α, V, model)/e_contrib_func( H_shift[obs_ind_run], obs_ind_run, β, α, V, model)
        end
        s_vec[j] = exp_part_run * prod_part_run
    end
    return s_vec
end

# Kaplan-Meier estiamtor function
function KMest(tiempo, muerte)
    t_KMest = sort( unique( tiempo ) )
    n_KMest = [count( i->(i >= j), tiempo) for j in t_KMest]
    d_KMest = [sum( muerte[ findall(j->j==i,tiempo ) ] ) for i in t_KMest]
    s_KMest = cumprod( 1 .- d_KMest./n_KMest )
    return t_KMest, s_KMest
end

# Kaplan-Meier estiamtor function
function make_plot_vec( t, s)
    if t[1] != 0.0
        t = [ 0.0; t ]
        s = [ 1.0; s ]
    end
    t_plot = [ t[1] ]
    s_plot = [ s[1] ]
    for i in 2:( length(t))
        push!( t_plot,  t[i], t[i] )
        push!( s_plot,  s[i-1], s[i] )
    end
    return t_plot, s_plot
end

function posterior( x::Array{Float64}, A::Float64, α::Float64, model::Model, LogNorm_v::Float64=0.1, l_mc::Int64=1000)
    V = [ clayton_corm_lognorm_2dim_sampler( x[1], LogNorm_v) for j in 1:l_mc ]
    return model_loglikelihood( x[2:2], A, α, V, model)
end

function cred_band_mat(S::Array{Float64,2},n::Int64)
    for j in 1:n
        b = zeros(Float64, size(S)[2])
        for i in 1:251
         b[i] = maximum(S[:,i])-minimum(S[:,i])
        end
        b_max_index = sortperm(b)[end]
        s_out_index = findmax( abs.( S[:,b_max_index] .- mean(S[b_max_index]) ) )[2]
        S = S[1:end .!= s_out_index, 1:end]
    end
    return S
end
