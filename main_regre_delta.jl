# Load GANTR code
include( "gantr.jl" )
# Load simulated data generation code
include( "binary_sampler.jl" )

using JLD2, LaTeXStrings

compriskDelta0pt1_p = sim_study_sampler_comprisk( 200, 0.1, 1.0, 2.2, 1.75, 1.2, 2.2, 5.3, 1.5)

# Save and load simulated data
#@save "compriskDelta0pt1_p.txt" compriskDelta0pt1_p
#@load "compriskDelta0pt1_p.txt"

# Define surial times and covariables
T, Z = compriskDelta0pt1

# No censoring
C = repeat( [1], inner=length(T))
Z_ar = [ [Z[i]] for i in 1:length(Z) ]

# Define data struct
data = Data( T, C, Z_ar )

# Define regression functions
function f_1( z::Array{Float64,1}, β::Array{Float64,1})
    return 1.0*(z[1] <= β[1])  # male white
end

function f_2( z::Array{Float64,1}, β::Array{Float64,1})
    return 1.0*(z[1] > β[1])  # male white
end

# Define model struct
model =  Model( data, 1., [ f_1, f_2 ])

# Precompile loglikelihood
model_loglikelihood( [rand(Uniform())], 1.0, 1.0, [ clayton_corm_lognorm_2dim_sampler( 0.2, 0.1) for j in 1:1000 ], model)

# Precompile MCMC
MCMC_comprisk_uniform( 3, 3, 3, 1.0, 1.0, model, 0.1, [1.0], 0.5, 0.0, 2.0)
# Run MCMC
run_uniform = MCMC_comprisk_uniform( 5000, 1, 1000, 1.0, 1.0, model, 0.1, [1.0], 0.5, quantile(Z,0.25), quantile(Z,0.75))

# Save and load MCMC run
#@save "run_uniform.txt" run_uniform
#@load "run_uniform.txt"

# Precompile MCMC
MCMC_comprisk_randwalk( 3, 1, 3, 1.0, 1.0, model, 0.1, 0.1, 0.1, [1.0], 0.5, quantile(Z)[2], quantile(Z)[4] )
# Run MCMC
run_rw = MCMC_comprisk_randwalk( 10000, 1, 1000, 1.0, 1.0, model, 0.1, 0.2, 0.1, [0.7], 0.5, quantile(Z)[2], quantile(Z)[4] )

# Save and load MCMC run
#@save "run_rw.txt" run_rw
#@load "run_rw.txt"

surv_vec_delta( [ 1. ], [1.], 1., 1., 1., 0.1, 3, 10., 2., model)

# surv_pop_1_map_unif = surv_vec_delta( [ 0.66 ], run_uniform[4], 1., 1., run_uniform[3], 0.1, 1000, 5., 50., model)
# surv_pop_2_map_unif = surv_vec_delta( [ 1.44 ], run_uniform[4], 1., 1., run_uniform[3], 0.1, 1000, 5., 50., model)
# surv_pop_1_mean_unif = surv_vec_delta( [ 0.66 ],  mean( run_uniform[2][500:5001] ), 1., 1., mean( run_uniform[1][500:5001] ), 0.1, 1000, 5., 50., model)
# surv_pop_2_mean_unif = surv_vec_delta( [ 1.44 ], mean( run_uniform_[2][500:5001] ), 1., 1., mean( run_uniform[1][500:5001] ), 0.1, 1000, 5., 50., model)

surv_pop_1_map_rw = surv_vec_delta( [ 0.66 ], run_rw_p[4], 1., 1., run_rw_p[3], 0.1, 1000, 5., 50., model)
surv_pop_2_map_rw = surv_vec_delta( [ 1.44 ], run_rw_p[4], 1., 1., run_rw_p[3], 0.1, 1000, 5., 50., model)
surv_pop_1_mean_rw = surv_vec_delta( [ 0.66 ],  mean( run_rw_p[2][500:5001] ), 1., 1., mean( run_rw_p[1][500:5001] ), 0.1, 1000, 5., 50., model)
surv_pop_2_mean_rw = surv_vec_delta( [ 1.44 ], mean( run_rw_p[2][500:5001] ), 1., 1., mean( run_rw_p[1][500:5001] ), 0.1, 1000, 5., 50., model)

using PyPlot

S_pop1 = zeros(Float64,100,251)
for i in 1:100
 S_pop1[i,:] = surv_vec_delta( [ 0.66 ],  run_rw_p[2][1001+90*i], 1., 1., run_rw_p[1][1001+90*i], 0.1, 1000, 5., 50., model)
end

S_pop2 = zeros(Float64,100,251)
for i in 1:100
 S_pop2[i,:] = surv_vec_delta( [ 1.44 ],  run_rw_p[2][1001+90*i], 1., 1., run_rw_p[1][1001+90*i], 0.1, 1000, 5., 50., model)
end

S2_pop1 = cred_band_mat(S_pop1,5)
S2_pop2 = cred_band_mat(S_pop2,5)

surv_pop_1_band_rw_u = [ findmax(S2_pop1[:,i])[1] for i in 1:251]
surv_pop_1_band_rw_d = [ findmin(S2_pop1[:,i])[1] for i in 1:251]
surv_pop_2_band_rw_u = [ findmax(S2_pop2[:,i])[1] for i in 1:251]
surv_pop_2_band_rw_d = [ findmin(S2_pop2[:,i])[1] for i in 1:251]

################################################################################

using Optim

x0 = rand(Uniform(),2)
lower = [ 0.00001, 0.00001 ]
upper = [ 0.99999, 1.99999 ]

f_opt(x::Array{Float64,1}) = - posterior(x,1.0,1.0,model,0.1,1000)

op = optimize( f_opt, lower, upper, x0, Fminbox(LBFGS()), Optim.Options(iterations = 20) )

op.minimizer

# Save and load op
#@save "op.txt" op
#@load "op.txt"
