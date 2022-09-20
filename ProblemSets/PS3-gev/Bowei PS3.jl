#PS 3, worked with Collin DeVore and William Meyers
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
#1. 
function Problem_Set_3()

url="https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"
data_get=CSV.read(HTTP.get(url).body, DataFrame)
X=[data_get.age data_get.white data_get.collgrad]
Z=hcat(data_get.elnwage1, data_get.elnwage2,data_get.elnwage3,data_get.elnwage4,data_get.elnwage5,data_get.elnwage6,data_get.elnwage7,data_get.elnwage8)
y=data_get.occupation

function mlogit(alpha_1, X, y)
        
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    alpha = alpha_1[1:end-1]
    gamma = alpha_1[end]
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        bigY[:,j] = y.==j
        diff_Z=Z[:,j]-Z[:,J]
        num[:,j] = exp.((X*bigAlpha[:,j]) .+ (gamma * diff_Z) )
        dem .+= num[:,j]
    end

    P = num ./ repeat(dem,1,J)
    loglike = -sum( bigY .* log.(P) )
        
    return loglike
end

    alpha_zero = zeros(7 * size(X,2)+1)
    alpha_start = rand(7 * size(X,2)+1)
    alpha_hat_optim = optimize(alpha_1 -> mlogit(alpha_1, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100000, show_trace=true, show_every=50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println(alpha_hat_mle)

#2. Interpret the estimated coefficient γ
# γ is the coefficient, it is the impact of the wages on the probability of an occupation.

#3. 
X_wc = [data_get.elnwage1 data_get.elnwage2 data_get.elnwage3]
X_bc = [data_get.elnwage4 data_get.elnwage5 data_get.elnwage6 data_get.elnwage7]
X_ot = [data_get.elnwage8]

function nmlogit(alpha_new, X, y)
    
    beta_wc = alpha_new[1:3]
    beta_bc = alpha_new[4:7]
    beta_ot = 0
    lambda_wc = alpha_new[7]
    lambda_bc = alpha_new[8]
    gamma_new = alpha_new[9]

#nested logit
    num_new_wc = (exp(X_wc .* beta_wc .+ gamma_new .* diff_Z) ./ lambda_wc)
    num_new_wc_1 = (num_new_wc[:,j]) .^ (lambda_wc .- 1)
    num_new_wc_2 = num_new_wc .* num_new_wc_1

    num_new_bc = (exp(X_bc .* beta_bc .+ gamma_new .* diff_Z) ./ lambda_bc)
    num_new_bc_1 = (num_new_bc[:,j]) .^ (lambda_bc .- 1)
    num_new_bc_2 = num_new_bc .* num_new_bc_1

    num_new_ot = (exp(gamma_new .* diff_Z) ./ lambda_ot)
    num_new_ot_1 = (num_new_ot[:,j]) .^ (lambda_ot .- 1)
    num_new_ot_2 = num_new_ot .* num_new_ot_1

    dem_new = 1 .+ (exp(((data_get.elnwage1 .* alpha_wc) .+ (gamma_new .* diff_Z)) ./ lambda_wc) .+ exp(((data_get.elnwage2 .* alpha_wc) .+ (gamma_new .* diff_Z)) ./ lambda_wc) .+ exp(((data_get.elnwage3 .* alpha_wc) 
    .+ (gamma_new .* diff_Z)) ./ lambda_wc)) .^ (lambda_wc) .+ (exp(((data_get.elnwage4 .* alpha_bc) .+ (gamma_new .* diff_Z)) ./ lambda_bc) .+ exp(((data_get.elnwage5 .* alpha_bc) .+ (gamma_new .* diff_Z)) ./ lambda_bc) 
    .+ exp(((data_get.elnwage6 .* alpha_bc) .+ (gamma_new .* diff_Z)) ./ lambda_bc) .+ exp(((data_get.elnwage7 .* alpha_bc) .+ (gamma_new .* diff_Z)) ./ lambda_bc)) .^ (lambda_bc) 
    #.+ exp((data_get.elnwage8 .* alpha_ot) 
    .+ exp(((gamma_new .* diff_Z) ./ lambda_ot)) .^ lambda_ot
# j ∈ WC
    P_prof = num_new_wc_2 ./ dem_new
    P_mang = num_new_wc_2 ./ dem_new
    P_sale = num_new_wc_2 ./ dem_new
# j ∈ BC
    P_unsk = num_new_bc_2 ./ dem_new
    P_craf = num_new_bc_2 ./ dem_new
    P_oprt = num_new_bc_2 ./ dem_new
    P_tran = num_new_bc_2 ./ dem_new
# j ∈ Others
    P_othe = num_new_ot_2 ./ dem_new

    P = [P_prof P_mang P_sale P_unsk P_craf P_oprt P_tran P_othe]
    log_like = sum(y .* log.(P) .+ (1 .- y) .* log.(1 .- P)) 
    return
    log_like
    paramt = vcat[beta_wc beta_bc beta_ot lambda_wc lambda_bc lambda_ot]

    end

alpha_hat_optim = optimize(alpha_new -> mlogit(alpha_new, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(alpha_hat_optim.minimizer)

end

Problem_Set_3()
