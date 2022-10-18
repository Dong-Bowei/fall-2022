#Worked with Collin DeVore
function wrapper()
    #using SMM
    using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV 

    #1. 
    url="https://raw.githubusercontent.com/OU-PHD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df_2 = CSV.read(HTTP.get(url).body, DataFrame)
    X=[ones(size(df_2,1),1) df_2.age df_2.race.==1 df_2.collgrad.==1]
    y=df_2.married.==1

    function logit_gmm(α, X, y)
        P = exp.(X*α) ./ (1 .+ exp.(X*α))
        g = y .- P
        J = g' * I * g
        return J
    end

    α_optim = optimize(a -> logit_gmm(a, X, y), rand(size(X, 2)), LBFGS(), Optim.Options(g_tol = 1e-8, iterations = 100_000))
    println(α_optim)

    #2.
    #a. Maximum Likelihood. 
    #Cited from Dr. Ransom (2022)
    using FreqTables

    freqtable(df_2, :occupation) 
    df_2 = dropmissing(df_2, :occupation)
    df_2[df_2.occupation.==8 ,:occupation] .= 7
    df_2[df_2.occupation.==9 ,:occupation] .= 7
    df_2[df_2.occupation.==10,:occupation] .= 7
    df_2[df_2.occupation.==11,:occupation] .= 7
    df_2[df_2.occupation.==12,:occupation] .= 7
    df_2[df_2.occupation.==13,:occupation] .= 7
    freqtable(df_2, :occupation) 

    X = [ones(size(df_2,1),1) df_2.age df_2.race.==1 df_2.collgrad.==1]
    y = df_2.occupation

    function mlogit(alpha, X, y)
        
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

    alpha_zero = zeros(6*size(X,2))
    alpha_rand = rand(6*size(X,2))
    alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
    alpha_start = alpha_true.*rand(size(alpha_true))
    println(size(alpha_true))
    alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println(alpha_hat_mle)

    #b. GMM with the MLE estimates as starting values.
    #Worked with Collin DeVore
    #Cited from Dr. Ransom (2022)
    
    function gmmmle(α, X, y)
        N = 2237 #number of rows of matrix X
        J = 7 #number of occupations
        d = zeros(N,J) #specifying d
        #d_stack = d[:]
        K = 4 #number of colums of matrix X
        bigAlpha = [reshape(α, K, J-1) zeros(K)]
        
        num = zeros(N, J)
        dem = zeros(N)

        for j = 1:J
            d[:, j] = y .== j
            num[:, j] = exp.(X * bigAlpha[:, j])
            dem .+= num[:, j]
        end 
        
        P = num ./ repeat(dem, 1, J) #specifying P

        #GMM estimates
        #cited from lecture slides
        g = d[:] .- P[:]
        J = g' * I * g

        return J
    end

    start_vals = alpha_hat_optim.minimizer
    alpha_hat_gmm = optimize(a -> mlogit(a, X, y), start_vals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_gmm_1 = alpha_hat_gmm.minimizer
    println(alpha_hat_gmm_1)

    #c. GMM with random starting values
    alpha_rand = rand(6*size(X,2))
    alpha_hat_gmm_rand = optimize(a -> mlogit(a, X, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_gmm_rand_1 = alpha_hat_gmm_rand.minimizer
    println(alpha_hat_gmm_rand_1)
    #From (b) we get:
    #[0.1910168847231602, -0.03352610275086075, 0.5963966905947069, 0.41650526984995023, -0.1698627202270262, -0.035977768704916945, 1.3068422942718465, -0.4309980299164959, 0.6894704734271244, -0.010457782918224218, 0.5231633851739385, -1.4924747793367363, -2.2675130986388985, -0.0052992961250457405, 1.3914055143869726, -0.9849688237806108, -1.3984219975204124, -0.014298116825487239, -0.017654617929369138, -1.495117265888482, 0.24547998756864106, -0.0067264852714428724, -0.5382893935838678, -3.789781637951679]
    #From (c), the random process, we get:
    #[0.19105783354229722, -0.033527128143727214, 0.5963957579409531, 0.4165057971221914, -0.16979943706221248, -0.0359793655673747, 1.306841129518494, -0.43099645813149723, 0.6895024439032883, -0.01045858857115259, 0.5231630431237818, -1.4924749638986639, -2.267440973802294, -0.005301080156369734, 1.391402633874976, -0.9849646737156312, -1.3983824389653763, -0.014299115623115932, -0.017654639127669982, -1.4951193550075967, 0.24549004601571317, -0.0067267512299609275, -0.5382887347575018, -3.789778539943963]
    #As we can see from (b) and (c) above, the results are clearly close to each other so the minimizers might be the same, so this means the objective function is globally concave.

    #3. I am not sure I am on the right track about this one
    using Distributions, Random
    function mullogit(N, J, K)
        
        J = 3
        K = 2
        N = 3
        β = [0.1 0.2 0.3; 0.4 0.5 0.6] #β is a K by J matrix
        X = rand(N, K) #randomly generate a matrix of N by K
        P = exp.(X * β) ./ sum.(eachrow(X * β))
        ϵ = rand(Uniform(0,1), N, 1)

        y = zeros(N, 1)
        #Collin DeVore taught me why&how to set up the bins
        for i = 1:3
            if P[i,1] > ϵ[i,1]
                y[i] = 1
            elseif P[i,2] > ϵ[i,2] > P[i,1]
                y[i] = 2
            elseif ϵ[i,3] < 1
                y[i] = 3
            end
        end

        #Y = sum. Y[i] .* 1
    
        return(y)
    end
            
    #5.Cited from lecture slides by Dr. Ransom (2022)
    #I did not finish this question
    function ols_smm(θ, X, y, D)
        
        K = size(X,2)
        N = size(y,1)
        
        β = θ[1:end-1]
        if length(β)==1
            β = β[1]
        end

        P = exp.(X * β) ./ sum.(eachrow(X * β))
       
        ε = randn(N)
        y_zero = zeros(N)
        gmodel = zeros(N+1,D)
        gdata  = vcat(y,var(y))
      
        Random.seed!(1234) 
        
        for d in 1:D
            for i in 1:N
                if P[i,1] > ϵ[i,1]
                    y[i] = 1
                elseif P[i,2] > ϵ[i,2] > P[i,1]
                    y[i] = 2
                elseif ϵ[i,3] < 1
                    y[i] = 3
                end
            end
        end

        err = vec(gdata .- mean(gmodel; dims=2))
        J = err'*I*err

        return J
    end

    smm_rand = rand(6*size(X,2))
    alpha_hat_smm_rand = optimize(a -> ols_smm(a, X, y, D), smm_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_smm_rand_1 = alpha_hat_smm_rand.minimizer
    println(alpha_hat_smm_rand_1)


wrapper()








