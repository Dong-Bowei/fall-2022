#worked with Collin DeVore, William Meyers

function ProblemSet_4()


    #1. 
    #add ForwardDiff

    using Random
    using LinearAlgebra
    using Statistics
    using Optim
    using DataFrames
    using CSV
    using HTTP
    using GLM
    using FreqTables
    using ForwardDiff 


        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # question 1 (Cited from PS 3 answer from Dr. Tyler Ransome, 2022)
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS4-mixture/nlsw88t.csv"
        data_get = CSV.read(HTTP.get(url).body, DataFrame)
        X = [data_get.age data_get.white data_get.collgrad]
        Z = hcat(data_get.elnwage1, data_get.elnwage2, data_get.elnwage3, data_get.elnwage4, 
        data_get.elnwage5, data_get.elnwage6, data_get.elnwage7, data_get.elnwage8)
        y = data_get.occ_code
        
        function mlogit(alpha_2, X, Z, y)
            
            alpha = alpha_2[1:end-1]
            gamma = alpha_2[end]
            K = size(X,2)
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N,J)
            for j=1:J
                bigY[:,j] = y.==j
            end
            bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
            
            T = promote_type(eltype(X),eltype(alpha_2))
            num   = zeros(T,N,J)
            dem   = zeros(T,N)
            for j=1:J
                num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
                dem .+= num[:,j]
            end
            
            P = num./repeat(dem,1,J)
            
            loglike = -sum( bigY.*log.(P) )
            
            return loglike
        end
        
        alpha_zero = zeros(7 * size(X,2)+1)
        alpha_start = rand(7 * size(X,2)+1)
        alpha_hat_optim = optimize(alpha_2 -> mlogit(alpha_2, X, Z, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100000, show_trace=true, show_every=50))
        alpha_hat_mle = alpha_hat_optim.minimizer
        println(alpha_hat_mle)
        #use automatic differentiation to speed up the estimation
        startvals = [2*rand(7*size(X,2)).-1; .1]
        td = TwiceDifferentiable(alpha_2 -> mlogit(alpha_2, X, Z, y), startvals; autodiff = :forward)
        alpha_hat_optim_1 = optimize(alpha_2 -> mlogit(alpha_2, X, Z, y), startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100000, show_trace=true, show_every=50))
        alpha_hat_mle_1 = alpha_hat_optim_1.minimizer
        println(alpha_hat_mle_1)

        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # question 2 worked with Collin DeVore
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        #The eatimated coefficient γ does not make more sense now than in problem set 3
        
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # question 3 worked with Collin DeVore
        #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        #(a)
        using Distributions
        include("lgwt.jl")
        d = Normal(0,1)
        nodes, weights = lgwt(7,-4,4)
        sum(weights.*pdf.(d,nodes))
        sum(weights.*nodes.*pdf.(d,nodes))
        
        #(b)
        d = Normal(0, 2)
        nodes, weights = lgwt(7, -4, 4)
        sum(weights .* nodes.^2 .* pdf.(d, nodes))          
        nodes, weights = lgwt(10, -4, 4)
        sum(weights .* nodes.^2 .* pdf.(d, nodes))   
        
        #(c)
        #(d)
       
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        # question 4 worked with Collin DeVore, William Meyersm Collin taught me
        #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

        function mlogit(alpha_3, X, Z, y)
            
            alpha = alpha_3[1:end-1]
            gamma = alpha_3[end]
            K = size(X,2)
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N,J)
            
           #I can only finish the parts above

ProblemSet_4()

