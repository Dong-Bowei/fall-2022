
#worked with Collin DeVore

function Problem_Set_5()

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, DataFramesMeta
cd("/Users/boweidong/Dropbox/My Mac (Boweis-MacBook-Air.local)/Desktop")

#1.
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
include("create_grids.jl")
df = @transform(df, bus_id = 1:size(df,1))
df_y = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
df_y_long = DataFrames.stack(df_y, Not([:bus_id,:RouteUsage,:Branded]))
rename!(df_y_long, :value => :Y)
df_y_long = @transform(df_y_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(df_y_long, Not(:variable))
#reshape the odometer variable
df_x = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
df_x_long = DataFrames.stack(df_x, Not([:bus_id]))
rename!(df_x_long, :value => :Odometer)
df_x_long = @transform(df_x_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(df_x_long, Not(:variable))
# create df_long
df_long = leftjoin(df_y_long, df_x_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])

#2. (worked with Collin DeVore, cited from https://juliastats.org/GLM.jl/v0.11/)
Y_2 = @formula(Y ~ Odometer + Branded)
Y_2_lg = glm(Y_2, df_long,  Binomial(), ProbitLink())

#3. 
#(a).
url_1 = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
data_get = CSV.read(HTTP.get(url_1).body, DataFrame)
Y = Matrix(data_get[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
X_odo = Matrix(data_get[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
X_st = Matrix(data_get[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])

#(b).
zval, zbin, xval, xbin, xtran = create_grids()

#(c). Compute the future value term
@views @inbounds function optimstop(theta, Xst, Zst)

#create a loop
    F_value = zeros(length(xtran, 1), 2, T .+ 1, dims = 3)
    for t in T: -1 : 1
        for i in 0 : 1
            for z in 1 : zbin
                for x in 1 : xbin
                    obj = 1 .+ (z - 1) .* xbin
                    ut = theta[1] .+ theta[2] .* Xst .+ theta[3] .* b[i]
                    beta = 0.9 #Collin helped me went throught this question
                    int = Xtran[1 + (z .- 1)]'* F_value[(z - 1) * xbin + 1 : z * xbin, b + 1, t + 1]
                    F_value = beta .* (log.exp((ut)) .+ exp(int))
                end
            end
        end
    end                

    
#(d). Construct the log likelihood
   for i in 0 : 1
       for t in 1 : 2001
          loglik = 0
          r = 1 .+ (z .- 1) .* xbin
          r_1 = Xst .+ (z .- 1) .* xbin
          utd = theta[1] .+ theta[2] .* Xst .+ theta[3] .* b[i]
          int = Xtran[1 + (z .- 1)]'* F_value[(z - 1) * xbin + 1 : z * xbin, b + 1, t + 1]
          int_new = (xtran[r_1:] .- xtran[r,:])' * F_value[r: r + xbin - 1, b[i] + 1, t + 1]
          v_ot = int 
          v_it = utd .+ int_new
          P = (exp.(int_new .- int)) ./ (1 .+ exp.(int_new .- int))
          P_1 = 1 .- P
          loglik_new = -(((Y[i,t] == 1) .* log.(P)) .- ((Y[i, t] == 0) .* log.(P_1)))
          return loglik_new
        end
    end
end

        

theta_new = zeros(2 .* length(xtran, 1))
theta_new_1 = zeros(3 .* size(Xst, 2) .+ 1)
theta_opt = optimize(theta -> optimstop(theta), theta_new, LBFGS(), Optim.Options(g_tol = 1e - 5, iterations = 100000, show_trace = true, show_every = 50))
theta_hat_mle = theta_new_1.minimizer
println(theta_hat_mle)

optimstop(theta, Xst, Zst)

#(e)Wrap up in a function
#(f)
@views @inbounds function myfun()

    #create a loop
    zval, zbin, xval, xbin, xtran = create_grids()

    @views @inbounds function optimstop(theta, Xst, Zst)
    
    #create a loop
        F_value = zeros(length(xtran, 1), 2, T .+ 1, dims = 3)
        for t in T: -1 : 1
            for i in 0 : 1
                for z in 1 : zbin
                    for x in 1 : xbin
                        obj = 1 .+ (z - 1) .* xbin
                        ut = theta[1] .+ theta[2] .* Xst .+ theta[3] .* b[i]
                        beta = 0.9 #Collin helped me went throught this question
                        int = Xtran[1 + (z .- 1)]'* F_value[(z - 1) * xbin + 1 : z * xbin, b + 1, t + 1]
                        F_value = beta .* (log.exp((ut)) .+ exp(int))
                    end
                end
            end
        end                
    
        

       for i in 0 : 1
           for t in 1 : 2001
              loglik = 0
              r = 1 .+ (z .- 1) .* xbin
              r_1 = Xst .+ (z .- 1) .* xbin
              utd = theta[1] .+ theta[2] .* Xst .+ theta[3] .* b[i]
              int = Xtran[1 + (z .- 1)]'* F_value[(z - 1) * xbin + 1 : z * xbin, b + 1, t + 1]
              int_new = (xtran[r_1:] .- xtran[r,:])' * F_value[r: r + xbin - 1, b[i] + 1, t + 1]
              v_ot = int 
              v_it = utd .+ int_new
              P = (exp.(int_new .- int)) ./ (1 .+ exp.(int_new .- int))
              P_1 = 1 .- P
              loglik_new = -(((Y[i,t] == 1) .* log.(P)) .- ((Y[i, t] == 0) .* log.(P_1)))
              return loglik_new
            end
        end
    end
    
            
    
    theta_new = zeros(2 .* length(xtran, 1))
    theta_new_1 = zeros(3 .* size(Xst, 2) .+ 1)
    theta_opt = optimize(theta -> optimstop(theta), theta_new, LBFGS(), Optim.Options(g_tol = 1e - 5, iterations = 100000, show_trace = true, show_every = 50))
    theta_hat_mle = theta_new_1.minimizer
    println(theta_hat_mle)
    
    optimstop(theta, Xst, Zst)
            myfun()

#(g)Wrap up
#(h)
Problem_Set_5()

#(i). 
#Yes I ordered peach tea

