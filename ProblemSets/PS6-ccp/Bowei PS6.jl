#worked with Collin DeVore
#1.

function  wrapper()
    
    using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, DataFramesMeta

    cd("/Users/boweidong/Dropbox/My Mac (Boweis-MacBook-Air.local)/Desktop")
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
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
    
    #2. 
    logit = @formula(Y ~ Odometer * Odometer^2 * RouteUsage * RouteUsage^2 * Branded * time * time^2) 
    lg_glm = glm(logit, df_long,  Binomial(), LogitLink())
    #3. cited frojm Dr. Tylor Ransom
    #a.
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body,DataFrame)
    Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
    X = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    Z = Vector(df[:,:RouteUsage])
    B = Vector(df[:,:Branded])
    N = size(Y,1)
    T = size(Y,2)
    Xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Zstate = Vector(df[:,:Zst])
    zval,zbin,xval,xbin,xtran = create_grids()
    
    data_parms = (β = 0.9,
              Y = Y,
              B = B,
              N = N,
              T = T,
              X = X,
              Z = Z,
              Zstate = Zstate,
              Xstate = Xstate,
              xtran = xtran,
              zbin = zbin,
              xbin = xbin,
              xval = xval)
              
              #b. Compute the future value terms

              odo = kron(ones(zbin),xval)
              rout = kron(ones(xbin),zval)
              brad = zeros(size(odo))
              tim = zeros(size(brad))

              function  3_b(Xstate, Zstate, xtran, θ, d)

                FV = zeros(length(xtran,1), 2, T + 1, dims = 3)
                
                for t = 2:T
                    for b = 0:1
                        for z = 1:d.zbin
                            for x=1:d.xbin
                                tim = t 
                                brad = b[i]
                                p_0 = predict()
                                FV[length(xtran,1),b + 1, t] = d.β .* log(p_0)
                            end
                        end
                    end
                end

                for i =1:d.N
                    row_0 = (d.Zstate[i] - 1) * d.xbin + 1
                    for t=1:d.T
                        row1  = d.Xstate[i,t] + (d.Zstate[i]-1)*d.xbin
                        FVT1[i:t] = (xtran[row_1, :] .- xtran[row_0, ; ])' * FV[row_0:row_0 + xbin - 1, B[i] + 1, t + 1]
                    end
                end

                #c. 
                df_long = @transform(df_long, fv = FVT1)
                theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded), df_long, binomial(), LogitLink(), offset = df_long.fv)

                #d. 
                lg_glm_1 = glm(logit, df_long,  Binomial(), LogitLink())


wrapper()