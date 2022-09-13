#PS2. worked with Collin DeVore and William Meyers
#add Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
function Problem_set_2()

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
#1. Basic optimization in Julia.
f(x)=-x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
negf(x)=x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval=rand(1)
result=optimize(negf, startval, LBFGS())

#2. 
url="https://raw.githubusercontent.com/OU-PHD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df_2 = CSV.read(HTTP.get(url).body, DataFrame)
X=[ones(size(df_2,1),1) df_2.age df.race.==1 df_2.collgrad.==1]
y=df_2.married.==1

function ols(beta, X, y)
    SSR=(y.-X*beta)'*(y.-X*beta)
    return SSR
end

beta_hat_ols=optimize(b->ols(b,X,y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000,show_trace=true))
println(beta_hat_ols.minimizer)

bols=inv(X'*X)*X'*y
df_2.white=df_2.race.==1
bols_lm=lm(@formula(married~age+white+collgrad), df_2)

#3.
function loglike(beta_1, X,y)
    P_1=exp.(X*beta_1) ./ (1 .+ exp.(X*beta_1))
    P_2=1 ./ (1 .+ exp.(X*beta_1))
    loglh=sum(y.*log.(P_1) .+ (1 .-y).*log.(P_2))
    return loglh 
end

beta_hat_log=optimize(beta_1->loglike(beta_1,X,y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000,show_trace=true))
println(beta_hat_log.minimizer)


#4. 
fm=@formula(married~age+white+collgrad)
logit=glm(fm, df,Binomial(), LogitLink())

#5.
freqtable(df_2, :occupation)
df_2=dropmissing(df_2, :occupation)
df_2[df_2.occupation.==8, :occupation].=7
df_2[df_2.occupation.==9, :occupation].=7
df_2[df_2.occupation.==10, :occupation].=7
df_2[df_2.occupation.==11, :occupation].=7
df_2[df_2.occupation.==12, :occupation].=7
df_2[df_2.occupation.==13, :occupation].=7
freqtable(df_2, :occupation)

X_2=[ones(size(df_2,1),1) df_2.age df_2.race.==1 df_2.collgrad.==1]
y_2=df_2.occupation

function    multino(beta, X, y)
    beta_1 = reshape(beta[1:4,1], (4,1))    
    beta_2 = reshape(beta[5:8,1], (4,1))
    beta_3 = reshape(beta[9:12,1], (4,1))
    beta_4 = reshape(beta[13:16,1], (4,1))
    beta_5 = reshape(beta[17:20,1], (4,1))
    beta_6 = reshape(beta[21:24,1], (4,1))
    
    den = sum(1 .+ exp.(X * beta_1) .+ exp.(X * beta_2).+ exp.(X * beta_3).+ exp.(X * beta_4).+ exp.(X * beta_5).+ exp.(X * beta_6))
    
    mul_1 = exp.(X * beta_1) ./ den
    mul_2 = exp.(X * beta_2) ./ den
    mul_3 = exp.(X * beta_3) ./ den
    mul_4 = exp.(X * beta_4) ./ den
    mul_5 = exp.(X * beta_5) ./ den
    mul_6 = exp.(X * beta_6) ./ den

    mul = zeros(size(X_2,1),6)
    for i in 1:size(mul,1)
        if y[i,1] == 1
            mul[i,1] = 1
        elseif y[i,1] == 2
            mul[i,2] = 1
        elseif y[i,1] == 3
            mul[i,3] = 1
        elseif y[i,1] == 4
            mul[i,4] = 1
        elseif y[i,1] == 5
            mul[i,5] = 1
        else
            mul[i,6] = 1
        end
    end
    multino_log = sum((mul[:,1] .* log.(mul_1)) .+ (mul[:,2] .* log.(mul_2)) .+ (mul[:,3] .* log.(mul_3))  .+ (mul[:,4] .* log.(mul_4))  .+ (mul[:,5] .* log.(mul_5))  .+ (mul[:,6] .* log.(mul_6)) )
    return multino_log
end

beta_hat_log_1 = optimize(beta -> -multino(beta, X_2, y_2), zeros(24,1), LBFGS(), Optim.Options(g_tol = 1e-6, iterations = 100000, show_trace = true))
println(beta_hat_log_1.minimizer)

end

Problem_Set_2()