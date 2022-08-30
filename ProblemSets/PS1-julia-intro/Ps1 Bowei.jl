#worked with Collin DeVore and William Meyers
#add JLD2
#add Random
#add LinearAlgebra
#add Statistics
#add CSV
#add DataFrames
#add FreqTables
#add Distributions
#0. GitHub setup#
#1. Initializing variables and practice with basic matrix operations#
#(a)# create the following 4 matrices of random numbers, setting the seed to be '1234'. Name the matrices and set dimentions as noted
using Random, JLD2, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions, JLD
function q1()
    Random.seed!(1234)
#i. A_10*7-random numbers distributed U[-5,10]   
    A=rand(Uniform(-5,10),10,7)
#ii. B_10*7-random numbers distributed N(-2,15) [st dev is 15] 
    B=rand(Normal(-2,15),10,7)
#iii. C_5*7-the first 5 rows and 5 cols of A and the last 2 cols and first 5 rows of B
    C=hcat(A[1:5,1:5],B[1:5,6:7])
#iv. D_10*7-where D_i,j=A_i,j if A_i,j <=0, or 0 otherwise#  
    D=A.*collect(A.<0)
#(b). Use a built-in Julia function to list the number of elements of A 
    println(A)
#(c). Use a series of built-in Julia functions to list the numbers of unique elemens of A
    unique(D)
#(d). Using the reshape() function, create a new matrix called E which is the the 'vec' operator^2 applied to B. Can you find an easier way to accomplish this?#
    E=reshape(B, (70,1))
    E=vec(B) #is the easier way to accomplish this
#(e). Create a new array called F which is 3-dimensional and contains A in the first col of the third dimension and B in the second col of the third dimension.
    F=cat([A;;;B;;;],dims=3)
#(f). Use the permutedims() function to twist F so that it is now F_2*10*7 instead of F_10*7*2. Save this new matrix as F.
    F=permutedims(F, [3,1,2])
    size(F)
#(g). create a matrix G which is equal to B*C (the Kronecker product of B and C). What happens when you try C*F?
    G=kron(B,C)
    H=kron(C,F) #ERROR: MethodError: no method matching kron(::Matrix{Float64}, ::Array{Float64, 3})
#(h). Save the metrices A, B, C, D, E, F and G as a .jld file named matrixpractice
    #add JLD
    save("matrixpractice.jld"; A,B,C,D,E,F,G)
#(i). Save only the matrices A,B,C and D as a .jld file called firstmatrix.
    save("firstmatrix.jld"; A,B,C,D)
#(j). export C as a .csv file called Cmatrix. you will first need to transform C into a DataFrame.
    Cmatrix=DataFrame(C, :auto)
    CSV.write("Cmatrix.csv", Cmatrix)
#(k). Export D as a tab-delimited .dat file called Dmatrix. You will first need to transform D into a DataFrame.
    Dmatrix=DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", Dmatrix)
#(l). Wrap a function definition around all the code for question 1. Call the function q1(). The function should have 0 inputs and should output the arrays A,B,C and D. At the very bottom of your script you should add the code A,B,C,D = q1()
    return A,B,C,D
end
A,B,C,D=q1()



#2. Practice with loops and comprehensions
#(a). Write a loop or use a comprehensive that computes the element-by-element product of A and B. Name the new matrix AB. Create a matrix called AB2 that accomplishes this task without a loop or comprehensions
using LinearAlgebra, Statistics, Distributions
function q2()
    random.seed!(1234)
    A=rand(Uniform(-5,10),10,7)
    B=rand(Normal(-2,15),10,7)
    for i in 1:size(A,1)
        for j in 1:size(A,2)
            AB[i,j]=A[i,j]*B[i,j]
        end
    end

#nonloop method
    AB=.*(A,B)
#(b). Write a loop that creates a col vector called Cprime which contains only the elements of C that are between -5 and 5(inclusive). Create a vector called Cprime2 whihch does this calculation without a loop.
    a=C[:]
    b=[(5>=x>=-5) for x in a]
    vec(b)
    c=b.*a 
    for i in c
        Cprime=filter!(i->i!=0,c)
        println(Cprime)
    end
    vec(c)

     Cprime2=filter(x-> 5>=x>=-5,c)
     vec(Cprime2)
#(c). Using loops or comprehensions, create a 2-dimensional array called X that is of dimension N*K*T where N=15,169, K=6 and T=5. For all t, the cols of X should be (in order):
#an intercept(i.e. vector of ones)
#a dummy variable that is 1 with probability .75*(6-t)/5
#a continuous variable distributed normal with mean 15+t-1 and standard deviation 5(t-1)
#a continuous variable distributed normal with mean π(6-t)/3 and standard deviation 1/e 
#a discrete variable distributed "discrete normal" with mean 12 and standard deviation 2.19. (A discrete normal random variable is properly called a binomail parameters n and p where n=20 and p=0.6. Use the following code(after loading Julia's Distributions package) to generate this vector of X: rand(Binomial(20,0.6),N), where N is the length of the vector)
#a discrete variable distributed binomail with n=20 and p=0.5
#add Distributions
#t=1
    C1=ones(15169)
    #C2=rand(Uniform(0,1),15169,1)
    C5=rand(Binomial(20,0.6),15169,1)
    C6=rand(Binomial(20,0.5),15169,1)
    #for t in 1:5
        #for i in 1:15169
            #if C2t[i,1]<=.75*(6-t)/5
                #C2t[i,1]=1
            #else
                #C2t[i,1]=0
            #end
        #end
        #C3t=rand(Normal(15+t-1,5(t-1)),15169,1)
        #C4t=rand(Normal(π*(6-t)/3,1/ℯ),15169,1)
       #end
    


     t=1
     C21=rand(Uniform(0,1),15169,1)
       for i in 1:15169
           if C21[i,1]<=((.75*(6-t))/5)
               C21[i,1]=1
           else
               C21[i,1]=0
            end
        end
        C31=rand(Normal(15+t-1,5(t-1)),15169,1)
        C41=rand(Normal(π*(6-t)/3,1/ℯ),15169,1)
    
 
     t=2
     C22=rand(Uniform(0,1),15169,1)
       for i in 1:15169
           if C22[i,1]<=.75*(6-t)/5
               C22[i,1]=1
           else
               C22[i,1]=0
            end
        end
        C32=rand(Normal(15+t-1,5(t-1)),15169,1)
        C42=rand(Normal(π*(6-t)/3,1/ℯ),15169,1)

    t=3
    C23=rand(Uniform(0,1),15169,1)
    for i in 1:15169
        if C23[i,1]<=.75*(6-t)/5
            C23[i,1]=1
        else
            C23[i,1]=0
         end
     end
     C33=rand(Normal(15+t-1,5(t-1)),15169,1)
     C43=rand(Normal(π*(6-t)/3,1/ℯ),15169,1)
 
   
    t=4
    C24=rand(Uniform(0,1),15169,1)
    for i in 1:15169
        if C24[i,1]<=.75*(6-t)/5
            C24[i,1]=1
        else
            C24[i,1]=0
        end
     end
     C34=rand(Normal(15+t-1,5(t-1)),15169,1)
     C44=rand(Normal(π*(6-t)/3,1/ℯ),15169,1)
 

    t=5
    C25=rand(Uniform(0,1),15169,1)
      for i in 1:15169
          if C25[i,1]<=.75*(6-t)/5
              C25[i,1]=1
          else
              C25[i,1]=0
          end
        end
        C35=rand(Normal(15+t-1,5(t-1)),15169,1)
        C45=rand(Normal(π*(6-t)/3,1/ℯ),15169,1)
 
    X = [C1;;C21;;C31;;C41;;C5;;C6;;;C1;;C22;;C32;;C42;;C5;;C6;;;C1;;C23;;C33;;C43;;C5;;C6;;;C1;;C24;;C34;;C44;;C5;;C6;;;C1;;C25;;C35;;C45;;C5;;C6]
#(d). Use comprehensions to create a matrix β which is K*T and whose elements evolve across time in the following fashion
#1, 1.25, 1.5
#ln(t)
#-√t
#ℯᵗ-ℯᵗ⁺¹ 
#t 
#t/3
    T=5

    C1_new_=[0.75+0.25*t for t in 1:T]
    C1_new=reshape(C1_new_, (1,T))
    C2_new_=[√t for t in 1:T]
    C2_new=reshape(C2_new_, (1,T))
    C3_new_=[-√t for t in 1:T]
    C3_new=reshape(C3_new_, (1,T))
    C4_new_=[ℯ^t-ℯ^(t-1) for t in 1:T]
    C4_new=reshape(C4_new_, (1,T))
    C5_new_=[t for t in 1:T]
    C5_new=reshape(C5_new_, (1,T))
    C6_new_=[t/3 for t in 1:T]
    C6_new=reshape(C6_new_, (1,T))
    β = cat(C1_new, C2_new, C3_new, C4_new, C5_new, C6_new, dims=1)

#(e). Use comprehension to create a matrix Y which is N*T defined by Yₜ=Xₜβₜ+εₜ, where εₜ∼N(0, σ=.36)
    ε=rand(Normal(0,0.1269),15169,5)
    Y=X.*β+ε
    println(Y)
#(f). Wrap a function definition all of the code for question 2. Call the function q2(). The function should have take as inputs the arrays A,B and C. It should return nothing. At the very bottom of your script you should add the code q2(A,B,C). Make sure q2() gets called after q1().
end
q2(A,B,C)


#3. Reading in Data and calculating summary Statistics
#(a). Clear the workspace and import the file nlsw88.csv into Julia as a DataFrame. Make sure you appropriately convert missing values and variable names. Save the result as nlsw88.jld.
function q3()
    nlsw88=CSV.read("/Users/boweidong/Dropbox/My Mac (Boweis-MacBook-Air.local)/Desktop/nlsw88.csv", DataFrame, header=1, limit=2246)
    jldsave("nlsw88.jld")
    display(nlsw88)
#(b). What percentage of the sample has never been married? What percentage are college graduates?
    nlsw88.never_married
    nm=freqtable(nlsw88.never_married)
    234/(2012+234)
    College=sum(nlsw88.collgrad)
    532/(1714)
#(c). Use the freqtable() function to report what persentage of the sample is in each race category.
    freqtable(nlsw88.race)
#(d). Use the describe() function to create a matrix called summarystats which lists the mean, median, standard deviation, min, max, number of unique elements, and interquartile range (75th percentile minus 25th percentile) of the data frame. How many grade observations are missing?
    summarystats_1=describe(nlsw88, :all) 
    describe(nlsw88.grade) #2 of grades are missing
    select(summarystats_1, 1,2,3,5,7,8)
#(e). Show the joint distribution of industry and occupation using a cross-tanulation.
    freqtable(nlsw88.industry, nlsw88.occupation)
#(f). Tabulate the mean wage over industry and occupation categories. 
#Hint: you should first subset the data frame to only include the cols industry, occupation and wage. You should then follow the "split-apply-combine" direction here.
    new_tab=select(nlsw88,11,12,14)
#(g). Wrap a function of defination around all of the code for question 3. Call the function q3(). The function should have no inputs and no outputs. At the very bottom of your script you should add the code q3().
end
q3()


#4. Practice with functions
#(a). Load firstmatrix.jld.
function q4()
    jldsave("firstmatrix.jld";A,B,C,D)
    load("firstmatrix.jld")
#(b). Write a function called matrixops that takes as inputs the matrices A and B from question (a) of problem 1 and has 3 outputs:
#(i). the element-by-element product of the inputs
#(ii). the product A'B, and 
#(iii). the sum of all the elements of A+B.
        function matrixops(x,y)
            return  .*(x,y), x'*y, x+y
    end
#(c). Starting on line 2 of the function, write a comment that explains what matrixops does.
            function matrixops(x,y)
                return  .*(x,y) ,x'*y ,x+y 
        end
#to calculate the product A'*B
#to calculate the element-by-element product of the inputs,
#to calculate the elements of A+B
#(d). Evaluate matrixops() using A and B from question (a) of probklem 1.
            matrixops(A,B)
#(e). Just before the first excutable line of matrixops.m(i.e. right after the first-line comments), write an if statement which gives an error if the two inputs are not the same size. Have the error say "inputs must have the same size."
            function matrixopsm(x,y)
                if size(x)!=size(y)
                return "error: inputs must have the same size"
           else
                return .*(x,y),x'*y,x+y 
         end
    end
#(f). Evaluate matrixops.m using C and D from question (a) of problem 1. What happens?
            matrixopsm(C,D) #"error: inputs must have the same size"
#(g). Now evaluate matrixops.m using ttl_exp and wage from nlsw88.jld. 
#Hint: Before doing this, you will need to convert the data frame columns to Arrays. r.g. convert(Array, nlsw88, ttl_exp), depending on what you called the data frame object [I called it nlsw88]. 
            ttl_exp_=convert(Array, nlsw88.ttl_exp)
            wage_=convert(Array, nlsw88.wage)
            matrixopsm(ttl_exp_,wage_)
#(h). Wrap the function definition around all of the code for question 4. Call the function q4(). The function should have no inputs or outputs. At the bottom of your script you should add the code q4().
end
q4()


#5. Turn in your files as a commit to the ProblemSets/PS1-julia-intro/folder on your GitHutfork. You can do this by simply clicking "upload files" in the appropriate folder, or you can use the GitHub desktop app, R studio, VS code, or the command line to stage, commit and push the files.





 










