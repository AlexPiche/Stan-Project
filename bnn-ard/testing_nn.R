library(rstan)

load("../dorothea/dorothea.rda")
N <- 800

data = iris[1:100,]
data[,1:4] = scale(data[,1:4])
data[,5] = as.integer(data[,5])-1

N = 80
Nt = nrow(data)-N
train_ind = sample(100,N)
test_ind = setdiff(1:100, train_ind) 

yt = y.train
stan.dat=list(
  num_nodes=10, 
  num_middle_layers=3, 
  d=100000, 
  N=800, 
  Nt=350, 
  X=x.train, 
  y=y.train,  
  Xt=x.valid)

m <- stan_model("bnn.stan")
s <- sampling(m, data = stan.dat, iter = 1000, chains = 4)

fitmat = as.matrix(s)
predictions = fitmat[,grep("predictions", colnames(fitmat))]
parameters = fitmat[,grep("beta", colnames(fitmat))]

mean_predictions = colMeans(predictions)
plot(1:Nt, yt)
lines(1:Nt, mean_predictions, type='p', col='red')
