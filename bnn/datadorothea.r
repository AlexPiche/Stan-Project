load("../dorothea/dorothea.rda")

N <- 800
d <- 100000
y <- y.train
X <- x.train
Xt <- x.test
Nt <- dim(x.test)[1]
num_nodes <- 100
num_middle_layers <- 1

y <- plyr::mapvalues(y, -1, 0)

rstan::stan_rdump(c('N','d','y','X','Xt', 'Nt', 'num_nodes', 'num_middle_layers'),file="dorothea.data.R") 
