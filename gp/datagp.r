#load("../dorothea/dorothea.rda")
#
#D <- 100000
#x1 <- x.train
#z1 <- y.train
#z1 <- plyr::mapvalues(z1, -1, 0)
#N1 <- 800
#x2 <- x.valid
#N2 <- 350

data(mtcars)
mtcars <- mtcars[sample(nrow(mtcars)),]

D <- 10
N1 <- 24
N2 <- 8
z1 <- mtcars[1:24,9]
x1 <- mtcars[1:24,-9]
x2 <- mtcars[25:32,-9]
z2 <- mtcars[25:32,9]

rstan::stan_rdump(c('D', 'N1', 'x1', 'z1', 'N2', 'x2'),file="gp.data.R")


keynames <- list.files(path = "", full.names = TRUE)  
dataKey <- lapply(keynames, read.csv, stringsAsFactors=FALSE, row.names = 1)
