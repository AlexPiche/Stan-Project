
load("dorothea.rda")

pmatrix <- scale(x.train)
princ <- prcomp(x.train)

nComp <- 75
pca_x.train <- data.frame(predict(princ, newdata=x.train)[,1:nComp])
pca_x.valid <- data.frame(predict(princ, newdata=x.valid)[,1:nComp])

my.model <- glm(y.train~., data=pca_x.train)

yhat_valid <- predict(my.model, newdata = data.frame(pca_x.valid))


N <- 800
Ntest <- dim(pca_x.valid)[1]
D <- nComp
y <- y.train
x <- scale(pca_x.train)
ytest <- y.valid
xtest <- scale(pca_x.valid)
lambda <- 0.5

y <- plyr::mapvalues(y, -1, 0)
ytest <- plyr::mapvalues(ytest, -1, 0)

rstan::stan_rdump(c('N','Ntest','D','y','x','ytest', 'xtest', 'lambda'),
                  file="dorothea.data.R")

N <- 800
Nt <- dim(pca_x.valid)[1]
d <- nComp
num_nodes <- 25
num_middle_layers <- 1
y <- y.train
X <- scale(pca_x.train)
yt <- y.valid
Xt <- scale(pca_x.valid)
y <- plyr::mapvalues(y, -1, 0)
yt <- plyr::mapvalues(yt, -1, 0)

rstan::stan_rdump(c('N','d', 'num_nodes', 'num_middle_layers',
                    'Nt','y','X','yt', 'Xt'),file="dorotheabnn.data.R")


N1 <- 800
N2 <- dim(pca_x.valid)[1]
D <- nComp
z1 <- y.train
x1 <- scale(pca_x.train)
z2 <- y.valid
x2 <- scale(pca_x.valid)
z1 <- plyr::mapvalues(z1, -1, 0)
z2 <- plyr::mapvalues(z2, -1, 0)

rstan::stan_rdump(c('D', 'N1', 'x1', 'z1', 'N2', 'x2', 'z2'),
                  file="dorotheagp.data.R")
