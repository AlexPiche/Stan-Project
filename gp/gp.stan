// Predict from Gaussian Process Logistic Regression
// Fixed covar function: eta_sq=1, rho_sq=1, sigma_sq=0.1

functions {
  real ARD(row_vector x, row_vector z, vector sigma, int D){
    real toRet;
    toRet <- 0.0;
    for(d in 1:D){
      toRet <- toRet + 0.5*pow(x[d]-z[d], 2)/sigma[d];
    }
    return toRet;
  }

  real BER(vector y, vector yhat, int Nt){
    real TN;
    real FN;
    real TP;
    real FP;
    real BER;
    real b1;
    TN <- 0.0;
    FN <- 0.0;
    TP <- 0.0;
    FP <- 0.0;
    for(i in 1:Nt){
      if(round(yhat[i]) == 0 && y[i] == 0) {
        TN <- TN + 1.0;
      }
      else if(round(yhat[i]) == 0 && y[i] == 1) {
        FN <- FN + 1.0 ;
      }
      else if(round(yhat[i]) == 1 && y[i] == 1) {
        TP <- TP + 1.0 ;
      }
      else if(round(yhat[i]) == 1 && y[i] == 0) {
        FP <- FP + 1.0 ;
      }
    }
    BER <- 0.5 * (FP/(FP+TN) + FN/(FN+TP));
    return BER;
  }
}

data {
  int<lower=1> D;     
  int<lower=1> N1;     
  matrix[N1,D] x1; 
  int<lower=0,upper=1> z1[N1];
  int<lower=1> N2;
  matrix[N2,D] x2;
  vector[N2] z2;
}
transformed data {
  int<lower=1> N;
  matrix[N1+N2,D] x;
  vector[N1+N2] mu;
  N <- N1 + N2;
  for (n in 1:N1) x[n] <- x1[n];
  for (n in 1:N2) x[N1 + n] <- x2[n];
  for (i in 1:N) mu[i] <- 0;
}
parameters {
  vector[N1] y1;
  vector[N2] y2;
  vector<lower=0>[D] rho2;
  real<lower=0> eta2;
  real<lower=0> sigma2;
}
transformed parameters {
  cov_matrix[N1+N2] Sigma;
  for (i in 1:(N-1)){
    for (j in (i+1):N){
      Sigma[i,j] <- eta2*exp(-ARD(x[i], x[j], rho2, D)) + if_else(i==j, 0.1, 0.0);
      Sigma[j, i] <- Sigma[i,j];
    }
  }
  for(k in 1:N)
    Sigma[k,k] <- eta2 + sigma2;
}
model {
  vector[N] y;
  for (n in 1:N1) y[n] <- y1[n];
  for (n in 1:N2) y[N1 + n] <- y2[n];

  sigma2 ~ inv_gamma(1,1);
  eta2 ~ inv_gamma(1,1);
  rho2 ~ inv_gamma(1,1);

  y ~ multi_normal(mu,Sigma);
  for (n in 1:N1)
    z1[n] ~ bernoulli_logit(y1[n]);
}

// to generate probabilistic predictionsfor z2
generated quantities {
  vector[N1] pr_z_eq_1;
  vector[N2] pr_z_eq_2;
  real<lower=0> average_error;
  vector[N1] log_lik;

  for (n in 1:N1){
    pr_z_eq_1[n] <- inv_logit(y1[n]);
    log_lik[n] <- bernoulli_logit_log(z1[n], pr_z_eq_1[n]);
  }
  for (n in 1:N2){
    pr_z_eq_2[n] <- inv_logit(y2[n]);
  }
  average_error <- BER(z2, pr_z_eq_2, N2);//sum(errors) / Nt;

}
