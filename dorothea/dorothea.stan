functions {
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
  int N;
  int Ntest;
  int D;
  int<lower=0,upper=1> y[N];
  matrix[N,D] x;
  vector<lower=0,upper=1>[Ntest] ytest;
  matrix[Ntest,D] xtest;
  real<lower=0> lambda;
}
parameters {
  //real<lower=-0.5,upper=0.5> alpha;
  vector<lower=-0.5,upper=0.5>[D] beta;
  real alpha;
  //vector[D] beta;
  vector<lower=0>[D] sigma;
}
transformed parameters {
  vector[D] zeros;        // zeros for mean of MVN
  for (i in 1:D)
    zeros[i] <- 0.0;    
}
model {
  alpha ~ normal(0,5);
  sigma ~ inv_gamma(1,2);
  beta ~ multi_normal_prec(zeros, diag_matrix(sigma));
  //increment_log_prob(- lambda * dot_self(beta)); // Ridge to perform variable selection
  //for (d in 1:D){
    //increment_log_prob(- lambda * fabs(beta[d])); // Lasso to perform variable selection
  //}

  for(n in 1:N)
    y[n] ~ bernoulli(inv_logit(alpha + x[n]*beta)); //more efficient
}
generated quantities {
  real<lower=0> errors[Ntest];
  real<lower=0> average_error;
  vector[N] log_lik;
  vector[Ntest] predictions;

  for(n in 1:Ntest){
    predictions[n] <- inv_logit(alpha + xtest[n]*beta);
    errors[n] <- fabs(ytest[n] - round(predictions[n]));
  }
  average_error <- BER(ytest, predictions, Ntest);//sum(errors) / Nt;
  //average_error <- sum(errors) / Ntest;
  for(n in 1:N)
    log_lik[n] <- bernoulli_logit_log(y[n], alpha + x[n] * beta);
}
