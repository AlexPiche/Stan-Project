functions {
  matrix scaled_inv_logit(matrix X) {
    matrix[rows(X), cols(X)] res;
    for(i in 1:rows(X))
      for(j in 1:cols(X)) 
        res[i,j] <- tanh(X[i,j]);//inv_logit(X[i,j])*2-1;
    return res;
  }
  
  vector calculate_alpha(matrix X, vector bias, matrix beta_first, matrix[] beta_middle, vector beta_output) {
    int N;
    int num_nodes;
    int num_layers;
    matrix[rows(X),rows(beta_first)] layer_values[rows(bias)];
    vector[rows(X)] alpha;
    
    N <- rows(X);
    num_nodes <- rows(beta_first);
    num_layers <- rows(bias);
    
    layer_values[1] <- scaled_inv_logit(bias[1] + X * beta_first');   
    for(i in 2:(num_layers-1)){
      layer_values[i] <- scaled_inv_logit(bias[i] + layer_values[i-1] * beta_middle[i-1]');
    }
    alpha <- bias[num_layers] + layer_values[num_layers-1] * beta_output;
    return alpha;
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
  int<lower=0> N;
  int<lower=0> d;
  int<lower=0> num_nodes;
  int<lower=1> num_middle_layers;
  int<lower=0> Nt;
  int y[N];
  matrix[N,d] X;
  vector[Nt] yt;
  matrix[Nt,d] Xt;
}
transformed data {
  int num_layers;
  num_layers <- num_middle_layers + 2;
}
parameters {
  vector[num_layers] bias;
  matrix[num_nodes,d] beta_first;
  matrix[num_nodes,num_nodes] beta_middle[num_middle_layers];
  vector[num_nodes] beta_output;
  real<lower=0> sigma2;
  vector<lower=0>[num_nodes*d] phi;
}
transformed parameters {
  real sigma;
  vector[num_nodes*d] one_over_sqrt_phi;
  sigma <- sqrt(sigma2);
  for(i in 1:(num_nodes*d))
    one_over_sqrt_phi[i] <- sqrt(phi[i]);
}
model{
  vector[N] alpha;
  alpha <- calculate_alpha(X, bias, beta_first, beta_middle, beta_output);
  y ~ bernoulli_logit(alpha);
  
  //priors
  bias ~ normal(0,1);
  phi ~ gamma(1,1);
  to_vector(beta_output) ~ normal(0,1);
  to_vector(beta_first) ~ normal(0, sigma * one_over_sqrt_phi);
  for(i in 1:(num_middle_layers)) 
    to_vector(beta_middle[i]) ~ normal(0,1);
}
generated quantities{
  vector[Nt] predictions;
  real<lower=0> errors[Nt];
  real<lower=0> average_error;
  vector[N] log_lik;
  vector[Nt] alphat;
  vector[N] alpha;
  alphat <- calculate_alpha(Xt, bias, beta_first, beta_middle, beta_output);

  for(i in 1:Nt){
    predictions[i] <- inv_logit(alphat[i]);
    errors[i] <- fabs(yt[i] - round(predictions[i]));
  }

  average_error <- BER(yt, predictions, Nt);//sum(errors) / Nt;

  alpha <- calculate_alpha(X, bias, beta_first, beta_middle, beta_output);

  for(n in 1:N)
    log_lik[n] <- bernoulli_logit_log(y[n], alpha[n]);
}
