functions {
  matrix scaled_inv_logit(matrix X) {
    matrix[rows(X), cols(X)] res;
    for(i in 1:rows(X))
      for(j in 1:cols(X)) 
        res[i,j] <- inv_logit(X[i,j])*2-1;
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
  real b1;
  real b2;
  matrix[num_nodes, d] W1;
  vector[num_nodes] W2;
  real<lower=0> sigma2;
  vector<lower=0>[num_nodes*d] phi;
}
transformed parameters {
  real sigma;
  vector[num_nodes*d] one_over_sqrt_phi;
  sigma <- sqrt(sigma2);
  for(i in 1:(num_nodes*d))
    one_over_sqrt_phi[i] <- 1;///sqrt(phi[i]);
}
model{
  matrix[N, d] middle_layer;
  vector[N] output_layer;
  to_vector(W1) ~ normal(0, 1);
  W2 ~ normal(0, 1);
  b1 ~ normal(0, 1);
  b2 ~ normal(0, 1);

  middle_layer <- scaled_inv_logit(X * W1 + b1);
  for(i in 1:N)
    output_layer[i] <- inv_logit(middle_layer[i,] * W2 + b2);

  y ~ bernoulli_logit(output_layer);
}
generated quantities{
  vector[Nt] predictions_test;
  real<lower=0> errors[Nt];
  real<lower=0> average_error;
  matrix[Nt, d] middle_layer_test;
  vector[Nt] output_layer_test;
  vector[N] log_lik;
  matrix[N, d] middle_layer;
  vector[N] output_layer;

  middle_layer <- scaled_inv_logit(X * W1 + b1);
  for(i in 1:N){
    output_layer[i] <- inv_logit(middle_layer[i,] * W2 + b2);
    log_lik[i] <- bernoulli_logit_log(y[i], output_layer[i]);
  }

  middle_layer_test <- scaled_inv_logit(Xt * W1 + b1);

  for(i in 1:Nt){
    output_layer_test[i] <- inv_logit(middle_layer_test[i,] * W2 + b2);
    predictions_test[i] <- inv_logit(output_layer_test[i]);
    errors[i] <- fabs(yt[i] - round(predictions_test[i]));
  }

  average_error <- BER(yt, predictions_test, Nt);//sum(errors) / Nt;

}
