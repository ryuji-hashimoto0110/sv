// svmcmc.cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins("cpp11")]]


// [[Rcpp::export]]
arma::vec sample_s(arma::vec h,
                   double mu, double phi, double sigma_eta, double rho,
                   arma::vec p, arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                   arma::vec Y_star, 
                   arma::vec d, 
                   int T);


// [[Rcpp::export]]
Rcpp::List kalman_filter(arma::vec s, 
                         double mu, double phi, double sigma_eta, double rho,
                         arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                         arma::vec Y_star, 
                         arma::vec d, 
                         int T);

// [[Rcpp::export]]
arma::vec sim_smoother(arma::vec s, 
                       double mu, double phi, double sigma_eta, double rho,
                       arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                       arma::vec Y_star, 
                       arma::vec d, 
                       int T);

// [[Rcpp::export]]
double loglikelihood(arma::vec s, 
                     double mu, double phi, double sigma_eta, double rho,
                     arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                     arma::vec Y_star, 
                     arma::vec d, 
                     double mu_0, double sigma_0, 
                     double T);

double calc_posterior(arma::vec x, double mu,
                      arma::vec s, 
                      arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                      arma::vec Y_star, arma::vec d, double T,
                      double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0);

// [[Rcpp::export]]
double calc_posterior_maximize(arma::vec x, double mu,
                               arma::vec s,
                               arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                               arma::vec Y_star, arma::vec d, double T,
                               double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0);

// [[Rcpp::export]]
arma::vec deriv1(arma::vec x, double mu, 
                 arma::vec s,
                 arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                 arma::vec Y_star, arma::vec d, double T,
                 double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0);

// [[Rcpp::export]]
arma::mat deriv2(arma::vec x, double mu, 
                 arma::vec s,
                 arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                 arma::vec Y_star, arma::vec d, double T,
                 double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0);

// [[Rcpp::export]]
arma::vec Opt(arma::vec x, double mu, 
              arma::vec s,
              arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
              arma::vec Y_star, arma::vec d, double T,
              double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0);

// [[Rcpp::export]]
arma::vec aug_kalman_filter(arma::vec s,
                            arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                            arma::vec Y_star, arma::vec d, int T,
                            double mu, double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0,
                            bool call_r_opt_func);


// [[Rcpp::export]]
Rcpp::List svmcmc(arma::vec Y,         // 日次リターンを格納したベクトル
                  bool call_r_opt_func // Rの最大化関数を呼び出すか否か
                  ){ 
  
  /*---設定---*/
  int nsim, nburn;
  nsim  = 5000;       // サンプリングの個数
  nburn = 0.1 * nsim; // 稼働検査期間
  
  /*---初期値の設定---*/
  double phi, sigma_eta, mu, rho;
  arma::vec h;
  mu        = -8.0;
  phi       = 0.97;
  sigma_eta = 0.1;
  rho       = 0;
  h         = arma::zeros(Y.n_elem) + mu;
  
  /*---パラメータの事前分布---*/
  double a_0, b_0, n_0, S_0, mu_0, sigma_0;
  arma::vec p, m, v, a, b;
  mu_0    = -8.0; // mu~N(mu_0,sigma_0)
  sigma_0 = 1;
  a_0     = 20;   // phi~Beta(a_0,b_0)
  b_0     = 1.5;
  n_0     = 5;    // sigma_eta^2~IG(n_0/2,S_0/2)
  S_0     = 0.05;
  // s
  p       = {0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 
             0.18842, 0.12047, 0.05591, 0.01575, 0.00115};
  m       = {1.92677, 1.34744, 0.73504, 0.02266, -0.85173, 
             -1.97278, -3.46788, -5.55246, -8.68384, -14.65};
  v       = {0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 
             0.98583, 1.57469, 2.54498, 4.16591, 7.33342};
  v       = sqrt(v); // 上のベクトルはv^2なのでsqrtを取る．
  //v     = {0.3356, 0.4218, 0.5174, 0.6373, 0.7918,
  //         0.9929, 1.2549, 1.5953, 2.0411, 2.7080}
  a       = {1.01418, 1.02248, 1.03403, 1.05207, 1.08153, 
             1.13114, 1.21754, 1.37454, 1.68327, 2.50097};
  b       = {0.5071, 0.51124, 0.51701, 0.52604, 0.54076, 
             0.56557, 0.60877, 0.68728, 0.84163, 1.25049};
  
  /*---変数の定義---*/
  int T;
  double dcst;
  arma::vec Y_star, d, s, theta;
  arma::vec mu_result, phi_result, sigma_eta_result, rho_result;
  Rcpp::NumericVector Y_, d_;
  Rcpp::List result;
  T      = Y.n_elem;        // 全時間長
  dcst   = 0.0001;          // 十分に小な定数
  Y_     = Rcpp::wrap(Y);   // YをRcppのNumericVector化
  d_     = Rcpp::ifelse(Y_ > 0, 1.0, -1.0); // y_t>0なら1, y_t<0なら-1を要素にもつ長さTのベクトル
  d      = Rcpp::as<arma::vec>(d_); // d_をarma::vec化
  Y_star = log(Y%Y + dcst); // log(y^2)
  
  mu_result        = arma::zeros(nsim);
  phi_result       = arma::zeros(nsim);
  sigma_eta_result = arma::zeros(nsim);
  rho_result       = arma::zeros(nsim);
  
  /*---イテレーション開始---*/
  int k;
  for(k = -nburn; k < nsim; k++){
    
    /*---多項分布からのsのサンプリング---*/
    s = sample_s(h, 
                 mu, phi, sigma_eta, rho, 
                 p, m, v, a, b, 
                 Y_star, d, T);
    
    /*---シミュレーションスムーザによるh_tのサンプリング---*/
    h = sim_smoother(s, 
                     mu, phi, sigma_eta, rho, 
                     m, v, a, b, 
                     Y_star, d, T);
    
    /*---拡大カルマンフィルタによるvartheta, muのサンプリング---*/
    theta     = aug_kalman_filter(s, 
                                  m, v, a, b, 
                                  Y_star, d, T, 
                                  mu, mu_0, sigma_0, a_0, b_0, n_0, S_0, call_r_opt_func);
    
    mu        = theta[0];
    phi       = theta[1];
    sigma_eta = theta[2];
    rho       = theta[3];
    
    /*---保存---*/
    if(0 <= k){
      mu_result[k]        = mu;
      phi_result[k]       = phi;
      sigma_eta_result[k] = sigma_eta;
      rho_result[k]       = rho;
    }
  }
  
  result = Rcpp::List::create(mu_result, phi_result, sigma_eta_result, rho_result, Y_star, h);
  
  return(result);
}


arma::vec sample_s(arma::vec h, // h_t, t=1,...,T
                   double mu, double phi, double sigma_eta, double rho, // theta={mu, phi, sigma_eta, rho}
                   arma::vec p, arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                   arma::vec Y_star, arma::vec d, int T){
  
  int t;
  arma::vec eps_star, eta;
  arma::vec result = arma::zeros(T);
  for(t = 0; t < T; t++){
    
    /*---事後分布を計算---*/ 
    eps_star = arma::pow((Y_star[t]-h[t]-m), 2) / (2*v%v); // 指数部分の中身1 (epsilon_starの尤度)
    if(t!=T-1){                                            // 指数部分の中身2 (etaの尤度)
      eta = arma::pow((h[t+1]-mu)-phi*(h[t]-mu)-d[t]*rho*sigma_eta*arma::exp(m/2)%(a+b%(Y_star[t]-h[t]-m)), 2) 
             /  (2*sigma_eta*sigma_eta*(1-rho*rho));
    }else{
      eta = arma::pow(-phi*(h[t]-mu)-d[t]*rho*sigma_eta*arma::exp(m/2)%(a+b%(Y_star[t]-h[t]-m)), 2)
             / (2*sigma_eta*sigma_eta*(1-rho*rho));
    }
    
    arma::vec pi = p % (1/v) % (arma::exp(- (eps_star - arma::min(eps_star))
                                          - (eta - arma::min(eta))));
    pi           = pi / arma::sum(pi);
    Rcpp::NumericVector j = {0,1,2,3,4,5,6,7,8,9};
    double s_t   = Rcpp::sample(j, 1, true, Rcpp::wrap(pi))[0];
    result[t]    = s_t;
  }
  
  return(result);
}


Rcpp::List kalman_filter(arma::vec s, // s_t, t=1,...,T
                         double mu, double phi, double sigma_eta, double rho, // theta={mu, phi, sigma_eta, rho}
                         arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                         arma::vec Y_star, arma::vec d, int T){
  
  /*---カルマンフィルタの実行---*/
  int t;
  double h = mu;
  double P = sigma_eta*sigma_eta / (1-phi*phi);
  double h_star = 0; 
  double A_star = -1;
  arma::vec e, D, K, L, f, F;
  arma::mat J;
  e = arma::zeros(T);
  D = arma::zeros(T);
  K = arma::zeros(T);
  L = arma::zeros(T);
  J = arma::zeros(T,2);
  f = arma::zeros(T);
  F = arma::zeros(T);
  
  for(t = 0; t < T; t++){
    arma::vec H = {d[t]*rho*sigma_eta*b[s[t]]*v[s[t]]*exp(m[s[t]]/2), sigma_eta*sqrt(1-rho*rho)};
    arma::vec G = {v[s[t]], 0};
    arma::vec W = {0, 1-phi, d[t]*rho*sigma_eta*a[s[t]]*exp(m[s[t]]/2)};
    arma::vec beta = {1, mu, 1};
    arma::vec B = {0, 1, 0};
    
    e[t] = Y_star[t] - m[s[t]] - h;
    D[t] = P + arma::dot(G,G);
    K[t] = (phi*P + arma::dot(H,G)) / D[t];
    L[t] = phi - K[t];
    J(t, arma::span(0,1)) = (H - K[t]*G).t();
    f[t] = Y_star[t] - m[s[t]] - h_star;
    F[t] = - A_star;
    
    h      = arma::dot(W,beta) + phi*h + K[t]*e[t];
    P      = phi*P*L[t] + arma::dot(H, J(t,arma::span(0,1)).t());
    h_star = W[2] + phi*h_star + K[t]*f[t];
    A_star = phi - 1 + phi*A_star + K[t]*F[t];
  }
  
  Rcpp::List result = Rcpp::List::create(e, D, J, L, f, F);
  
  return(result);
}


arma::vec sim_smoother(arma::vec s, // s_t, t=1,...,T
                       double mu, double phi, double sigma_eta, double rho, // theta={mu, phi, sigma_eta, rho}
                       arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                       arma::vec Y_star, arma::vec d, int T){
  
  /*---カルマンフィルタの実行---*/
  int t;
  Rcpp::List kalman_list;
  
  kalman_list = kalman_filter(s, mu, phi, sigma_eta, rho, m, v, a, b, Y_star, d, T);
  arma::vec e = kalman_list[0];
  arma::vec D = kalman_list[1];
  arma::mat J = kalman_list[2];
  arma::vec L = kalman_list[3];
  
  /*---平滑化の実行---*/
  double r, U, C, kappa, V;
  arma::vec eta, result;
  r = 0;
  U = 0;
  eta = arma::zeros(T);
  result = arma::ones(T) * mu;
  for(t = T-1; t > -1; t--){
    arma::mat H  = {d[t]*rho*sigma_eta*b[s[t]]*v[s[t]]*exp(m[s[t]]/2), sigma_eta*sqrt(1-rho*rho)};
    arma::mat G  = {v[s[t]], 0};
    arma::mat mJ = J(t,arma::span(0,1));
    arma::mat I  = arma::eye(2,2);
    
    arma::mat matC   = H * (I - G.t()*G/D[t] - mJ.t()*mJ*U) * H.t();
    C                = matC(0,0);
    kappa            = R::rnorm(0, sqrt(C));
    arma::mat matV   = H * (G.t()/D[t] + mJ.t()*U*L[t]);
    V                = matV(0,0);
    arma::mat mateta = H * (G.t()*e[t]/D[t] + mJ.t()*r) + kappa;
    eta[t]           = mateta(0,0);
    r                = e[t]/D[t] + L[t]*r - kappa*V/C;
    U                = 1/D[t] + L[t]*U*L[t] + V*V/C;
  }
  
  arma::mat H_0    = {0, sigma_eta/sqrt(1-phi*phi)};
  arma::mat mJ_0   = H_0;
  arma::mat I      = arma::eye(2,2);
  arma::mat matC_0 = H_0 * (I - mJ_0.t()*mJ_0*U) * H_0.t();
  double C_0       = matC_0(0,0);
  double kappa_0   = R::rnorm(0, sqrt(C_0));
  arma::mat mateta = H_0 * mJ_0.t()*r + kappa_0;
  double eta_0     = mateta(0,0);
  result[0]        = mu + eta_0;
  
  for(t = 1; t < T; t++){
    result[t] = mu*(1-phi)+d[t-1]*rho*sigma_eta*a[s[t-1]]*exp(m[s[t-1]]/2)+phi*result[t-1]+eta[t-1];
  }
  
  return(result);
}


double loglikelihood(arma::vec s, // s_t, t=1,...,T
                     double mu, double phi, double sigma_eta, double rho, // theta={mu, phi, sigma_eta, rho}
                     arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                     arma::vec Y_star, arma::vec d, double mu_0, double sigma_0, double T){
  
  /*---カルマンフィルタの実行---*/
  Rcpp::List kalman_list;
  kalman_list = kalman_filter(s, mu, phi, sigma_eta, rho, m, v, a, b, Y_star, d, T);
  arma::vec D = kalman_list[1];
  arma::mat f = kalman_list[4];
  arma::vec F = kalman_list[5];
  
  /*---対数尤度の計算---*/
  double C_1, mu_1;
  C_1  = 1 / ( 1/sigma_0 + arma::sum(F%(1/D)%F) );
  mu_1 = C_1 * ( mu_0/sigma_0 + arma::sum(F%(1/D)%f) );
  
  double result;
  result = -T*log(2*arma::datum::pi)/2 - arma::sum(log(abs(D)))/2 -
           log(abs(sigma_0))/2 + log(abs(C_1))/2 - 
           ( arma::sum(f%(1/D)%f) + mu_0*(1/sigma_0)*mu_0 - mu_1*(1/C_1)*mu_1 )/2;
  
  return(result);
}


double calc_posterior(arma::vec x, double mu, // x={phi, sigma_eta, rho}
                      arma::vec s,
                      arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                      arma::vec Y_star, arma::vec d, double T,
                      double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0){
  
  double phi, sigma_eta, rho;
  phi       = x[0];
  sigma_eta = x[1];
  rho       = x[2];
  
  /*---事前分布の定義---*/
  double log_prior_mu, log_prior_phi, log_prior_sigma_eta;
  log_prior_mu        = R::dnorm(mu, mu_0, sigma_0, true);
  log_prior_phi       = R::dbeta(phi, a_0, b_0, true);  // phi~Beta(a_0,b_0)
  log_prior_sigma_eta = R::dgamma(1/(sigma_eta*sigma_eta), n_0/2, 2/S_0, true)
                        - 2 * log(sigma_eta*sigma_eta); // sigma_eta^2~IG(n_0/2,S_0/2)
  
  /*---対数尤度の計算---*/
  double loglike = loglikelihood(s, mu, phi, sigma_eta, rho, m, v, a, b, Y_star, d, mu_0, sigma_0, T);
  
  /*---事後分布の計算---*/
  double result = loglike + log_prior_phi + log_prior_sigma_eta + log_prior_mu;

  return(result);
}


double calc_posterior_maximize(arma::vec x, double mu, // x={phi_, sigma_eta_, rho_}
                               arma::vec s,
                               arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                               arma::vec Y_star, arma::vec d, double T,
                               double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0){
  
  double phi_, sigma_eta_, rho_, phi, sigma_eta, rho;
  phi_       = x[0];
  phi        = (exp(phi_)-1) / (exp(phi_)+1);
  sigma_eta_ = x[1];
  sigma_eta  = exp(sigma_eta_);
  rho_       = x[2];
  rho        = (exp(rho_)-1) / (exp(rho_)+1);
  
  /*---事前分布の定義---*/
  double log_prior_mu, log_prior_phi, log_prior_sigma_eta;
  log_prior_mu        = R::dnorm(mu, mu_0, sigma_0, true);
  log_prior_phi       = R::dbeta(phi, a_0, b_0, true);  // phi~Beta(a_0,b_0)
  log_prior_sigma_eta = R::dgamma(1/(sigma_eta*sigma_eta), n_0/2, 2/S_0, true)
                        - 2 * log(sigma_eta*sigma_eta); // sigma_eta^2~IG(n_0/2,S_0/2)
  
  /*---対数尤度の計算---*/
  double loglike = loglikelihood(s, mu, phi, sigma_eta, rho, m, v, a, b, Y_star, d, mu_0, sigma_0, T);
  
  /*---ヤコビアンの計算---*/
  double jacobian = phi_ + sigma_eta_ + rho_ + 2*log(2) 
                    - 2*log(exp(phi_)+1) - 2*log(exp(rho_)+1);

  /*---事後分布の計算---*/
  double result = loglike + log_prior_phi + log_prior_sigma_eta + log_prior_mu + jacobian;

  return(result);
}


arma::vec deriv1(arma::vec x, double mu, // x={phi_, sigma_eta_, rho_}
                 arma::vec s,
                 arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                 arma::vec Y_star, arma::vec d, double T,
                 double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0){
  
  int i;
  arma::vec e = arma::zeros(3);
  double epsilon = 0.001;
  arma::vec result = arma::zeros(3);
  
  for(i = 0; i < 3; i++){
    e[i] = 1;
    result[i] = (
      calc_posterior_maximize(x+epsilon*e, mu, s, m, v, a, b, Y_star, d, T, mu_0, sigma_0, a_0, b_0, n_0, S_0) 
      - calc_posterior_maximize(x-epsilon*e, mu, s, m, v, a, b, Y_star, d, T, mu_0, sigma_0, a_0, b_0, n_0, S_0)) 
      / (2*epsilon);
    e[i] = 0;
  }
  
  return(result);
}


arma::mat deriv2(arma::vec x, double mu, // x={phi, sigma_eta, rho}
                 arma::vec s,
                 arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                 arma::vec Y_star, arma::vec d, double T,
                 double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0){
  
  int i, j;
  arma::vec e_i, e_j;
  e_i = arma::zeros(3);
  e_j = arma::zeros(3);
  double epsilon = 0.000001;
  
  arma::mat result = arma::zeros(3,3);
  
  for(i = 0; i < 3; i++){
    for(j = 0; j < 3; j++){
      e_i[i] = 1;
      e_j[j] = 1;
      result(i,j) = (
        calc_posterior_maximize(x+epsilon*e_i+epsilon*e_j, mu, s, m, v, a, b, Y_star, d, T, mu_0, sigma_0, a_0, b_0, n_0, S_0)
        + calc_posterior_maximize(x-epsilon*e_i-epsilon*e_j, mu, s, m, v, a, b, Y_star, d, T, mu_0, sigma_0, a_0, b_0, n_0, S_0)
        - calc_posterior_maximize(x-epsilon*e_i+epsilon*e_j, mu, s, m, v, a, b, Y_star, d, T, mu_0, sigma_0, a_0, b_0, n_0, S_0)
        - calc_posterior_maximize(x+epsilon*e_i-epsilon*e_j, mu, s, m, v, a, b, Y_star, d, T, mu_0, sigma_0, a_0, b_0, n_0, S_0)
      ) / (4*epsilon*epsilon);
      e_i[i] = 0;
      e_j[j] = 0;
    }
  }
  
  return(result);
}


arma::vec Opt(arma::vec x, double mu, // x={phi_, sigma_eta_, rho}
              arma::vec s, 
              arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
              arma::vec Y_star, arma::vec d, double T,
              double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0){

  Rcpp::Environment stats("package:stats"); 
  Rcpp::Function optim = stats["optim"];
  
  Rcpp::List control = Rcpp::List::create(Rcpp::Named("fnscale") = -1.0);
  Rcpp::List out = optim(Rcpp::_["par"]    = x,
                         // Make sure this function is not exported! <- ??
                         Rcpp::_["fn"] = Rcpp::InternalFunction(&calc_posterior_maximize),
                         Rcpp::_["mu"] = mu, Rcpp::_["s"] = s, Rcpp::_["m"] = m, Rcpp::_["v"] = v, Rcpp::_["a"] = a,
                         Rcpp::_["b"] = b, Rcpp::_["Y_star"] = Y_star, Rcpp::_["d"] = d, Rcpp::_["T"] = T, 
                         Rcpp::_["mu_0"] = mu_0, Rcpp::_["sigma_0"] = sigma_0, Rcpp::_["a_0"] = a_0, Rcpp::_["b_0"] = b_0,
                         Rcpp::_["n_0"] = n_0, Rcpp::_["S_0"] = S_0,
                         Rcpp::_["method"]  = "Nelder-Mead",
                         Rcpp::_["control"] = control,
                         Rcpp::_["hessian"] = false);

  arma::vec result = out["par"];
  
  return(result);
}


arma::vec aug_kalman_filter(arma::vec s,
                            arma::vec m, arma::vec v, arma::vec a, arma::vec b, 
                            arma::vec Y_star, arma::vec d, int T,
                            double mu, double mu_0, double sigma_0, double a_0, double b_0, double n_0, double S_0,
                            bool call_r_opt_func){
  
  double alpha, phi, phi_, sigma_eta, sigma_eta_, rho, rho_, threshold;
  arma::vec x, deriv1_vec;
  bool found_max;
  
  /*---最大化問題---*/
  threshold = 0.0001; // 閾値
  phi       = 0.8;    // 初期値
  phi_      = log((1+phi)/(1-phi)); // 定義域が実数全体になるように変数変換
  sigma_eta = 0.15;
  sigma_eta_= log(sigma_eta);
  rho       = -0.3;
  rho_      = log((1+rho)/(1-rho));
  x         = {phi_, sigma_eta_, rho_};
  if(call_r_opt_func==true){
    x = Opt(x, mu, s, m, v, a, b, Y_star, d, T, mu_0, sigma_0, a_0, b_0, n_0, S_0);
  }else{
    found_max = false; // 終了要件
    while(found_max == false){
      alpha       = 0.001; // 学習率
      deriv1_vec  = deriv1(x, mu, s, m, v, a, b, Y_star, d, T, mu_0, sigma_0, a_0, b_0, n_0, S_0);
      if(arma::sum(arma::abs(deriv1_vec)) < threshold){
        found_max = true;
        break;
      }else{
        x = x + alpha * deriv1_vec;
      }
    }
  }
  
  /*---二階微分の計算---*/
  arma::mat deriv2_mat = deriv2(x, mu, s, m, v, a, b, Y_star, d, T, mu_0, sigma_0, a_0, b_0, n_0, S_0);
  
  /*---varthetaのサンプリング---*/
  arma::mat inverse_deriv2_mat = arma::inv(deriv2_mat);
  arma::vec vartheta = x + inverse_deriv2_mat * Rcpp::as<arma::vec>(Rcpp::rnorm(3,0,1));
  phi_       = vartheta[0];
  sigma_eta_ = vartheta[1];
  rho_       = vartheta[2];
  phi        = (exp(phi_)-1) / (exp(phi_)+1);
  sigma_eta  = exp(sigma_eta_);
  rho        = (exp(rho_)-1) / (exp(rho_)+1);
  
  /*---muのサンプリング---*/
  double C_1, mu_1;
  Rcpp::List kalman_list;
  kalman_list = kalman_filter(s, mu, phi, sigma_eta, rho, m, v, a, b, Y_star, d, T);
  arma::vec D = kalman_list[1];
  arma::vec f = kalman_list[4];
  arma::vec F = kalman_list[5];
  C_1  = 1 / ( 1/sigma_0 + arma::sum(F%(1/D)%F) );
  mu_1 = C_1 * ( mu_0/sigma_0 + arma::sum(F%(1/D)%f) );
  mu   = R::rnorm(mu_1, C_1);
  
  arma::vec result = {mu, phi, sigma_eta, rho};
  
  return(result);
}


