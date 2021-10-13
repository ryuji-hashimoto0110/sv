# Usage

svmcmc.cppは確率的ボラティリティ変動モデルに特化したサンプリング方法であるMixture SamplerのRcpp,RcppArmadilloによる実装である．
ローカルのフォルダにcloneしてワーキングディレクトリにsvmcmc.cppに配置したのち，Rで以下を実行することでシミュレーションができる．

```r
library(MASS)
library(Rcpp)
library(RcppArmadillo)
Rcpp::sourceCpp("svmcmc.cpp")

#---
# シミュレーションデータを生成
#--

set.seed(111)
MU        = -8.0
PHI       = 0.97
SIGMA_ETA = 0.3
RHO       = -0.3
vmu       = c(0,0)
matsigma  = matrix(c(1, RHO*SIGMA_ETA,
                     RHO*SIGMA_ETA, SIGMA_ETA^2), 
                   nrow=2, byrow=TRUE)
h         = -8.0
Y         = c()
for(i in 1:2000){
  par     = mvrnorm(1, vmu, matsigma)
  epsilon = par[1]
  eta     = par[2]
  h       = MU + PHI * (h-MU) + eta
  y       = epsilon * exp(h / 2)
  Y       = append(Y, y)
}

#---
# mixture samplerの実行
#---

set.seed(111)
call_r_opt_func = TRUE
list_mcmc       = svmcmc(Y, call_r_opt_func)

#---
# 結果の描画
#---

vmu        = list_mcmc[[1]]
vphi       = list_mcmc[[2]]
vsigma_eta = list_mcmc[[3]]
vrho       = list_mcmc[[4]]
oldpar = par(no.readonly = T)
par(mfrow  = c(2,2))
par(oma    = c(0, 0, 0, 0))
par(mar    = c(4, 4, 2, 1))
plot(vmu, type='l', main='mu (ground truth=-8)')
plot(vphi, type='l', main='phi (ground truth=0.97)')
plot(vsigma_eta, type='l', main='sigma_eta (ground truth=0.3)')
plot(vrho, type='l', main='rho (ground truth=-0.3)')
par(oldpar)

Y_star = list_mcmc[[5]]
h      = list_mcmc[[6]]
plot(Y_star, type='l', lwd=1, col='blue', ylim=c(-10,-6))
par(new=T)
plot(h, type='l', lwd=1, col='orange', ylim=c(-10,-6))
legend('topleft',
       legend=c('Y_star', 'h'), col=c('blue', 'orange'))
par(oldpar)
```

![simuoation_result](https://user-images.githubusercontent.com/75870240/137111765-95d26068-76a1-4011-a196-bd2a99c5f1f8.png)
