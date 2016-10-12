
rm(list=ls(all=TRUE))

require(glmnet)

set.seed(100)

n=1000
theta = runif(n, min = -1, max = 1)
sparse_index = sample (1:n, size = 800, replace = FALSE)
theta[sparse_index] = 0

sigma2 = 0.1

y = rep(0, n)

for ( i in 1:n){
  y[i] = rnorm(1, mean = theta[i], sd = sqrt(sigma2) )
}

lasso_estimate = function( y, lambda){
  return ( sign(y) * max( abs(y) - lambda, 0 ) )
}

lambda_grid = seq(0, 2, by = 0.05)

theta_hat = matrix(0, ncol = length(lambda_grid), nrow = n)
for ( i in 1:length(lambda_grid)){
  for (j in 1:n){
    theta_hat[j,i] = lasso_estimate( y = y[j], lambda = lambda_grid[i] )
  }
}

'
for (i in 1:length( lambda_grid) ){
  plot( y=theta_hat[,i], x = theta, main = toString( lambda_grid[i] ) , xlim=c(-1, 1), ylim=c(-1,1) )
  abline( 0, 1 )
}
'

mse = rep(0, length( lambda_grid ) ) 
for (i in 1:length( lambda_grid ) ){
  mse[i] = mean( (theta_hat[,i] - theta)^2 )
}

plot( y=mse, x=lambda_grid, type = "l", lwd = 3)


##########################################
#      Application to the dataset
##########################################

# cleaning up

rm(list=ls(all=TRUE))
require(glmnet)
set.seed(100)

X = read.csv("/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_5/data/diabetesX.csv", header = TRUE )
y = read.table("/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_5/data/diabetesY.txt", sep=";", header = FALSE )
y = as.numeric( y )
y = (y-mean(y))/sd(y)
X = as.matrix( X )

n = length(y)
p = ncol(X)

# estimating sigma2
ols = lm( y ~ X - 1 )
beta_hat_ols = ols$coefficients
sigma2_hat = sum( (y - X %*% beta_hat_ols)^2 )/p

# lasso fit
fit = glmnet( y=y, x=X, family = "gaussian", intercept = FALSE )

# path of coeficients
plot.glmnet( x=fit, xvar="lambda", lwd=2 )
beta_hat_mat = coef.glmnet( fit )[-1, ]

# lambda grid
lambda_grid = seq(0, 0.8, by = 0.02)

# mse and s_lambda
mse_vec = rep(0, length( lambda_grid ) )
s_lambda = rep(0, length( lambda_grid ) )

for( i in 1:length( lambda_grid ) ){
  fit = glmnet( y=y, x=X, family = "gaussian", intercept = FALSE, lambda = lambda_grid[i] )
  beta_hat = fit$beta
  mse_vec[i] = mean ( ( y - X %*% beta_hat )^2 )
  s_lambda[i] = p - sum( beta_hat == 0 )
}

dev.off()
pdf("/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_5/data/cp_cv_tr_error.pdf")
plot( y = mse_vec, x = lambda_grid, type = "l", lwd = 3, ylim=c(0.4, 1.3), ylab = "", xlab = "lambda", cex.axis=2, cex.lab=2 )
#points( y = mse_vec, x = lambda_grid )

lines( y = mse_vec + 2 * (s_lambda/n) * sigma2_hat, x = lambda_grid, col = "darkblue", lwd = 3 )
#points( y = mse_vec + 2 * (s_lambda/n) * sigma2_hat, x = lambda_grid, col = "darkblue" )

###############################
#     Cross validation
###############################

n_cv = 10

mse = matrix(0, ncol = length( lambda_grid ), nrow = n_cv ) 

for (i in 1:n_cv){

  ind_test = sample(1:n, replace = FALSE, size=100)
  ind_training = setdiff(1:n, ind_test)
  
  for (j in 1:length( lambda_grid ) ){
    
    fit = glmnet( y=y[ind_training], x=X[ind_training, ], family = "gaussian", intercept = TRUE, lambda = lambda_grid[j] )
    beta_hat = fit$beta
    alpha_hat = fit$a0
    
    mse[i, j] = mean ( ( y[ind_test] - alpha_hat - X[ind_test,] %*% beta_hat )^2 )
    
  }

}

moose = apply( FUN = mean, X=mse, MARGIN = 2)
moose_li = apply( FUN = min, X=mse, MARGIN = 2)
moose_ls = apply( FUN = max, X=mse, MARGIN = 2)

lines( y=moose, x = lambda_grid, lwd=3, col= "gray40")

legend("topright", legend=c("CP", "CV test error", "training error"),
       col=c("darkblue", "gray40", "black"), lwd = c(3,3,3), inset = 0.02 )
dev.off()

#points( y=moose, x = lambda_grid, pch = 1, col = "gray40")

#lines( y=moose_li, x = lambda_grid, type = "l", lwd=3, col = "gray40")
#points( y=moose_li, x = lambda_grid, pch = 1, col = "gray40")

#lines( y=moose_ls, x = lambda_grid, type = "l", lwd=3, col = "gray40")
#points( y=moose_ls, x = lambda_grid, pch = 1, col = "gray40")

