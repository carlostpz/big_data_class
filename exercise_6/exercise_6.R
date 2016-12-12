
rm(list=ls(all=TRUE))

##########################################
#      Application to the dataset
##########################################

require(glmnet)
set.seed(100)

X = read.csv("/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_5/data/diabetesX.csv", header = TRUE )
y = read.table("/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_5/data/diabetesY.txt", sep=";", header = FALSE )
y = as.numeric( y )
y = (y-mean(y))/sd(y)
X = as.matrix( X )

n = length(y)
p = ncol(X)

#########################
#   proximal gradient
#########################

lasso_prox_gradient = function( y, X, gamma, n_iter, lambda ){

Xt = t(X)
Xt_X = Xt %*% X

beta = solve(Xt_X) %*% Xt %*% y

beta_mat = matrix( 0, ncol = p, nrow = n_iter )  
log_lik = rep(0, n_iter)

for (iter in 1:n_iter){
  
  u = beta - gamma * ( - Xt %*% y + Xt_X %*% beta  )
  aux = abs(u) - gamma * lambda 
  beta = sign(u) * I( aux > 0 ) * aux
  beta_mat[iter, ] = beta
  
  log_lik[iter] = crossprod( y, X%*%beta ) + lambda * sum( abs( beta ) )

}

return( list( "beta" = as.numeric( beta ),
              "coefs" = beta_mat,
              "log_lik" = log_lik  ) )

}

fit = lasso_prox_gradient( y=y, X=X, gamma = 0.1, lambda = 0.5, n_iter = 2000)

# least squares
as.numeric( solve( t(X) %*% X ) %*% t(X) %*% y )

plot( fit$log_lik, type = "l")

fit_glmnet = glmnet( y = y, x = X, family = "gaussian", standardize = FALSE, lambda = 0.5/(2*n), intercept = FALSE )
beta_lasso = fit_glmnet$beta

#as.numeric(beta)
#as.numeric(beta_lasso)

####################################
#   accelerated proximal gradient
####################################

accelerated_lasso_prox_gradient = function( y, X, gamma, n_iter, lambda ){
  
  p = ncol(X)
  zeroes = rep(0, p)
  
  Xt = t(X)
  Xt_X = Xt %*% X
  
  #z = solve(Xt_X) %*% Xt %*% y
  z = zeroes
  beta_old = z
  s_old = 1
  
  beta_mat = matrix( 0, ncol = p, nrow = n_iter )  
  log_lik = rep(0, n_iter)
  
  for (iter in 1:n_iter){
    
    u = z - gamma * ( - Xt %*% y + Xt_X %*% z  )
    beta_new = sign(u) * pmax( abs(u) - gamma * lambda, zeroes) 
    
    s_new = ( 1 + sqrt( 1 + 4 * (s_old^2) ) ) / 2

    z = beta_new + (s_old-1) / s_new * (beta_new - beta_old)
    
    beta_mat[iter, ] = beta_new
    
    log_lik[iter] = crossprod( y - X%*%beta_new ) + lambda * sum( abs( beta_new ) )
    
    # updating beta and s
    
    beta_old = beta_new
    s_old = s_new
    
  }
  
  return( list( "beta" = as.numeric( beta_new ),
                "coefs" = beta_mat,
                "log_lik" = log_lik  ) )
  
}


# Applying the function
fit = accelerated_lasso_prox_gradient( y=y, X=X, gamma = 0.001, lambda = 0.001, n_iter = 200)

# comparing with glmnet
fit_glmnet = glmnet( y = y, x = X, family = "gaussian", standardize = FALSE, lambda = 0.00001, intercept = FALSE )
beta_lasso = fit_glmnet$beta

# convergence?
plot( fit$log_lik, type = "l")
