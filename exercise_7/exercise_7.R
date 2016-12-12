
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
#         ADMM
#########################

lasso_admm = function( y, X, rho, n_iter, lambda ){
  
  Xt = t(X)
  Xt_X = Xt %*% X
  
  p = ncol(X)
  Id = diag(p)
  
  # initializing variables
  beta = solve(Xt_X) %*% Xt %*% y
  #beta = rep(0, p)
  gamma = rep(0, p)
  z = beta
  
  # creating matrixes to store variables
  beta_mat = matrix( 0, ncol = p, nrow = n_iter )  
  gamma_mat = matrix( 0, ncol = p, nrow = n_iter )  
  z_mat = matrix( 0, ncol = p, nrow = n_iter )  
  
  log_lik = rep(0, n_iter)
  
  mat_inv = solve( 2*Xt_X + rho * Id )
  
  for (iter in 1:n_iter){
    
    # update for beta
    beta = mat_inv %*% ( 2 * Xt %*% y - gamma + rho * z ) 
    
    #update for z
    z = sign( beta + gamma/rho) * pmax( abs( beta + gamma/rho) - lambda/rho , 0 )
    
    #update for gamma
    gamma = gamma + rho*(beta - z)
    
    # calculate log likelihood
    log_lik[iter] = crossprod( y - X %*% beta ) + lambda * sum( abs( beta ) )
    
    # storing variables
    beta_mat[iter, ] = beta
    z_mat[iter, ] = z
    gamma_mat[iter, ] = gamma
    
  }
  
  return( list( "beta" = beta_mat,
                "z" = z_mat,
                "gamma" = gamma_mat,
                "log_lik" = log_lik  ) )
  
}

fit = lasso_admm( y = y, X = X, lambda = 0.01, n_iter = 200, rho = 0.01)

plot(fit$log_lik, type = "l")

