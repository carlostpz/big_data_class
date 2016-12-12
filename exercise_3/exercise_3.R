
############################################
#          Exercise 3 - SGD
############################################

rm(list=ls(all=TRUE))

# simulated data size
n = 1000
# number of covariates
p = 4

# true covariate effects
beta_real = c(1, 0.5, -0.5, -1)

# each covariate value is drawn from N(0, 1)
X = matrix(0, ncol = p, nrow = n)
for (i in 1:n){
  for (j in 1:p){
    X[i,j] = rnorm(1)
  }
}

# true probabilities of succes
probs_real = 1 / (1 + exp( - X %*% beta_real ) )

# simulated response
y = rep(0, n)
for (i in 1:n){
  y[i] = rbinom(n=1, size = 1, prob = probs_real[i] )
}

############
#  Item c
############

n_iter = 10000

# calculates the negative log likelihood of ( y[index], X[index, ])
neg_log_lik_function = function( beta, ... ){
  
  # User must specify X=, index= and y= in any order   
  # Index reffers to previously sampled index from 1:n
  
  parameters = list(...)
  
  X = parameters$X
  y = parameters$y
  index = parameters$index
  
  return( as.numeric( (1 - y[index] ) * X[index, ] %*% beta ) + 
            + as.numeric( log ( 1 + exp( -X[index, ] %*% beta ) ) )  )
  
}

# Calculates gradient based on one observation only (does not include rescale)
gradient_function = function ( beta, ... ){
  
  # User must specify X=, index= and y= in any order   
  # Index reffers to previously sampled index from 1:n
  
  parameters = list(...)
  
  X = parameters$X
  y = parameters$y
  index = parameters$index
  
  w_i = 1/( 1 + exp( - crossprod( X[index, ], beta ) ) )
  gradient = ( w_i - y[index]) * X[index, ]
  
  return( gradient )
  
}


# Calculates gradient based on one observation only (does not include rescale)
direction_function = function ( beta, ... ){
  
  # User must specify X=, index= and y= in any order   
  # Index reffers to previously sampled index from 1:n
  
  parameters = list(...)
  
  X = parameters$X
  y = parameters$y
  index = parameters$index
  
  w_i = 1/( 1 + exp( - crossprod( X[index, ], beta ) ) )
  gradient = ( w_i - y[index]) * X[index, ]
  
  return( -gradient )
  
}


# Backtracking line search function
line_search = function ( f, beta_init, alpha_bar, rho, gradient_function, direction, cc, ... ){
  
  # f -> function of argument beta to be optimized
  # direction -> function of beta that calculates the direction we want to move through
  # ... -> additional arguments to be passed to the function f() and direction()
  
  beta = beta_init
  alpha = alpha_bar
  
  beta_new = beta + alpha * direction( beta_init, ... )
  
  print( c(f( beta_new, ... ) -( f( beta, ... ) + 
             cc * alpha * crossprod( gradient_function( beta, ... ) , direction( beta, ... ) ) ) ) )
  
  while( f( beta_new, ... ) > f( beta, ... ) + 
         cc * alpha * crossprod( gradient_function( beta, ... ) , direction( beta, ... ) ) ){
    
    alpha = rho * alpha
    beta = beta_new
    beta_new = beta + alpha * direction( beta_init, ... )
    
  }
  
  output = list( "alpha" = alpha, "beta" = beta_new )
  
  return( output )
  
}

# Implementing SGD

sgd_logistic = function(n_iter, y, X, last_iter, rho, cc, alpha0){
  
  # initial guess
  beta = rep(0, p)
  beta_vec = matrix(0, nrow = n_iter, ncol = p)
  
  # initializing negative log likelihood
  neg_log_lik_vec = rep(0, n_iter)
  
  for (iter in 1:n_iter){
    
    # sampling a data point
    index = sample( x = 1:n, size = 1 )
    
    # calculating the noisy gradient based on the previously sampled point (rescaling also)
    gradient = n * gradient_function(y=y, X=X, index = index, beta = beta) 
    
    # calculating the negative log likelihood for the sampled observation
    neg_log_lik = neg_log_lik_function( y=y, X=X, beta=beta, index = index)
    
    # Line search
    alpha_beta = line_search( f = neg_log_lik_function, 
                              beta_init = beta, 
                              alpha_bar = alpha0, 
                              rho = rho, 
                              direction = direction_function, 
                              gradient_function = gradient_function,
                              cc = cc, 
                              X = X,
                              y = y,
                              index = index)
    
    print(iter)
    
    # Updaing alpha and beta by the line search output
    beta = alpha_beta$beta
    alpha = alpha_beta$alpha
    
    # storing beta
    beta_vec[iter, ] = beta
    
  }
  
  # using the last l values to estimate beta
  l = last_iter
  fitted_beta = apply( X = beta_vec[ (n_iter - l + 1):n_iter, ], FUN = mean, MARGIN = 2)
  
  return( list( "fitted_beta"=fitted_beta, "all_betas"=beta_vec ) )
  
}

# applying the function
fit = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=5000, alpha0 = 10^(-1), cc=10^(-4), rho=0.9)

# beta values per iteration
plot( fit$all_betas[,1], type="l")
plot( fit$all_betas[,2], type="l")
plot( fit$all_betas[,3], type="l")
plot( fit$all_betas[,4], type="l")

# answer obteined after averaging the last 1000 values of beta
fit$fitted_beta

# comparing with results from the glm function
glm ( y ~ X - 1 , family = "binomial")$coef


