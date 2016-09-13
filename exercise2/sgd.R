
############################################
#          Exercise 2 - SGD
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

# Implementing SGD

sgd_logistic = function(n_iter, y, X, last_iter, init_step, delay, learning_rate){

# initial guess
beta = rep(0, p)
beta_vec = matrix(0, nrow = n_iter, ncol = p)

# initializing negative log likelihood
neg_log_lik_vec = rep(0, n_iter)

# initializing the length of the gradient at each iteration
grad_len = rep(0, n_iter)

# delay
t0 = delay

# learning rate
kappa = learning_rate

# Initial stp size
k = init_step

for (iter in 1:n_iter){
  
  # sampling a data point
  index = sample( x = 1:n, size = 1 )
  
  # calculating the noisy gradient based on the previously sampled point
  w_i = 1/( 1 + exp( - crossprod( X[index, ], beta ) ) )
  gradient = n * ( w_i - y[index]) * X[index, ]
  
  # step size
  alpha = k * 1/( ( t0 + iter )^kappa )
  
  # main step: SGD update
  beta = beta - alpha * gradient
  
  # storing beta
  beta_vec[iter, ] = beta
  
  # length of the gradient
  grad_len[iter] = sqrt( sum( gradient^2 ) )
  
}

# using the last l values to estimate beta
l = last_iter
fitted_beta = apply( X = beta_vec[ (n_iter - l + 1):n_iter, ], FUN = mean, MARGIN = 2)

return( list( "fitted_beta"=fitted_beta, "gradient_length"=grad_len, "all_betas"=beta_vec ) )

}

# applying the function
fit = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=1000, init_step=0.1, delay=2, learning_rate=0.8)

# Beta values per iteration
plot( fit$all_betas[,1], type="l")
plot( fit$all_betas[,2], type="l")
plot( fit$all_betas[,3], type="l")
plot( fit$all_betas[,4], type="l")

# length of the gradient goes to zero?  Answer: NO
# plot( fit$gradient_length, type = "l" )

# last value obtained for beta
fit$all_betas[n_iter, ]

# answer obteined after averaging the last 1000 values of beta
fit$fitted_beta

# comparing with results from the glm function
glm ( y ~ X - 1 , family = "binomial")$coef

######################################
#           Comparisons
######################################

# Different learning_rates
fit1 = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=1000, init_step=0.1, delay=2, learning_rate=0.5)
fit2 = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=1000, init_step=0.1, delay=2, learning_rate=0.8)
fit3 = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=1000, init_step=0.1, delay=2, learning_rate=1.0)
fit4 = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=1000, init_step=0.1, delay=2, learning_rate=1.2)

fit1$fitted_beta
fit2$fitted_beta
fit3$fitted_beta
fit4$fitted_beta


# Different initial stepsize
fit1 = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=1000, init_step=0.01, delay=2, learning_rate=0.8)
fit2 = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=1000, init_step=0.1, delay=2, learning_rate=0.8)
fit3 = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=1000, init_step=1, delay=2, learning_rate=0.8)
fit4 = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=1000, init_step=10, delay=2, learning_rate=0.8)

fit1$fitted_beta
fit2$fitted_beta
fit3$fitted_beta
fit4$fitted_beta


######################################
#   Applying sgd to the real data
######################################

data = read.table("/home/tadeu/ut_austin/Courses/big_data/exercises/data_ex1.txt", sep=",", as.is=TRUE, header=FALSE)
y = data[,2]
y[ which( y == "M" ) ] = 1
y[ which( y == "B" ) ] = 0
y = as.numeric(y)

X = as.matrix( data[, 3:12] )

for (column in 1:ncol(X)){
  X[, column] = ( X[, column] - mean( X[, column] ) ) / sd ( X[, column] )
}


n = nrow(X)
X = cbind( rep(1, n), X)

p = ncol(X)

# applying the function
fit = sgd_logistic(n_iter=10000, y=y, X=X, last_iter=1000, init_step=0.01, delay=2, learning_rate=0.8)

# Beta values per iteration
plot( fit$all_betas[,1], type="l")
plot( fit$all_betas[,2], type="l")
plot( fit$all_betas[,3], type="l")
plot( fit$all_betas[,4], type="l")

# length of the gradient goes to zero?  Answer: NO
# plot( fit$gradient_length, type = "l" )

# last value obtained for beta
fit$all_betas[n_iter, ]

# answer obteined after averaging the last 1000 values of beta
fit$fitted_beta

# comparing with results from the glm function
glm ( y ~ X - 1 , family = "binomial")$coef

