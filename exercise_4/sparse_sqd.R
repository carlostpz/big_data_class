
rm(list=ls(all=TRUE))

require(Matrix)
require(Rcpp)
require(RcppEigen)

# reading data
X = readRDS( file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/url_Xt.rds" )


# splitting in training and test sets 

# read response data
y = readRDS( file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/url_y.rds" )
n = length(y)

n_training = 1.5*(10^(6))

# test set 1
n_test_1 = 0.5*(10^(6))
#X_test_1 = X[, (n_training + 1):(n_training + n_test_1)]

# test set 2
n_test_2 = n - n_training - n_test_1
X_test_2 = X[, (n_training + n_test_1 + 1):n ]

#y_test_1 = y[ (n_training + 1):(n_training + n_test_1)]
y_test_2 = y[ (n_training + n_test_1 + 1):n ]

# Read the cpp function
sourceCpp( "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/sparse_sgd_data.cpp" )


# applying cpp function

start_time = proc.time()

fit = sgd_logistic_sparse_cpp ( n_pass = 3, 
                                y_vec = y, 
                                X_mat = X, 
                                alpha = 10^(-2),
                                lambda = 0.5,
                                eps = 10^(-6),
                                decay = 0.5 )

end_time = proc.time()

end_time - start_time

# plotting log likelihood (strange graph)
# plot( fit$log_lik[seq(1, n, by = 10)], type = "l")

beta_hat = fit$beta
hist(beta_hat)

p_hat = rep(0, n_test_2)
for (i in 1:100){
  p_hat[i] = 1 / ( 1 + exp( X_test_2[, i] %*% beta_hat ) )
}

