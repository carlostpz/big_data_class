
rm(list=ls(all=TRUE))

require(Matrix)
require(Rcpp)
require(RcppEigen)

# number of observations in the first test set
n_test_1 = 279226

# reading design matrix (randomized split)
X_training = readRDS( file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/scaled_randomized_split_transposed/url_Xt_training.rds" )
X_test_1 = readRDS( file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/scaled_randomized_split_transposed/url_Xt_test_1.rds" )
X_test_2 = readRDS( file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/scaled_randomized_split_transposed/url_Xt_test_2.rds" )

y_training = readRDS( file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/scaled_randomized_split_transposed/url_y_training.rds" )
y_test_1 = readRDS( file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/scaled_randomized_split_transposed/url_y_test_1.rds" )
y_test_2 = readRDS( file = "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/scaled_randomized_split_transposed/url_y_test_2.rds" )


# Read the cpp function
sourceCpp( "/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_4/data/sparse_sgd_data.cpp" )

# applying cpp function

start_time = proc.time()

fit = sgd_logistic_sparse_cpp ( n_pass = 1, 
                                y_vec = y_training, 
                                X_mat = X_training, 
                                alpha = 0.1,
                                lambda = 10^(-6),
                                eps = 10^(-4),
                                decay = 0.99 )

end_time = proc.time()

end_time - start_time

# plotting log likelihood (strange graph)
plot( fit$log_lik[seq(1, length(y_training), by = 100)], type = "l")

beta_hat = fit$beta
alpha_hat = fit$alpha

fit_pred = logistic_prediction( X_test = X_test_1, beta_hat = beta_hat, alpha_hat = alpha_hat )

y_hat = fit_pred$y_hat
p_hat = fit_pred$p_hat

hist( p_hat )

mean( y_hat == y_test_1 )

plot( p_hat[1:100], type = "o" )
points( y = y_test_1[ 1:100 ], x=1:100, pch=19 )
abline( h=0.5, lty=2, lwd=2 )

