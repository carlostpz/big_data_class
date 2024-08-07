
rm(list=ls(all=TRUE))

require(stargazer)
require(Rcpp)
require(RcppArmadillo)
require(RcppParallel)

# reading c++ functions
sourceCpp("/home/tadeu/ut_austin/Courses/big_data/project/code/lda_vi_sgd.cpp")

set.seed(200)

# number of topics
n_topics = 20

# number of iterations in the inner loop ( loop for phi and gamma)
n_local_iter = 5

# size of the minibatch
minibatch_size = 10

# Robins-Monroe stepsize parameters, tau is delay, kappa is forgetting rate
step_t=0; tau=1; kappa=0.8 

# reading the data
original_data = as.matrix( read.table( "/home/tadeu/ut_austin/Courses/big_data/project/data/docword_nips.txt", skip = 3, as.is=TRUE ) )  

# dimension of the data matrix
D = max( original_data[,1])
N = max( original_data[,2]) 

# initializing gamma
gamma_init = abs( matrix(1 + rnorm( n = n_topics * D, mean = 0, sd = sqrt(.1) ) , nrow = D, ncol = n_topics ) )

# defining alpha (hyper parameter)
alpha = matrix( 1/n_topics , nrow = D, ncol = n_topics )

D=100

# defining eta (hyper parameter)
eta = matrix(0.01, nrow = n_topics, ncol = N )

# initializing lambda
lambda_init1 = eta + matrix( rexp( n = n_topics * N, rate = 100 * 100 /( n_topics * N ) ), nrow = n_topics, ncol = N )
lambda_init2 = eta + matrix( rexp( n = n_topics * N, rate = 100 * 100 /( n_topics * N ) ), nrow = n_topics, ncol = N )
lambda_init3 = eta + matrix( rexp( n = n_topics * N, rate = 100 * 100 /( n_topics * N ) ), nrow = n_topics, ncol = N )

D = max( original_data[,1])

# Start the clock!
ptm <- proc.time()

fit1 = lda_vi_cpp ( original_data = original_data, 
                   n_topics = n_topics, 
                   n_local_iter = n_local_iter, 
                   minibatch_size = minibatch_size,
                   step_t = step_t,
                   tau = tau,
                   kappa = kappa,
                   gamma_init = gamma_init,
                   alpha = alpha,
                   eta = eta,
                   lambda_init = lambda_init1,
                   when_calc_lik = 50,
                   n_loops_through_data = 10)

# Stop the clock
proc.time() - ptm

fit2 = lda_vi_cpp ( original_data = original_data, 
                    n_topics = n_topics, 
                    n_local_iter = n_local_iter, 
                    minibatch_size = minibatch_size,
                    step_t = step_t,
                    tau = tau,
                    kappa = kappa,
                    gamma_init = gamma_init,
                    alpha = alpha,
                    eta = eta,
                    lambda_init = lambda_init2,
                    when_calc_lik = 50,
                    n_loops_through_data = 10)


fit3 = lda_vi_cpp ( original_data = original_data, 
                    n_topics = n_topics, 
                    n_local_iter = n_local_iter, 
                    minibatch_size = minibatch_size,
                    step_t = step_t,
                    tau = tau,
                    kappa = kappa,
                    gamma_init = gamma_init,
                    alpha = alpha,
                    eta = eta,
                    lambda_init = lambda_init3,
                    when_calc_lik = 50,
                    n_loops_through_data = 10)


####################################################
#               showing results
####################################################

# log likelihood 
##################
pdf( file = "/home/tadeu/ut_austin/Courses/big_data/project/results/full_lik_init.pdf", width = 10, height = 5 )
plot ( y = fit1$log_likelihood[1,], x = seq(1, 15000, by = 50), xlim = c(1501, 15000), ylim = c(-1.25*10^7, -1.18*10^7),
       type = "l", cex.axis=1.5, cex.lab = 1.5, ylab = "Full likelihood", xlab = "Iteration", lwd = 2 )
lines( y = fit2$log_likelihood[1,], x = seq(1, 15000, by = 50), lty = 2, lwd = 2 )
lines( y = fit3$log_likelihood[1,], x = seq(1, 15000, by = 50), lty = 3, lwd = 2 )
dev.off()

fit = fit1
#pdf( file = "/home/tadeu/ut_austin/Courses/big_data/project/results/full_lik.pdf", width = 10, height = 5 )
#plot ( y = fit$log_likelihood[1,1:3000], x = seq(1, 3000, by = 50), lwd = 2,
#       type = "l", cex.axis=1.5, cex.lab = 1.5, ylab = "Full likelihood", xlab = "Iteration" )
#dev.off()

gamma = fit$gamma
lambda = fit$lambda

# theta hat
#############

# topic proportions per document

theta_hat = matrix( 0, nrow = D, ncol = n_topics )
# estimating theta by its mode
for ( d in 1:D){
  theta_hat[d, ] = ( gamma[d, ] - 1 ) / ( sum( gamma[d, ] ) - n_topics )
}


# beta hat
############

# words distribution per topic
beta_hat = matrix(0,nrow = n_topics, ncol = N )

# estimating beta by its mean
for ( k in 1:n_topics){
  beta_hat[k, ] = lambda[k, ] / sum( lambda[k, ] )
}

# score
score_matrix = matrix(0, nrow = n_topics, ncol = N)
for (k in 1:n_topics){
  for (n in 1:N){
    score_matrix[k, n] = beta_hat[k, n] * log( beta_hat[k,n] ) - beta_hat[k,n] * mean( log( beta_hat[ ,n] ) )
  }
}


# reading the vocabulary
actual_words = read.table( file = "/home/tadeu/ut_austin/Courses/big_data/project/data/nips_vocab.txt", as.is = TRUE, header = FALSE )
actual_words = actual_words[, 1]

# most frequent words in each topic
n_freq = 10

# using beta_hat as point estimate
frequent_words_beta_hat = matrix(0, ncol = n_topics, nrow = n_freq)
for (i in 1:n_topics ){
  frequent_words_beta_hat[, i] = actual_words[ order( beta_hat[i, ], decreasing = TRUE )[1:n_freq] ]
}
fix( frequent_words_beta_hat )

# using scores as point estimate
frequent_words_score = matrix(0, ncol = n_topics, nrow = n_freq)
for (i in 1:n_topics ){
  frequent_words_score[, i] = actual_words[ order( score_matrix[i, ], decreasing = TRUE )[1:n_freq] ]
}
fix( frequent_words_score )

# latex table
# stargazer( frequent_words_score[,1:10], summary=FALSE, rownames=FALSE)
stargazer( frequent_words_score[,11:20], summary=FALSE, rownames=FALSE)


