
rm(list=ls(all=TRUE))

data = read.table("/home/tadeu/ut_austin/Courses/big_data/exercises/data_ex1.txt", sep=",", as.is=TRUE, header=FALSE)
y = data[,2]
y[ which( y == "M" ) ] = 1
y[ which( y == "B" ) ] = 0
y = as.numeric(y)

X = as.matrix( data[, 3:12] )

for (column in 1:ncol(X)){
  X[, column] = ( X[, column] - mean( X[, column] ) ) / sd ( X[, column] )
}

n = nrow( X )
uns = rep(1, n)
X = cbind(uns, X)

n = nrow(X)
p = ncol(X)
n_iter = 20000
beta = rep(0, p)

alpha = 0.01
uns = rep(1, n)


##########
# Item b
##########

beta_vec = matrix(0, nrow = n_iter, ncol = p)
neg_log_lik_vec = rep(0, n_iter)

for (iter in 1:n_iter){
  w = 1/( 1 + exp( - X %*% beta ) )
  gradient = t( crossprod( w - y, X ) )
  beta = beta - alpha * gradient
  beta_vec[iter, ] = beta
  neg_log_lik = as.numeric( t( uns - y) %*% X %*% beta ) + sum( log ( 1 + exp( -X %*% beta ) ) )
  neg_log_lik_vec[iter] = neg_log_lik
}

plot( beta_vec[,11], type="l")
plot( neg_log_lik_vec, type="l", ylim=c(70, 80) )

beta_vec[ n_iter, ]

glm ( y ~ X - 1 , family = "binomial")$coef

##########
# Item d
##########

beta = rep (0, p)
beta_mat = matrix(0, nrow = n_iter, ncol = p)
gradient_len = c(0, n_iter)

n_iter = 10

for (iter in 1:n_iter){
  w = 1/( 1 + exp( - X %*% beta ) )
  gradient = t( crossprod( w - y, X ) )
  hessian = t(X) %*% diag( as.vector( w * (1-w) ) ) %*% X
  beta = beta - solve( hessian ) %*% gradient
  beta_mat[iter, ] = beta
  gradient_len[iter] = sqrt( sum( gradient^2 ) )
}

plot(gradient_len, type = "l")

beta_mat[ n_iter, ]
glm ( y ~ X - 1 , family = "binomial")$coef

