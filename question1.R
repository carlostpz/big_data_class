
rm(list=ls(all=TRUE))

set.seed(100)

require(Matrix)
require(KFAS)
require(rbenchmark)

#########
# Item c
#########

ldl_lm = function( X, y){
  
  A = crossprod(X)
  b = crossprod(X, y)
  aux = ldl( A )
  L = aux
  diag(L) = 1
  inv_diag = 1/diag( aux )
  c_vec = forwardsolve( l = L, x = b) 
  beta_hat = backsolve( r = t(L), x = inv_diag * c_vec  )
  
  return(beta_hat)
  
}

regular_lm = function (X, y){
  return( solve( crossprod( X ) ) %*% crossprod(X, y) )
}

n=10000
p_vec = c(10, 100, 500, 1000, 2000)

for (i in 1:5){
  
  p = p_vec[i]
  
  beta = runif( p )
  X = matrix( rnorm(n*p) , ncol = p, nrow = n )
  y = X %*% beta + rnorm( n, mean = 0, sd = 0.2 )
  W = diag(n)

  print(i)
  print ( benchmark( ldl_lm( X=X, y=y ), regular_lm( X=X, y=y ), 
                     replications = 1, columns = c("test", "elapsed", "relative" ) ) )
  
}


##########
# Item d
##########

lm_sparse = function( X, y){
  
  X = Matrix( X, sparse = TRUE )
  A = crossprod(X)
  b = crossprod(X, y )
  beta_hat = solve( a = A, b = b )
  
  return(beta_hat)
  
}

# Creating sparse design matrix
n = 2000
p = 500
X = matrix(rnorm(n*p), nrow=n)
mask = matrix(rbinom(n*p,1,0.05), nrow=n, ncol=p)
X = mask*X
X[1:10, 1:10] # quick visual check

# Simulating data
beta = runif( p )
y = X %*% beta + rnorm( n, mean = 0, sd = 0.2 )
W = diag(n)

# benchmark
print ( benchmark( ldl_lm( X=X, y=y ), regular_lm( X=X, y=y ), ldl_lm_sparse( X=X, y=y ),
                   replications = 1, columns = c("test", "elapsed", "relative" ) ) )

