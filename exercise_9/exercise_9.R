
rm(list=ls(all=TRUE))


sparse_matrix_decomp = function( X, c1, c2, tol, K){
  
  # function for binary search
  binary_search = function( a, epsilon, c){
    
    right = .99999 * max( abs(a) ) 
    left = 0
    
    delta = (right + left)/2
    num = sign(a) * pmax( abs(a) - delta, 0 )
    den = sqrt( sum( num^2 ) ) 
    
    u = num/den
    l1_norm_u = sum( abs(u) )
    
    # while delta is not close enough to c...
    while (  abs( l1_norm_u - c ) > epsilon ){
      
      if( l1_norm_u < c){
        right = delta
      }else{
        left = delta
      }
      
      delta = (right + left)/2
      num = sign(a) * pmax( abs(a) - delta, 0 )
      den = sqrt( sum( num^2 ) ) 
      
      u = num/den
      l1_norm_u = sum( abs(u) )
      
    }
    
    return( u )
    
  }
  
  #######################
  # algorithm for rank 1
  #######################
  
  smd_rank1 = function( X, c1, c2, tol){
  
  n = nrow(X)
  p = ncol(X)
    
  # initiate v
  v_old = rep( 1/ sqrt(p), p )
  u_old = rep( 1/ sqrt(n), n )
  
  # sum( (X - u_old %*% t(v_old) )^2 )
  
  # the purpose of this is to initiate the next loop 
  v_new = v_old
  u_new = u_old
  
  first_iteration = TRUE
  
  while ( max( abs(u_new - u_old ) ) > tol |  max( abs(v_new - v_old ) ) > tol | first_iteration ){ 
    
    # this will make sense once the first loop is completed
    u_old = u_new  # we do not need this line, actually
    v_old = v_new
    
    X_times_v = X %*% v_old
    
    # SPC
    #u_new = X_times_v / sqrt( sum( X_times_v^2 ) )
    
    #binary search for u
    if ( sum( abs( X_times_v / sqrt( sum( X_times_v^2 ) ) ) ) > c1 ){
      
      u_new = binary_search( a = X_times_v, epsilon = tol, c = c1)
      
    }else{ 
      u_new = X_times_v 
    }
    
    Xt_times_u = crossprod( X, u_new)
    
    # binary search for v
    if ( sum( abs( Xt_times_u / sqrt( sum( Xt_times_u^2 ) ) ) ) > c2 ){
      
      v_new = binary_search( a = Xt_times_u, epsilon = tol, c = c2)
      
    }else{
      v_new = Xt_times_u
    }
    
#    print( c( max( abs(u_new - u_old)) , max(abs(v_new - v_old) ) ) )
    
    first_iteration = FALSE
  
  }
  
  # picking u and v calculated in the previous loop
  u = u_new
  v = v_new
  
  # calculating the diagonal of the matrix D
  d = crossprod( u, X) %*% v
  
  return( list("u" = u, "v" = v, "d" = as.numeric(d) ) )
  
  }
  
  n = nrow(X)
  p = ncol(X)
  
  U = matrix( 0, nrow = n, ncol = K )
  V = matrix( 0, nrow = p, ncol = K )
  D = matrix (0, nrow = K, ncol = K )
  
  X_residual = X
  
  for (rank_iter in 1:K){
    
    result = smd_rank1( X = X_residual, c1 = c1, c2 = c2, tol = tol )
    
    u = result$u
    v = result$v
    d = result$d
    
    U[, rank_iter] = u
    V[, rank_iter] = v
    D[ rank_iter, rank_iter] = d
    
    X_residual = X_residual - d * u %*% t(v)
    
    print(rank_iter)
  
  }

return( list( "U" = U, "V" = V, "D" = D ) )

}

##############################
#         Simulation
##############################

n = 30
p = 5

X = matrix( runif(n*p), ncol = p, nrow = n )


###############################
#   Applying to simulation
###############################

c1 = 1.1; c2 = 1.1
tol = 10^(-4)
K = 5

ans = sparse_matrix_decomp( X, c1, c2, tol, K )

U = ans$U
V = ans$V
D = ans$D
  
X_hat = U %*% D %*% t(V)

plot( x = as.vector(X), y = as.vector(X_hat) )

sum( (X - X_hat)^2 )

########################################
#    Application to the real data
########################################

dataset = read.table("/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_9/marketing.csv", sep = ",", header = TRUE )
dataset = dataset[, -1]
dataset = as.matrix(dataset)
X = scale( dataset )

c1 = 1; c2 = 1
K = 36

fit = sparse_matrix_decomp( X = crossprod(X), c1 = 1.1, c2 = 1.1, tol = 10^(-9), K = K )

U = fit$U
V = fit$V
D = fit$D

# crossprod(U, U)

cov_hat = U %*% D %*% t(V)

plot( crossprod(X, X), cov_hat )
abline( 0, 1 )
