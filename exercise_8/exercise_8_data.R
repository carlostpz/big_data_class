
rm(list=ls(all=TRUE))

require(Rcpp)
require(RcppEigen)
require(Matrix)

data = read.table("/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/exercise_8_data.txt", sep = ",")
data = as.matrix(data)

data_sp = Matrix( data, sparse = TRUE )
image( data_sp )

makeD2_sparse = function (dim1, dim2)  {
  require(Matrix)
  D1 = bandSparse(dim1 * dim2, m = dim1 * dim2, k = c(0, 1), 
                  diagonals = list(rep(-1, dim1 * dim2), rep(1, dim1 * 
                                                               dim2 - 1)))
  D1 = D1[(seq(1, dim1 * dim2)%%dim1) != 0, ]
  D2 = bandSparse(dim1 * dim2 - dim1, 
                  m = dim1 * dim2, k = c(0, dim1), 
                  diagonals = list(rep(-1, dim1 * dim2), rep(1, dim1 * dim2 - 1) ) )
  return(rBind(D1, D2))
}

D = makeD2_sparse( nrow( data_sp), ncol(data_sp) )

lambda = 1
C_mat = t(D)%*%D * lambda
diag(C_mat) = diag(C_mat) + 1

y_vec = as.vector( t(data_sp) )

beta_hat = solve( a = C_mat, b = y_vec )

smooth_y = matrix( beta_hat, byrow = TRUE, nrow = nrow(data_sp), ncol = ncol(data_sp) )
smooth_y_sp = smooth_y
smooth_y_sp[ as.vector(data_sp==0) ]<-0
smooth_y_sp = Matrix( smooth_y_sp, sparse=TRUE )
  
image( smooth_y_sp )


sourceCpp("/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/exercise_8_data.cpp")

lambda = 100
C_mat = t(D)%*%D * lambda
diag(C_mat) = diag(C_mat) + 1


################################
#           Jacobi
################################

n_iter = 200
beta_hat_mat_jacobi = sparse_jacobi_cpp( A = C_mat , n_iter = n_iter, b=y_vec )
beta_hat_jacobi = beta_hat_mat_jacobi[n_iter, ]

smooth_y_jacobi = matrix( beta_hat_jacobi, byrow = TRUE, nrow = nrow(data_sp), ncol = ncol(data_sp) )
smooth_y_jacobi_sp = smooth_y_jacobi
smooth_y_jacobi_sp[ as.vector(data_sp==0) ]<-0
smooth_y_jacobi_sp = Matrix( smooth_y_jacobi_sp, sparse=TRUE )

image( smooth_y_jacobi_sp )

log_lik = rep(0, n_iter)

for (i in 1:n_iter){
  log_lik[i] = crossprod(y_vec - beta_hat_mat_jacobi[i, ] ) + lambda * crossprod( D%*%beta_hat_mat_jacobi[i, ] )
}

plot( log_lik, type="l")


################################
#         Gauss Siedel
################################

n_iter = 200
beta_hat_mat = sparse_gauss_siedel_cpp( A = C_mat , n_iter = n_iter, b=y_vec )
beta_hat_gauss_siedel = beta_hat_mat[n_iter, ]

smooth_y_gauss_siedel = matrix( beta_hat_gauss_siedel, byrow = TRUE, nrow = nrow(data_sp), ncol = ncol(data_sp) )
smooth_y_gauss_siedel_sp = smooth_y_gauss_siedel
smooth_y_gauss_siedel_sp[ as.vector(data_sp==0) ]<-0
smooth_y_gauss_siedel_sp = Matrix( smooth_y_gauss_siedel_sp, sparse=TRUE )

log_lik_gauss_siedel = rep(0, n_iter)

for (i in 1:n_iter){
  log_lik_gauss_siedel[i] = crossprod(y_vec - beta_hat_mat[i, ] ) + lambda * crossprod( D%*%beta_hat_mat[i, ] )
}

lines( log_lik_gauss_siedel, col = 2 )

image( smooth_y_gauss_siedel_sp )

################################
#             ADMM
################################

y=y_vec; D=D; rho=0.01; n_iter=10; lambda = 1

fused_lasso_admm = function( y, D, rho, n_iter, lambda ){
  
  # length of x
  dim_x = length( y )
  
  # length of z and gamma
  dim_z = nrow(D)
  dim_gamma = dim_z
  
  # cashing A
  A = t(D)%*%D * rho
  diag( A ) = diag( A ) + 1
  
  # cashing t(D)
  tD = t(D)
  
  # initializing variables
  x = y_vec
  gamma = rep(0, dim_gamma)
  z = as.numeric( D %*% x )
  
  # creating matrixes to store variables
  x_mat = matrix( 0, ncol = dim_x, nrow = n_iter )  
  gamma_mat = matrix( 0, ncol = dim_gamma, nrow = n_iter )  
  z_mat = matrix( 0, ncol = dim_z, nrow = n_iter )  
  
  log_lik = rep(0, n_iter)
  
  for (iter in 1:n_iter){
    
    # x update
    inv = sparse_jacobi_cpp( A = A, n_iter=50, b = as.numeric(y_vec - tD %*% (gamma - rho * z)) )
    x = inv[50, ]
    
    # z update
    Dx = D %*% x 
    Dx_num = as.numeric(Dx)
    z = sign( Dx_num + gamma/rho ) * pmax( abs( Dx_num + gamma/rho ) - lambda/rho , 0 )
    
    # gamma update
    gamma = gamma + rho * ( Dx_num - z )
    
    # calculate log likelihood
    log_lik[iter] = 0.5 * crossprod( y - x ) + lambda * sum( abs( Dx ) )
    
    # storing variables
    x_mat[iter, ] = x
    z_mat[iter, ] = z
    gamma_mat[iter, ] = gamma
    
  
  }
  
  return( list( "x" = x_mat,
                "z" = z_mat,
                "gamma" = gamma_mat,
                "log_lik" = log_lik  ) )

}

# applying admm method for fused lasso
n_iter = 200
fit = fused_lasso_admm( y=y_vec, D=D, rho=0.01, n_iter=n_iter, lambda = 1 )

plot( fit$log_lik, type = "l" )

y_fused_lasso = fit$x[n_iter, ]

smooth_y = matrix( y_fused_lasso, byrow = TRUE, nrow = nrow(data_sp), ncol = ncol(data_sp) )
smooth_y_sp = smooth_y
smooth_y_sp[ as.vector(data_sp==0) ]<-0
smooth_y_sp = Matrix( smooth_y_sp, sparse=TRUE )

image( smooth_y_sp )
