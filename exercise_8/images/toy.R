
rm(list=ls(all=TRUE))

require(EBImage)
require(Matrix)
require(Rcpp)
require(RcppEigen)
require(genlasso)
require(png)

# function to read image and convert to a numeric matrix in gray scale
load_image_matrix = function( file_name ){
  
# Reading Image
dt <- readImage( file_name )

colorMode(dt) = Grayscale
image_array = imageData(dt)

if( length( dim( image_array ) ) == 3 ){
  matrix_aux = apply( X = image_array, FUN = mean, MARGIN = c(1, 2) )
}else{
  matrix_aux = image_array
}

n = nrow( matrix_aux )
p = ncol( matrix_aux )

image_matrix = matrix(0, ncol = p, nrow = n)

#flipping image to the correct orientation
image_matrix[ 1:n, ] = matrix_aux[ n:1, ]
image_matrix[ ,1:p ] = matrix_aux[ ,p:1 ]

return( image_matrix )

}

# function for image smoothing
smooth_image_function = function( lambda, jacobi_iter, file_name ){
  
  # loading jacobi iterative routine for inverting sparse matrices
  sourceCpp("/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/exercise_8_data.cpp")
  
  # loading image from disk
  image_matrix = load_image_matrix( file_name = file_name ) 
  
  # adjacency matrix
  D = getD2dSparse( dim2 = nrow( image_matrix ), dim1 = ncol( image_matrix ) )
  
  y_vec = as.vector( t(image_matrix) )
  
  C_mat = t(D)%*%D * lambda
  diag(C_mat) = diag(C_mat) + 1
  
  n_iter = jacobi_iter
  beta_hat_mat_jacobi = sparse_jacobi_cpp( A = C_mat, n_iter = n_iter, b=y_vec )
  beta_hat_jacobi = beta_hat_mat_jacobi[n_iter, ]
  
  smooth_y_jacobi = matrix( beta_hat_jacobi, byrow = TRUE, nrow = nrow(image_matrix), ncol = ncol(image_matrix) )
  
  return( smooth_y_jacobi )
  
}

# forming the grayscale matrix
y_mat = load_image_matrix( file_name = '/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/toy/images/tt.jpg' )

# take a look at the image. how cool is it? =)
# image( y_mat, col = paste("gray",1:99,sep="") )
# dev.off()

# building noisy image
n = nrow( y_mat )
p = ncol( y_mat )
random_noise = matrix( rnorm(n*p, mean = 0, sd = 0.3) , ncol = p, nrow = n)
y_noise = y_mat + random_noise

# ploting noisy image
#writePNG( image = t(y_noise)[n:1,], 
#          target = '/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/toy/images/tt_noisy.png' )
png('/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/toy/images/tt_noisy.png', 
    width = nrow(y_noise), height = ncol(y_noise))
op <- par(mar = rep(0, 4))
image( y_noise, col = paste("gray",1:99,sep="") )
dev.off()


####################################
#           lambda = 1
####################################

# smoothing low noise image
smoothed_image = smooth_image_function ( lambda = 1, 
                                         file_name ='/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/toy/images/tt_noisy.png', 
                                         jacobi_iter = 100 )

png('/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/toy/images/tt_smooth_lambda_1.png',
    width = nrow(smoothed_image), height = ncol(smoothed_image) )
op <- par(mar = rep(0, 4))
image( smoothed_image , col = paste("gray",1:99,sep="") )
dev.off()


####################################
#           lambda = 5
####################################

# smoothing low noise image
smoothed_image = smooth_image_function ( lambda = 5, 
                                         file_name = '/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/toy/images/tt_noisy.png',
                                         jacobi_iter = 100 )

png('/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/toy/images/tt_smooth_lambda_5.png',
    width = nrow(smoothed_image), height = ncol(smoothed_image) )
op <- par(mar = rep(0, 4))
image( smoothed_image , col = paste("gray",1:99,sep="") )
dev.off()



####################################
#           lambda = 10
####################################

# smoothing low noise image
smoothed_image = smooth_image_function ( lambda = 10, 
                                         file_name ='/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/toy/images/tt_noisy.png',
                                         jacobi_iter = 100 )
png('/home/tadeu/ut_austin/Courses/big_data/exercises/exercise_8/toy/images/tt_smooth_lambda_10.png',  
    width = nrow(smoothed_image), height = ncol(smoothed_image) )
op <- par(mar = rep(0, 4))
image( smoothed_image , col = paste("gray",1:99,sep="") )
dev.off()
