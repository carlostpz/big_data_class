
// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/SparseCholesky>

using namespace Rcpp;
using namespace Eigen;

// Define shortcut for Sparse Matrix and its iterator
typedef Eigen::MappedSparseMatrix<double> SpMat;
typedef SpMat::InnerIterator InnerIterMat;

// Define shortcut for Sparse Vector and its iterator
typedef Eigen::SparseVector<double> SpVec;
typedef SpVec::InnerIterator InnerIterVec;

//[[Rcpp::export]]
Eigen::MatrixXd sparse_jacobi_cpp ( int n_iter,
                                    SpMat A,
                                    Eigen::VectorXd b ){

    // solves t(A)x = b for sparse matrix A through jacobi iterative inversion algorithm
    // Matrix A has to be symmetric
    
    // Number of covariates and observations
    int n = A.cols();
    
    // initial guess
    Eigen::VectorXd x(n);  x.fill(0);
    Eigen::MatrixXd x_mat(n_iter, n);
    double x_aux;
    int j;
    double A_ii = 1;
    double A_ij = 1;
    SpVec A_col_i;
    
    for ( int iter=0; iter<n_iter; iter++ ){
      
      // pointwise construction of x
      for ( int i=0; i<n; i++){
      
        x_aux = 0;
        A_col_i = A.innerVector(i);
        
        for ( InnerIterVec iterator( A_col_i ); iterator; ++iterator ){
           
          j = iterator.index();
          A_ij = iterator.value();
          if(  j != i ){
            x_aux = x_aux - A_ij * x(j);
          }else{
            A_ii = A_ij;
          }
          
        }
      // Finishing x_i
      x(i) = ( x_aux + b(i) )/A_ii;
         
    } 
    x_mat.row( iter ) = x;
  }
    
return x_mat;
    
}



//[[Rcpp::export]]
Eigen::MatrixXd sparse_gauss_siedel_cpp ( int n_iter,
                                          SpMat A,
                                          Eigen::VectorXd b ){
        
  // solves t(A)x = b for sparse matrix A through jacobi iterative inversion algorithm
  // Matrix A has to be symmetric
  
  // Number of covariates and observations
  int n = A.cols();
  
  // initial guess
  Eigen::MatrixXd x_mat(n_iter, n);
  double x_aux;
  Eigen::VectorXd x_old(n); x_old.fill(0);
  Eigen::VectorXd x_new(n); x_new.fill(0);
  int j;
  double A_ii = 1;
  double A_ij = 1;
  SpVec A_col_i;
  
  for ( int iter=0; iter<n_iter; iter++ ){
    
    // pointwise construction of x
    for ( int i=0; i<n; i++){
      
      x_aux = 0;
      A_col_i = A.innerVector(i);
      
      for ( InnerIterVec iterator( A_col_i ); iterator; ++iterator ){
        
        j = iterator.index();
        A_ij = iterator.value();
        if(  j != i ){
          x_aux = x_aux - A_ij * x_old(j);
        }else{
          A_ii = A_ij;
        }
        
      }
      // Finishing x_i
      x_new(i) = ( x_aux + b(i) )/A_ii;
      
    } 
    x_mat.row( iter ) = x_new;
    x_old = x_new;
  }
  
  return x_mat;
  
}

//[[Rcpp::export]]
SparseMatrix<double> sparse_inverse_cpp ( SpMat A ){

// Inverts sparse matrix A
  
int n = A.cols(); 
SimplicialLLT<SparseMatrix<double> > solver;
solver.compute(A);
SparseMatrix<double> I(n,n);
I.setIdentity();
SparseMatrix<double> A_inv = solver.solve(I);
  
return A_inv;

}  
  
