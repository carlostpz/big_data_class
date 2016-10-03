
// [[Rcpp::depends(RcppEigen)]]

#include <Rcpp.h>
#include <RcppEigen.h>

using namespace Rcpp;
using namespace Eigen;

// Define shortcut for Sparse Matrix and its iterator
typedef Eigen::MappedSparseMatrix<double> SpMat;
typedef SpMat::InnerIterator InnerIterMat;

// Define shortcut for Sparse Vector and its iterator
typedef Eigen::SparseVector<double> SpVec;
typedef SpVec::InnerIterator InnerIterVec;


//double dv_times_sv_alt( Eigen::VectorXd dv, SpMat sv){
//  
//  double ans = 0;
//  for (InnerIterMat iterator( sv, 0 ); iterator; ++iterator ){ // ++ has to come before iterator
//    ans = ans + iterator.value() * dv.coeff( iterator.index(), 0);
//  }
//  
// return ans;
//  
//}

//double dv_times_sv( Eigen::VectorXd dv, SpVec sv){
//  
//  double ans = 0;
//  for (InnerIterVec iterator( sv); iterator; ++iterator ){ // ++ has to come before iterator
//    ans = ans + iterator.value() * dv.coeff( iterator.index() );
//  }
//  
//  return ans;
//  
//}


//Eigen::VectorXd dv_plus_sv_alt( Eigen::VectorXd dv, SpMat sv){
//  
//  Eigen::VectorXd ans = dv;
//  
//  for (InnerIterMat iterator( sv, 0 ); iterator; ++iterator ){ // ++ has to come before iterator
//    ans( iterator.index(), 0 ) =  iterator.value() + dv.coeff( iterator.index(), 0);
//  }
//  
//  return ans;
//  
//}

//Eigen::VectorXd dv_plus_sv( Eigen::VectorXd dv, SpVec sv){
//  
//  Eigen::VectorXd ans = dv;
//  
//  for (InnerIterVec iterator( sv ); iterator; ++iterator ){ // ++ has to come before iterator
//    ans( iterator.index() ) =  iterator.value() + dv.coeff( iterator.index() );
//  }
//  
//  return ans;
//  
//}

// function that calculates 1/sqrt(x)
inline double invSqrt( const double& x ) {
  double y = x;
  double xhalf = ( double )0.5 * y;
  long long i = *( long long* )( &y );
  i = 0x5fe6ec85e7de30daLL - ( i >> 1 );//LL suffix for (long long) type for GCC
  y = *( double* )( &i );
  y = y * ( ( double )1.5 - xhalf * y * y );
  
  return y;
}

// calculates the neg log likelihood for one observation (y, x)
//double eval_neg_log_lik( Eigen::VectorXd beta, SpVec x, double y){
//  
//  double Xbeta = dv_times_sv( beta, x );
//  return -1 * y * Xbeta + log( 1 + exp( Xbeta ) );
//    
//}

//[[Rcpp::export]]
Rcpp::List sgd_logistic_sparse_cpp ( int n_pass, 
                                          Eigen::VectorXd y_vec, 
                                          SpMat X_mat, 
                                          double lambda,
                                          double alpha,
                                          double eps,
                                          double decay){

  // Number of covariates and observations
  int p = X_mat.rows();
  int n = X_mat.cols();
  
  // initial guess
  Eigen::VectorXd beta(p);
  
  // initializing log likelihood
  double log_lik;
  double cumu_log_lik;
  Eigen::VectorXd log_lik_vec( n * n_pass );
  
  // initializing sum of squared gradients
  Eigen::VectorXd sum_sq_gi( p );
  
  // indexing the observations
  int index_obs=0;
  SpVec x(p);
  double y ;
  
  // calculating current gradient
  Eigen::VectorXd gradient(p); 
  
  int i_index;
  double gradient_i_index;
  double x_i;
  double beta_i;
  double w_i;
  double g_i;
  double Xbeta;
  
  // when was the last time we updated beta_i?
  Eigen::VectorXd last_time_beta(p); 
  
  
  for ( int loop=0; loop < n_pass; loop++){

    for ( int iter=0; iter < n; iter++){
      
      x = X_mat.innerVector(iter);
      y = y_vec(iter);

      // updating beta ( needs to be done entry by entry )
      // notice that x_j = 0 => beta_j = 0, then we only update beta_j s.t. x_j != 0 
      for ( InnerIterVec i_(x); i_; ++i_ ){
        
        // non zero index and value of x
        i_index = i_.index();
        x_i = i_.value();
        
        // corresponding entry of beta and w
        beta_i = beta( i_index );
        w_i = 1/( 1 + exp( -x_i * beta_i ) );
        
        // corresponding entry of gi
        // we do not rescale the penalization term since that would cause the estimator of the gradient to be biased
        g_i = n * ( ( w_i - y ) * x_i + 2.0 * lambda * ( iter - last_time_beta( i_index ) ) * beta_i ) ;
        
        // udpading the sum of squares
        sum_sq_gi( i_index ) = sum_sq_gi( i_index ) + g_i * g_i;
        
        /////////////////////////
        // main AdaGrad update //
        /////////////////////////
        
        // we do not update beta_i when x_i = 0, but we should have done beta_i += 2*lambda*beta_i at every iteration, i.e., e now do beta_i += 2*lambda*K*beta_i,
        // with K = number of times we did not update beta_i when we should
        beta( i_index ) = beta_i - alpha * g_i / sqrt( sum_sq_gi( i_index ) + eps );
        
        // this was the last time beta_i was updated (so far)
        last_time_beta( i_index ) = iter;
        
      }
      
      // evaluating the log likelihood
      //log_lik = -1 * eval_neg_log_lik( beta, x, y ); // this line uses my own dot product sparse routine
      Xbeta = x.dot(beta);
      log_lik = -1 * y * Xbeta + log( 1 + exp( Xbeta ) );
      
      cumu_log_lik = (1-decay) * log_lik + decay * cumu_log_lik;
      log_lik_vec( iter + loop*n ) = cumu_log_lik;
      
    }
    
    
  }
  
  Rcpp::List ans = Rcpp::List::create(Rcpp::Named("beta") = beta,
                                      Rcpp::Named("log_lik") = log_lik_vec);
  
  return ans;
  
} 


