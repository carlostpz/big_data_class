
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


//[[Rcpp::export]]
Rcpp::List sgd_logistic_sparse_cpp (  int n_pass, 
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
  Eigen::VectorXd beta(p);  beta.fill(0);
  
  // initializing log likelihood
  double log_lik;
  double cumu_log_lik;
  Eigen::VectorXd log_lik_vec( n * n_pass );
  
  // initializing sum of squared gradients
  Eigen::VectorXd sum_sq_gj( p ); sum_sq_gj.fill( eps );
  
  // indexing the observations
  int index_obs=0;
  SpVec x_i(p);
  double y_i ;
  
  // calculating current gradient
  Eigen::VectorXd gradient(p); 
  
  int j_index;
  double beta_j;
  double w_i;
  double g_j;
  double Xbeta;
  double intercept;
  double sum_y_hat;
  double y_hat;
  double x_ij;
  double g_0;
  double sum_sq_g0 = eps;
  double beta_k;
  double beta_l;
  double gamma;
  
  // when was the last time we updated beta_i?
  Eigen::VectorXd last_time_beta(p); last_time_beta.fill(-1);
  //Eigen::VectorXd last_time_beta(p);
  
  // global iteration counter
  int k =0;
  for ( int loop=0; loop < n_pass; loop++){

    for ( int iter=0; iter < n; iter++){
      
      x_i = X_mat.innerVector(iter);
      y_i = y_vec(iter);

      // updating beta ( needs to be done entry by entry )
      // notice that x_j = 0 => beta_j = 0, then we only update beta_j s.t. x_j != 0 
      for ( InnerIterVec j_(x_i); j_; ++j_ ){
        
        // non zero index and value of x
        j_index = j_.index();
        x_ij = j_.value();
        
        // updating the penalty that was not considered when x_i=0 but should have been
        //gamma = 2 * alpha / sqrt( sum_sq_gj( j_index ) + eps );
        //beta( j_index ) = beta( j_index ) * pow(1 - gamma * lambda, k - last_time_beta( j_index ) - 1 );
        beta( j_index ) = beta( j_index ) * pow(1 - lambda, k - last_time_beta( j_index ) - 1 );
        
        // this was the last time beta_i was updated (so far)
        last_time_beta(j_index) = k;
        
      }
      ///////////////////////////////
      // end of 1st loop inside x_i
      ///////////////////////////////
      
      // now calculate w's with beta's already penalyzied
      
      w_i = 1.0/( 1.0 + exp( -intercept - x_i.dot(beta) ) ); 
      
      // updating the intercept
      g_0 = w_i - y_i;
      sum_sq_g0 += g_0 * g_0;
      intercept = intercept - alpha * g_0 / sqrt( sum_sq_g0 );
      
      // 2nd loop ( gradient update )
      
      for ( InnerIterVec j_(x_i); j_; ++j_ ){
        
        // non zero index and value of x
        j_index = j_.index();
        x_ij = j_.value();
        
        // corresponding entry of gj
        // we do not update beta_i when x_i = 0, but we should have done beta_i += 2*lambda*beta_i at every iteration, i.e., 
        // we now do beta_i += 2*lambda*K*beta_i, with K = number of times we did not update beta_i when we should
        g_j = ( w_i - y_i ) * x_ij  ;
        
        // udpading the sum of squares
        sum_sq_gj( j_index ) += g_j * g_j;
        
        // main AdaGrad update //
        
        //beta( j_index ) = beta_j - alpha * g_j / sqrt( sum_sq_gj( j_index ) + eps ) - 2.0 * alpha * lambda * beta_j;
        beta( j_index ) = beta( j_index ) * (1-lambda) - alpha * g_j / sqrt( sum_sq_gj( j_index ) );
        
      }
      ///////////////////////////////
      // end of 2nd loop inside x_i
      ///////////////////////////////
      
      k = k+1; // increment global counter
      
      // evaluating the log likelihood
      //log_lik = -1 * eval_neg_log_lik( beta, x, y ); // this line uses my own dot product sparse routine
      Xbeta = x_i.dot(beta);
      log_lik = -1.0 * y_i * (alpha + Xbeta) + log( 1.0 + exp( alpha + Xbeta ) );
      
      // moving average log likelihood
      cumu_log_lik = (1.0-decay) * log_lik + decay * cumu_log_lik;
      log_lik_vec( iter + loop*n ) = cumu_log_lik;
      
    }
    /////////////////////////////////////
    // end of loop through observations
    /////////////////////////////////////
  }
  //////////////////////////
  // end of the whole loop 
  //////////////////////////
  
  // last penalization
  
  
  for ( int l=0; l<p; l++ ){
  
    if ( last_time_beta(l) != -1 ){
    //gamma = 2 * alpha / sqrt( sum_sq_gj( l ) + eps );
    //beta( l ) = pow(1 - gamma * lambda, k - last_time_beta(l) ) * beta( l ) ;
    beta( l ) = beta( l ) * pow(1 - lambda, k - last_time_beta( l ) - 1 );
    }
    
  }
  
  Rcpp::List ans = Rcpp::List::create(Rcpp::Named("alpha") = intercept,
                                      Rcpp::Named("beta") = beta,
                                      Rcpp::Named("log_lik") = log_lik_vec,
                                      Rcpp::Named("last_time_beta") = last_time_beta,
                                      Rcpp::Named("sum_sq_gj") = sum_sq_gj,
                                      Rcpp::Named("sum_sq_g0") = sum_sq_g0,
                                      Rcpp::Named("w_i") = w_i,
                                      Rcpp::Named("k") = k);
  
  return ans;
  
} 


// Prediction

//[[Rcpp::export]]
Rcpp::List logistic_prediction( SpMat X_test, Eigen::VectorXd beta_hat, double alpha_hat){
  
  int p = beta_hat.size();
  int n = X_test.cols();
  
  Eigen::VectorXd p_hat(n);
  Eigen::VectorXd y_hat(n);
  
  SpVec x_i;
  
  for ( int i=0; i<n; i++){
    x_i = X_test.innerVector(i);
    p_hat(i) = 1.0 / ( 1.0 + exp( -alpha_hat - x_i.dot( beta_hat ) ) );
    if( p_hat(i) > 0.5 ){
      y_hat(i) = 1;
    }else{
      y_hat(i) = 0;
    }
  }
  
  Rcpp::List ans = Rcpp::List::create( Rcpp::Named("p_hat") = p_hat,
                                       Rcpp::Named("y_hat") = y_hat);
                    
  return ans;
  
}
