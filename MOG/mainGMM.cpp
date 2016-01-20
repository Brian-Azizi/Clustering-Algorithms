#include <armadillo>
#include <fstream>
#include <iostream>
#include <sstream>

#include "gmm.h"

int main(int argc, char **argv)
{
  // SET K
  int num_clusters = 3; // Default
  //    Input number of base distributions from argv if valid
  if (argc >= 2) {
    std::istringstream iss( argv[1] );
    if( !( iss >> num_clusters ) ) {
      std::cout << "Invalid input parameter" << std::endl;
      std::cout << "Using default number of clusters (" << num_clusters << ")" << std::endl;
    }
  }

  // LOAD DATA
  arma::Mat<double> X;
  const arma::uword K = num_clusters;
  
  // A) Toy Data Set
  char filename[] = "../data_files/toyclusters/toyclusters.dat";
  X.load(filename);
  const arma::uword N = X.n_rows;
  const arma::uword D = X.n_cols;
  
  // DECLARE PARAMETERS
  arma::Mat<double> means(K,D);
  arma::Mat<double> vars(K*D,D);
  //arma::Mat<double> sigmaTemp(D,D);
  arma::Mat<double> coeffs(K,1);
  arma::Mat<double> Gamma(N,K);
  double LogL;

  // INITIALIZE PARAMETERS
  //   initialize mu to be K distinct random data points
  means = gmmInitialMeans(X,K); 
  // means << 0 << 0 << arma::endr
  // 	<< 0 << 5 << arma::endr
  // 	<< 6 << 5 << arma::endr;
  //   initialize variances to be the identity
  vars = gmmInitialVars(K,D);
  //   initialize mixing coefficients
  coeffs = gmmInitialMix(K);
  
  std::cout << "Initial values: " << std::endl;
  std::cout << "means = \n" <<  means << std::endl;
  std::cout << "vars = \n" << vars << std::endl;
  std::cout << "coeffs = \n" << coeffs << std::endl;
  

  // COMPUTE INITIAL LOG LIKELIHOOD
  LogL = gmmLogLikelihood(X, K, means, vars, coeffs);
  std::cout << LogL << std::endl;


  // RUN EM ALGORITHM (output history of loglikelihood and parameters)
  //arma::uword maxIter = 200;
  //arma::uword num_runs = 20;
  Gamma = gmmEstep(X, K, means, vars, coeffs);
  gmmMstep(X, K, Gamma, means, vars, coeffs);
  std::cout << "After first M step:" << std::endl
	    << "means = \n" << means << std::endl
	    << "vars = \n" << vars << std::endl
	    << "coeffs = \n" << coeffs << std::endl;
  LogL = gmmLogLikelihood(X, K, means, vars, coeffs);
  std::cout << "LogL = " << LogL << std::endl;
  
  // SAVE OUTPUT

  // COMPUTE PROBABILITY DENSITY AT A GIVEN POINT


  return 0;
}
