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
  arma::Mat<double> coeffs(K,1);
  arma::Mat<double> Gamma(N, K);
  double LogL;

  // Parameters for best local search
  arma::Mat<double> MU(K, D);
  arma::Mat<double> SIGMA(K*D, D);
  arma::Mat<double> PI(K,1);
  arma::Mat<double> GAMMA(N, K);
  arma::Mat<double> logLikelihoodEvolution;


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
  std::cout << "Initial log likelihood = " << LogL << std::endl;


  // RUN EM ALGORITHM (output parameters and history of loglikelihood)
  arma::uword maxIter = 100;
  arma::Mat<double> J_hist = gmmRunEM(X, K, maxIter, means, vars, coeffs, Gamma);

  std::cout << "Result of EM algorithm:" << std::endl
	    << "means = \n" << means << std::endl
	    << "variances = \n" << vars << std::endl
	    << "mixing coefficients = \n" << coeffs << std::endl;
  //  std::cout << "Responsibilities = \n" << Gamma << std::endl;
  std::cout << "Final log Likelihood: " << *(J_hist.end() - 1) << std::endl;
  

  // SEARCH FOR BEST LOCAL MAXIMUM
  arma::uword numRuns = 20;
  logLikelihoodEvolution = gmmBestLocalMax(X, K, numRuns, maxIter, MU, SIGMA, PI, GAMMA);



  // SAVE OUTPUT
  char MuFile[] = "../data_files/toyclusters/MU.out";
  char SigmaFile[] = "../data_files/toyclusters/SIGMA.out";
  char PiFile[] = "../data_files/toyclusters/PI.out";
  char GammaFile[] = "../data_files/toyclusters/Gamma.out";
  char LogLFile[] = "../data_files/toyclusters/LogLikelihood.out";
  
  means.save(MuFile, arma::raw_ascii);
  vars.save(SigmaFile, arma::raw_ascii);
  coeffs.save(PiFile, arma::raw_ascii);
  Gamma.save(GammaFile, arma::raw_ascii);
  J_hist.save(LogLFile, arma::raw_ascii);

  // COMPUTE PROBABILITY DENSITY AT A GIVEN POINT


  return 0;
}
