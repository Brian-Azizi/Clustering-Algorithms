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
  arma::Mat<double> MU(K, D);
  arma::Mat<double> SIGMA(K*D, D);
  arma::Mat<double> PI(K,1);
  arma::Mat<double> GAMMA(N, K);


  // RUN EM ALGORITHM AND SEARCH FOR BEST LOCAL MAXIMUM
  arma::uword numRuns = 20;
  arma::uword maxIter = 100;
  arma::Mat<double> logL_hist(maxIter, 1);
  logL_hist = gmmBestLocalMax(X, K, numRuns, maxIter, MU, SIGMA, PI, GAMMA);

  
  // DISPLAY RESULTS
  std::cout << "Results of EM algorithm:" << std::endl
	    << "K = " << K << std::endl
	    << "Means = \n" << MU << std::endl
	    << "Variances = \n" << SIGMA << std::endl
	    << "Mixing coefficients = \n" << PI << std::endl
	    << "Final log Likelihood: " << *(logL_hist.end() - 1) << std::endl;
  


  // SAVE OUTPUT
  char MuFile[] = "../data_files/toyclusters/MU.out";
  char SigmaFile[] = "../data_files/toyclusters/SIGMA.out";
  char PiFile[] = "../data_files/toyclusters/PI.out";
  char GammaFile[] = "../data_files/toyclusters/GAMMA.out";
  char LogLFile[] = "../data_files/toyclusters/logL_hist.out";
  
  MU.save(MuFile, arma::raw_ascii);
  SIGMA.save(SigmaFile, arma::raw_ascii);
  PI.save(PiFile, arma::raw_ascii);
  GAMMA.save(GammaFile, arma::raw_ascii);
  logL_hist.save(LogLFile, arma::raw_ascii);



  // COMPUTE PROBABILITY DENSITY AT A GIVEN POINT
  arma::Mat<double> x_test(D, 1);
  x_test << 5.2 << arma::endr << 3.0 << arma::endr;
  double p_test = gmmDensity(x_test, K, MU, SIGMA, PI);
  
  std::cout << std::endl << "Test point = \n" << std::endl
	    << x_test << "GMM probability density at test point = " 
	    << p_test << std::endl << std::endl;

  return 0;
}
