#include <armadillo>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>

#include "gmm.h"





/* Return a random integer in the range [0 n) (distributed uniformly)
   n has to be 0 < n <= RAND_MAX
 */
int nrand(int n)
{
  if (n <= 0 || n > RAND_MAX) {
    throw std::domain_error("Argument to nrand is out of range");
  }
  
  const int bucket_size = RAND_MAX / n;
  int r;

  do r = std::rand() / bucket_size;
  while (r >= n);

  return r;
}





/* Initialize the means of the K base distributions in the GMM 
   X is the NxD design matrix
   K is the number of base distributions
   print_seed: (default is false) if true, will print the seed used for the RNG
   Returns a KxD matrix containing K randomly chosen rows of X
 */
arma::Mat<double> gmmInitialMeans(const arma::Mat<double>& X, const arma::uword& K, bool print_seed)
{
  // Initialize values
  arma::uword N = X.n_rows, D = X.n_cols;
  arma::Mat<double> means(K, D);

  // Set seed for the RNG
  int seed = time(NULL);
  srand(seed);

  // select K random rows of X
  arma::Col<arma::uword> rand_idx(K);
  rand_idx.ones();
  rand_idx *= 9999999;   // DANGER: This breaks if N >= 9999999

  arma::uword i = nrand(N);
  rand_idx(0) = i;

  for (arma::uword k = 1; k < K; ++k) {
    i = nrand(N);
    while ( arma::any(rand_idx.rows(0, k-1) == i) ) {
      i = nrand(N);
    }
    rand_idx(k) = i;
  }

  // let the mean be these rows
  for (arma::uword k = 0; k < K; ++k) {
    means.row(k) = X.row(rand_idx(k));
  }
  
  if (print_seed) {
    std::cout << "RNG Seed = " << seed << std::endl;
  }
  
  return means;
}





/* Initialize the variances of the base distributions of the GMM to be the identity
   K is the number of base distributions (clusters)
   D is the dimensionality of the feature space (i.e. width of the design matrix)
   Returns a (K*D)xD matrix containing K copies of the identity DxD matrix stacked vertically 
*/
arma::Mat<double> gmmInitialVars(const arma::uword& K, const arma::uword& D)
{
  arma::Mat<double> Vars(K*D, D);
  arma::Mat<double> sigma(D, D);
  sigma.eye();

  for (arma::uword k = 0; k < K; ++k) {
    Vars.rows(k * D, (k + 1) * D - 1) = sigma;
  }

  return Vars;
}






/* Initialize the mixing coefficients of the GMM to be uniform
   K is the number of base distributions (i.e. number of "clusters")
   Returns a Kx1 matrix with 1/K in each entry
 */
arma::Mat<double> gmmInitialMix(const arma::uword& K)
{
  arma::Mat<double> coeffs(K, 1);
  coeffs.ones();
  coeffs *= 1.0/K;
  return coeffs;
}






/* Compute the Log Likelihood of the GMM Model
   X is the NxD design matrix
   K is the number of base distribtions
   means is the KxD matrix containing the K cluster means in its rows
   vars is the (K*D)xD matrix containing the K cluster variances stacked vertically
   coeffs is the Kx1 matrix containing the mixing coefficients of the cluster
   Returns the log likelihood of the Gaussian Mixture Model 
 */
double gmmLogLikelihood (const arma::Mat<double>& X, const arma::uword& K,\
			 const arma::Mat<double>& means, const arma::Mat<double>& vars, const arma::Mat<double>& coeffs)
{
  /* Declare the variables */
  arma::uword N = X.n_rows, D = X.n_cols;
  arma::Mat<double> logL(1,1);	// Having a 1x1 matrix seems to work better than using a double
  arma::Mat<double> p(1,1);
  
  arma::Mat<double> x(D,1);
  arma::Mat<double> xT(1,D);    // For the transpose

  arma::Mat<double> mu(D,1);
  arma::Mat<double> muT(1,D);
  arma::Mat<double> sigma(D,D);
  arma::Mat<double> invSigma(D,D);
  double detSigma;
  double prob;

  logL.zeros();

  for (arma::uword i = 0; i < N; ++i) { 
    p.zeros();
    xT = X.row(i);
    x = xT.t();
    
    for (arma::uword k = 0; k < K; ++k) {
      // set parameters for base distro k
      prob = coeffs(k);
      muT = means.row(k);
      mu = muT.t();
      sigma = vars.rows(k * D, (k+1) * D - 1);
      detSigma = arma::det(sigma);
      invSigma = arma::pinv(sigma);
      
      // compute p(x_i)
      p += prob * ( 1.0/sqrt(detSigma * pow(2*arma::datum::pi, D)) ) * exp( -0.5 * (xT - muT) * invSigma * (x - mu));
    }
    logL += log(p);
  }
  
  double L = logL(0,0);
  return L;
}






/* E-Step of EM algorithm for the GMM
   X is the NxD design matrix
   K is the number of base distributions
   means is the KxD matrix containing the K cluster means in its rows
   vars is the (K*D)xD matrix containing the K cluster variances stacked vertically
   coeffs is the Kx1 matrix containing the mixing coefficients of the cluster
   Returns the NxK matrix whose (i,k)th element contains the resposibility of cluster k for explaining sample i.
*/
arma::Mat<double> gmmEstep(const arma::Mat<double>& X, const arma::uword& K, const arma::Mat<double>& means,\
			   const arma::Mat<double>& vars, const arma::Mat<double>& coeffs)
{
  // Declare variables
  arma::uword N = X.n_rows, D = X.n_cols;

  arma::Mat<double> x(D,1);
  arma::Mat<double> xT(1,D);

  arma::Mat<double> mu(D,1);
  arma::Mat<double> muT(1,D);
  arma::Mat<double> sigma(D,D);
  arma::Mat<double> invSigma(D,D);
  double detSigma;
  double prob;
  double gaussConstant;

  // Initialize return variable
  arma::Mat<double> Gamma(N,K);
  arma::Mat<double> tempGamma(1,1); // since assigning an element of Gamma (double) directly to a 1x1 matrix (product of matrices) causes issues


  // Compute numerator for the entire Gamma matrix
  for (arma::uword k = 0; k < K; ++k) {
    muT = means.row(k);
    mu = muT.t();
    sigma = vars.rows(k*D, (k+1)*D - 1);
    invSigma = arma::pinv(sigma);
    detSigma = arma::det(sigma);
    prob = coeffs(k);
    gaussConstant = 1.0 / sqrt(detSigma * pow(2*arma::datum::pi, D));

    for (arma::uword i = 0; i < N; ++i) {
      xT = X.row(i);
      x = xT.t();
      tempGamma = gaussConstant * prob * exp(-0.5*(xT - muT)*invSigma*(x - mu));
      Gamma(i,k) = tempGamma(0,0);
    }
  }

  // Normalize the rows of Gamma
  arma::Col<double> gammaSum(N);
  gammaSum = sum(Gamma, 1);  // holds the sum of each row of Gamma
  Gamma.each_col() /= gammaSum;

  return Gamma;
}

