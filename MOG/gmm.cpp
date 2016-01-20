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
arma::Mat<double> gmmInitialMeans(const arma::Mat<double>& X, const arma::uword& K)
{
  // Initialize values
  arma::uword N = X.n_rows, D = X.n_cols;
  arma::Mat<double> means(K, D);


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
  gammaSum = arma::sum(Gamma, 1);  // holds the sum of each row of Gamma
  Gamma.each_col() /= gammaSum;

  return Gamma;
}




/* Executes the M-Step of the EM algorithm for the GMM
   X is the NxD design matrix
   K is the number of base distributions in the model
   Gamma is the NxK matrix containing the cluster responsibilities
   means, vars and coeffs are passed by reference and will be modified to store the output
   of the M-Step:
   means will be a KxD matrix containing the new cluster means in its rows
   vars will be a DKxD matrix containing the new cluster variances stacked vertically
   coeffs will be a Kx1 matrix containing the new mixing coefficients
*/

void gmmMstep(const arma::Mat<double>& X, const arma::uword& K, const arma::Mat<double>& Gamma,	\
	      arma::Mat<double>& means, arma::Mat<double>& vars, arma::Mat<double>& coeffs)
{
  // Declare internal variables
  const arma::uword N = X.n_rows, D = X.n_cols;
  arma::Mat<double> x(D,1);
  arma::Mat<double> xT(1,D);
  
  arma::Mat<double> mu(D,1);
  arma::Mat<double> muT(1,D);
  arma::Mat<double> sigma(D,D);
  double prob;
  
  // Declare variables to be returned (i.e. assigned to function arguments that were passed by reference and not const)
  arma::Mat<double> MU(K,D);
  arma::Mat<double> SIGMA(K*D,D);
  arma::Mat<double> PI(K,1);

  // Find N_k
  arma::Row<double> clusterSize = arma::sum(Gamma);  // holds the sum of each column of Gamma (= effective cluster size)

  for (arma::uword k = 0; k < K; ++k) {
    // Find MU
    muT.zeros();
    for (arma::uword i = 0; i < N; ++i) {
      xT = X.row(i);
      muT += Gamma(i,k) * xT;
    }
    muT /= clusterSize(k);
    mu = muT.t();

    // Find Sigma
    sigma.zeros();
    for (arma::uword i = 0; i < N; ++i) {
      xT = X.row(i);
      x = xT.t();
      sigma += Gamma(i,k) * (x - mu) * (xT - muT);
    }
    sigma /= clusterSize(k);

    // Find Pi
    prob = clusterSize(k) / N;

    // Store the variables:
    MU.row(k) = muT;
    SIGMA.rows(k*D, (k+1)*D - 1) = sigma;
    PI(k) = prob;
  }
  
  // Return the result
  means = MU;
  vars = SIGMA;
  coeffs = PI;
}



/* Performs the EM algorithm to fit the Gaussian Mixture Model
   X is the NxD design matrix
   K is the number of base distributions
   maxIter is the maximum number of iterations
   means, vars, coeffs and Gamma are passed by reference and will be modified to store the output
   of the algorithm:
   means will be a KxD matrix containing the final cluster means in its rows
   vars will be a DKxD matrix containing the final cluster variances stacked vertically
   coeffs will be a Kx1 matrix containing the final mixing coefficients
   Gamma will be a NxK matrix containing the final responsibilities
   The function returns a column matrix containing the log likelihood values after each iteration

   The convergence criterion is a mixed error test with target error targetError
 */

arma::Mat<double> gmmRunEM(const arma::Mat<double>& X, const arma::uword& K, const arma::uword& maxIter,\
			   arma::Mat<double>& means, arma::Mat<double>& vars, arma::Mat<double>& coeffs, arma::Mat<double>& Gamma)
{
  
  // Declare internal variables to store results
  arma::Mat<double> MU = means, SIGMA = vars, PI = coeffs;
  arma::uword num = 0;
  const double targetError = 0.00001;
  
  // Declare return variable containing the evolution of the log Likelihood
  arma::Mat<double> J_hist(maxIter, 1);
  J_hist.zeros();

  // Iterate over E-step and M-step. 
  for (arma::uword n = 0; n < maxIter; ++n) {
    Gamma = gmmEstep(X, K, MU, SIGMA, PI);
    gmmMstep(X, K, Gamma, MU, SIGMA, PI);
    J_hist(n, 0) = gmmLogLikelihood(X, K, MU, SIGMA, PI);
    num = n + 1;

    // Check for convergence
    if (n == 0) {
      continue;
    }

    if ( fabs(J_hist(n,0) - J_hist(n-1,0)) < targetError*(1 + fabs(J_hist(n-1,0))) ) {
      break;
    }
  }
  
  
  if (num == maxIter) {
    std::cout << "Warning: No convergence of EM algorithm after " << maxIter << " iterations" << std::endl;
  }
  else if (num < maxIter) {
    std::cout << "EM algorithm converged after " << num << " iterations (of maximal " << maxIter << ")." << std::endl;
    // Fill in the remaining values of J_hist
    arma::Mat<double> fillin(maxIter - num, 1);
    fillin.each_row() = J_hist.row(num - 1);
    J_hist.rows(num, maxIter - 1) = fillin;
  }
  else {
    std::cout << "Error ?" << std::endl;
  }

  // Return parameters:
  means = MU;
  vars = SIGMA;
  coeffs = PI;
  Gamma = gmmEstep(X, K, means, vars, coeffs);

  // Return likelihood values.
  return J_hist;
  //  return J_hist.rows(0,num - 1);
}







/* Run the EM algorithm for the GMM multiple times with different initial cluster means and return the result with the largest
   log likelihood
   X is the NxD design matrix
   K is the number of base distributions
   numRuns is the total number of runs that will be performed
   maxIter is the maximum number of iterations for any single run
   means, vars, coeffs and Gamma are passed by reference and will be modified to store the output
   of the algorithm:
   means will be a KxD matrix containing the best cluster means found in its rows
   vars will be a DKxD matrix containing the best cluster variances found stacked vertically
   coeffs will be a Kx1 matrix containing the best mixing coefficients found
   Gamma will be a NxK matrix containing the best responsibilities found
   The function returns a column matrix containing the log likelihood values for the best run
 */

arma::Mat<double> gmmBestLocalMax(const arma::Mat<double>& X, const arma::uword& K, const arma::uword& numRuns,\
				  const arma::uword& maxIter, arma::Mat<double>& MU, arma::Mat<double>& SIGMA,\
				  arma::Mat<double>& PI, arma::Mat<double>& GAMMA)
{
  // useful constants
  arma::uword N = X.n_rows;
  arma::uword D = X.n_cols;
  

  // Declare Final Variables
  arma::Mat<double> bestMU(K, D);
  arma::Mat<double> bestSIGMA(D*K, D);
  arma::Mat<double> bestPI(K, 1);
  arma::Mat<double> bestGAMMA(N, K);
  arma::Mat<double> best_J_hist(maxIter, 1);
  double bestLogL = -INFINITY;

  // Declare temporary variables
  arma::Mat<double> currentMU(K, D);
  arma::Mat<double> currentSIGMA(D*K, D);
  arma::Mat<double> currentPI(K, 1);
  arma::Mat<double> currentGAMMA(N, K);
  arma::Mat<double> current_J_hist(maxIter, 1);
  double currentLogL;

  int seed = time(NULL);
  srand(seed);
  
  for (arma::uword run = 0; run < numRuns; ++run) {
    // Initialize
    currentMU = gmmInitialMeans(X, K);
    currentSIGMA = gmmInitialVars(K, D);
    currentPI = gmmInitialMix(K);
    
    // Run EM
    std::cout << "Run " << run + 1 << " / " << numRuns << ":" << std::endl;
    current_J_hist = gmmRunEM(X, K, maxIter, currentMU, currentSIGMA, currentPI, currentGAMMA);
    currentLogL = current_J_hist(maxIter - 1, 0);

    // store the best one
    if (run == 0 || currentLogL > bestLogL) {
      bestMU = currentMU;
      bestSIGMA = currentSIGMA;
      bestPI = currentPI;
      bestGAMMA = currentGAMMA;
      best_J_hist = current_J_hist;
      bestLogL = currentLogL;
    }

    std::cout << "Log Likelihood = " << currentLogL << std::endl << std::endl;
  }

  // Return parameters
  MU = bestMU;
  SIGMA = bestSIGMA;
  PI = bestPI;
  GAMMA = bestGAMMA;

  // Return
  return best_J_hist;
}
