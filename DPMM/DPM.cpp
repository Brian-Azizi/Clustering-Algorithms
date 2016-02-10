#include <armadillo>
#include <cmath>		// for lgamma function
#include <stdexcept>

#include "DPM.h"

// compute the log of the multivariate normal density at x with mean mu and covariance matrix sigma
// Assumes x is a Dx1 matrix, mu is a Dx1 matrix, sigma is a DxD symmetric, positive definite matrix
double logMvnPdf(arma::mat x, arma::mat mu, arma::mat sigma)
{
  arma::uword D = sigma.n_rows;
  
  double logDet, signDet;
  arma::log_det(logDet, signDet, sigma);
  
  arma::mat invSigma = arma::inv_sympd(sigma);
  arma::mat x_mu = invSigma * (x - mu);
  arma::mat mahalanobis = (x-mu).t() * x_mu;

  double logpdf = -0.5*D*log(2*arma::datum::pi) - 0.5*logDet - 0.5*mahalanobis(0,0);
  return logpdf;
  
}

// compute the log of the inverse wishart pdf at sigma with scale matrix S and degrees of freedom nu
// Assumes that sigma and S are symmetric and positive definite DxD matrices, and nu > D-1 
double logInvWishPdf(arma::mat sigma, arma::mat S, double nu)
{
  arma::uword D = sigma.n_rows;

  double logS_term, signS;
  arma::log_det(logS_term, signS, S);
  logS_term *= 0.5*nu;
  
  double log2_term = 0.5*D*nu * log(2);
  
  double logSigma_term, signSigma;
  arma::log_det(logSigma_term, signSigma, sigma);
  logSigma_term *= 0.5*(nu + D + 1);
  
  double trace_term;
  arma::mat invSigma = arma::inv_sympd(sigma);
  trace_term = 0.5 * arma::trace(S * invSigma);

  double logGamma_term = 0.25 * D * (D - 1) * log(arma::datum::pi);
  for (arma::uword j = 0; j < D ; ++j) { 
    logGamma_term += lgamma(0.5*(nu - j));
  }

  double logPDF = logS_term - log2_term - logGamma_term - logSigma_term - trace_term;
  return logPDF;
}


// Compute the log of the Gaussian-Inverse-Wishart density at (mu,sigma) with parameters m,k,S,nu:
// return logNIW(mu,sigma|m,k,S,nu) = logN(mu|m,sigma/k) + logIW(sigma|S,nu)
// Assumes mu and m are Dx1 matrices, sigma and S are symmetric, positive definite DxD matrices,
// k is a positive scalar and nu is a scalar > D-1
double logNormInvWishPdf (arma::mat mu, arma::mat sigma, arma::mat m, double k, arma::mat S, double nu)
{
  double logN = logMvnPdf(mu, m, sigma/k);
  double logIW = logInvWishPdf(sigma, S, nu);

  double logPDF = logN + logIW;
  return logPDF;
}

// return a sample from the categorical (discrete/multinoulli) distribution.
// Assumes that logP is a Kx1 matrix (where K is the total number of states)
// containing the (unnormalized) log probabilities
arma::uword logCatRnd (arma::mat logP)
{
  arma::uword K = logP.n_rows;
  arma::mat p = exp(logP);	// recover p using softmax trick
  p /= arma::accu(p);

  arma::mat cdf = arma::cumsum(p);
  double u = arma::randu();

  arma::uword cat = K;
  for (arma::uword k=0; k < K; ++k) {
    if (cdf(k,0) >= u) {
      cat = k;
      break;
    }
  }
  
  return cat;
}

// return a random sample from the multivariate normal distribution with mean mu and covariance sigma
// assumes that mu is a Dx1 matrix and sigma is a symmetric, positive semi definite DxD matrix
// (Tested.)
arma::mat mvnRnd(arma::mat mu, arma::mat sigma)
{
  // Decompose sigma = A*A_transpose using cholesky
  arma::uword D = sigma.n_rows;
  arma::mat A(D, D);
  bool b = arma::chol(A, sigma, "lower"); // b = true if cholesky was succesful

  if (!b) { 			// sigma is not positive definite;
    arma::mat E(D, D, arma::fill::eye);
    b = arma::chol(A, sigma + 0.0000000001*E, "lower"); // add slight perturbation to make it p.d. and try again
  }
  
  if (!b) {			// if chol failed again, sigma is not positive semi definite
    throw std::runtime_error("sigma is not positive semi-definite");
  }
  
  // generate sample from standard normal N(0,I)
  arma::mat z = arma::randn<arma::mat>(D,1);
  
  // transform to make it N(mu,sigma)
  arma::mat x = mu + A*z;
  
  return x;
}
