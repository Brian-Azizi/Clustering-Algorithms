#ifndef GUARD_dpm_h
#define GUARD_dpm_h

#include <armadillo>

double logMvnPdf(arma::mat, arma::mat, arma::mat);
double logInvWishPdf(arma::mat, arma::mat, double);
double logNormInvWishPdf(arma::mat, arma::mat, arma::mat, double, arma::mat, double);
arma::uword logCatRnd(arma::mat);
arma::mat mvnRnd(arma::mat, arma::mat);

#endif
