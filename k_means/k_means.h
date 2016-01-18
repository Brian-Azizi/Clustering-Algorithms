#ifndef GUARD_k_means_h
#define GUARD_k_means_h

#include <armadillo>
#include <iostream>
#include <stdexcept>

arma::Col<arma::uword> findClosestCentroids(const arma::Mat<double>&,
					    const arma::Mat<double>&);
arma::Mat<double> computeCentroids(const arma::Mat<double>&,
				   const arma::Col<arma::uword>&,
				   const arma::uword&);
double kMeansCost(const arma::Mat<double>&, const arma::Mat<double>&,
		  const arma::Col<arma::uword>&);
arma::Mat<double> runkMeans(const arma::Mat<double>&, arma::Col<arma::uword>&,
			    const arma::Mat<double>&, const arma::uword&);
arma::Mat<double> kMeansInitCentroids (const arma::Mat<double>&, const arma::uword&, bool print_seed = false);

#endif
