#include <armadillo>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <sstream>		// for inputing cli arguments

#include "k_means.h"

int main(int argc, char **argv)
{
  int num_clusters = 4;		// default number of clusters

  /* input num_clusters from argv */
  if (argc >= 2)
    {
      std::istringstream iss( argv[1] );
      
      if (!(iss >> num_clusters))
	{
	  std::cout << "Invalid input parameter" << std::endl;
	  std::cout << "Using default number of clusters." << std::endl;
	}
    }
  
  /* initialize Training Data (X) and number of clusters (K) */
  arma::Mat<double> X;
  const arma::uword K = num_clusters;

  // A) Toycluster Data
  char filename[] = "datas/toyclusters.dat";
  
  // B) demo.dat
  //char filename[] = "datas/demo.dat";
  
  // C) Polar
  //char filename[] = "imSeg/polar.dat";

  X.load(filename);
  const arma::uword N = X.n_rows; // N = #examples
  const arma::uword D = X.n_cols; // D = #features
  //std::cout << N << std::endl << D << std::endl;

  // DEBUG
  if (N == X.n_rows)
    std::cout << "So far so good" << std::endl;
  else
    std::cout << "Something is wrong. N = " << N << " and n_rows = " << X.n_rows << std::endl;
  
  /* Declare centroids and index vector containing cluster labels */
  arma::Mat<double> centroids(K, D);
  arma::Mat<double> bestCentroids(K,D);
  arma::Mat<double> initial_centroids(K, D);
  arma::Col<arma::uword> idx(N);
  arma::Col<arma::uword> bestIdx(N);
  double currentCost, bestCost;

  /* Set maximum number of iterations */
  arma::uword max_iter = 200;
  arma::uword num_runs = 3;

  int seed = time(NULL);
  srand(seed);

  for (arma::uword run = 0; run < num_runs; ++run) {
    std::cout << "RUN " << run + 1 << "/" << num_runs << std::endl;
    // Pick some random examples as initial centroids
    initial_centroids = kMeansInitCentroids(X, K);
  
    // Run K-Means algorithm
    if (run == 0) {
      bestCentroids = runkMeans(X, bestIdx, initial_centroids, max_iter, bestCost);
    } else {
      centroids = runkMeans(X, idx, initial_centroids, max_iter, currentCost);
      if (currentCost < bestCost) {
        bestCost = currentCost;
        bestCentroids = centroids;
        bestIdx = idx;
      }
    }
  }

  std::cout << "K-Means Done." << std::endl;
  std::cout << "Best local optimum had a cost of " << bestCost << std::endl;

  // Save output
  char centroidsFile[] = "centroids.out";
  char idxFile[] = "idx.out";
  bestCentroids.save(centroidsFile, arma::raw_ascii);
  bestIdx.save(idxFile, arma::raw_ascii);
  
  return 0;
}
