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
  //char filename[] = "../data_files/toyclusters/toyclusters.dat";
  
  /*// B) Image Data: Lenna (256x256 pixels unrolled into rows, RGB values in columns)
  X.load("data_files/lenna/lenna_256x256x3.dat");
  arma::Col<arma::uword> imSize(3);
  imSize << 256 << 256 << 3;
  const arma::uword N = imSize(0) * imSize(1);
  const arma::uword D = imSize(2);
  */

  /*// C) Image Data: fiona.jpg (2448x3264x3)
  X.load("data_files/fiona/fiona_2448x3264x3.dat");
  arma::Col<arma::uword> imSize(3);
  imSize << 2448 << 3264 << 3;
  const arma::uword N = imSize(0) * imSize(1);
  const arma::uword D = imSize(2);
  */
  
  // D) MNIST
  //char filename[] = "../data_files/MNIST/MNISTreduced.dat";

  // E) Polar
  char filename[] = "imSeg/polar.dat";

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
  /*// DEBUG
  if (N == X.n_rows)
    std::cout << "So far so good" << std::endl;
  else
    std::cout << "Something is wrong. N = " << N << " and n_rows = " << X.n_rows << std::endl;
  */

  // A) Save toycluster output
  //char centroidsFile[] = "../data_files/toyclusters/centroids.out";
  //char idxFile[] = "../data_files/toyclusters/idx.out";
  char centroidsFile[] = "centroids.out";
  char idxFile[] = "idx.out";
  bestCentroids.save(centroidsFile, arma::raw_ascii);
  bestIdx.save(idxFile, arma::raw_ascii);
  
  /*// B) Compress lenna into K colours
  arma::Mat<double> X_compressed(X.n_rows, X.n_cols);
  for (arma::uword k = 0; k != K; ++k) {
  arma::Col<arma::uword> indices = find(idx == k);
  X_compressed.each_row(indices) = centroids.row(k);
  }
  */
  /*
  // B) Save lenna output data
  X_compressed.save("data_files/lenna/lenna.out", arma::raw_ascii); 
  centroids.save("data_files/lenna/centroids.out", arma::raw_ascii);
  idx.save("data_files/lenna/idx.out", arma::raw_ascii);
  */
  /*
  // DEBUG
  if (N == X_compressed.n_rows)
    std::cout << "So far so good" << std::endl;
  else
    std::cout << "Something is wrong. N = " << N << " and n_rows = " << X_compressed.n_rows << std::endl;

  // C) Save fiona output data
 
  std::ofstream outFile_comp("data_files/fiona/fiona.out");
  std::ofstream outFile_idx("data_files/fiona/idx.out");
  
  if (outFile_comp.good() && outFile_idx.good()) {
    std::cout << "File open successful for X_compressed and idx" << std::endl;
    for (arma::uword i = 0; i != N; i++) {
      outFile_idx << idx(i) << std::endl;
      for (arma::uword j = 0; j != D; ++j) {
	outFile_comp << X_compressed(i,j) << "\t";
      }
      outFile_comp << std::endl;
    }
  }
  else
    std::cout << "Failed to open file for X_compressed or idx" << std::endl;
  outFile_comp.close();
  outFile_idx.close();
  
  std::ofstream outFile_centroids("data_files/fiona/centroids.out");
  if (outFile_centroids.good()) {
    std::cout << "File open successful for centroids" << std::endl;
    for (arma::uword k = 0; k != K; ++k) {
      for (arma::uword j = 0; j != D; ++j) {
	outFile_centroids << centroids(k,j) << '\t';
      }
      outFile_centroids << std::endl;
    }
  } else
    std::cout << "Failed to open file for centroids" << std::endl;
  outFile_centroids.close();
  */
  
  /*
  if (X_compressed.save("data_files/fiona/fiona.out", arma::raw_ascii))
    std::cout << "Save successful." << std::endl;
  else if (X_compressed.save("data_files/fiona/fiona.out", arma::csv_ascii))
    std::cout << "Save CSV successful." << std::endl;
  else
    std::cout << "Save failed." << std::endl;
  //  centroids.save("data_files/fiona/centroids.out", arma::raw_ascii);
  //  idx.save("data_files/fiona/idx.out", arma::raw_ascii);
  */

  // Save MNIST
  /*char centroidsFile[] = "./centroids.out";
  char idxFile[] = "./idx.out";
  bestCentroids.save(centroidsFile, arma::raw_ascii);
  bestIdx.save(idxFile, arma::raw_ascii);
  */

  return 0;
}
