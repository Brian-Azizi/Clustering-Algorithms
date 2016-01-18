#include <armadillo>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>

#include "k_means.h"


// Returns a column vector containing the labels of the closest centroids for a dataset X, where each row of X is a single example. The entries of the return are integers ranging from 0 to K-1 
arma::Col<arma::uword> findClosestCentroids(const arma::Mat<double>& X, 
					    const arma::Mat<double>& centroids)
{
  // Set K and N
  const arma::uword K = centroids.n_rows;
  const arma::uword N = X.n_rows;

  // Return the following variable
  arma::Col<arma::uword> idx(N);
  idx.ones();

  // store the distance of each example to each of the centroids in matrix D
  arma::Mat<double> D(N, K);   
  arma::Mat<double> temp(N,X.n_cols);
  for (arma::uword k = 0; k != K; ++k) { 
    for(arma::uword i = 0; i != N; ++i) {
      temp.row(i) = X.row(i) - centroids.row(k);
    }
    temp = temp%temp;
    D.col(k) = sum(temp, 1);
  }
  
  // store index of nearest cluster in idx
  arma::uword index;
  arma::Row<double> dRow(K);
  for (arma::uword i = 0; i != N; ++i) {
    dRow = D.row(i);
    dRow.min(index);
    idx(i) = index;
  }

  return idx;
}

/* computeCentroids(X, idx, K) takes a design matrix X, cluster membership vector idx and the number of clusters K, and outputs the cluster centroids.
X = NxD design matrix - rows contain examples, columns contain features
idx = Nx1 cluster membership - row i contains cluster index of ith example, one of {0, 1, ..., K-1}
K = number of clusters
output = KxD matrix - row k contains kth centroid  */
arma::Mat<double> computeCentroids(const arma::Mat<double>& X,
				   const arma::Col<arma::uword>& idx, const arma::uword& K)
{
  // Set N and D and declare return variable
  arma::uword N = X.n_rows, D = X.n_cols;
  arma::Mat<double> centroids(K, D);

  // for each cluster, find elements currently assigned to it
  // declare some temporary values
  arma::Mat<arma::uword> inCluster(N,1); //1 if row is in cluster k
  arma::Mat<double> temp(1, D); //accumulate points in cluster k
  temp.zeros();
  arma::uword counter = 0; //count points in cluster k

  for (arma::uword k = 0; k != K; ++k) {
    inCluster = any(idx == k, 1);
    for (arma::uword i = 0; i != N; ++i) {
      if (inCluster(i,0) == 1) {
	temp += X.row(i);
	++counter;
      }
    }
    // find and store the cluster mean
    if ( counter != 0) {
      centroids.row(k) = temp / counter;
    } else {
      centroids.row(k).zeros();
    }
    // reset accumulators
    temp.zeros();
    counter = 0;
  } 

  return centroids;
}
/* given data, centroids and cluster memberships, kMeansCost computes the value of the cost function
   J(c^{(0)}, ..., c^{(N-1)}, mu_0, ..., mu_{K-1}) = (1/N) * sum_{i=0}^{N-1} abs(x^(i) - mu_{c^(i)})^2 
*/
double kMeansCost(const arma::Mat<double>& X, arma::Mat<double>& centroids,
		  const arma::Col<arma::uword>& idx)
{
  // Initialize values
  arma::uword K = centroids.n_rows, N = X.n_rows, D = X.n_cols;
  double cost = 0;
  arma::uword counter = 0;
  arma::Mat<arma::uword> inCluster(N, 1);
  arma::Mat<double> temp(1, D);
  temp.zeros();

  for (arma::uword k = 0; k != K; ++k) {
    inCluster = any(idx == k, 1);
    for (arma::uword i = 0; i != N; ++i) {
      if (inCluster(i) == 1) {
	temp = X.row(i) - centroids.row(k);
	cost += accu(temp % temp);
	++counter;
      }
    }
  }
  if (counter != N) {
    throw std::logic_error("examples assigned to a cluster does not match total number of examples");
  }
  return cost / N;
}

/* runs kMeans. Stores centroids in output and cluster memberships in idx input (which is passed by reference).
X = NxD design matrix
idx = Nx1 cluster memberships. Argument is passed by reference and will store the final indeces.
initial_centroids = KxD centroids matrix. Contains set of centroids to initialize the algorithm
max_iter = integer indicating the upper limit of algorithm iterations. NOTE: currently set as total number of iterations rather than just upper limit.
output = NxD centroids matrix - contains final centroids.  
 */
arma::Mat<double> runkMeans(const arma::Mat<double>& X, arma::Col<arma::uword>& idx, 
			    const arma::Mat<double>& initial_centroids,
			    const  arma::uword& max_iter)
{
  // Initialize values
  arma::uword K = initial_centroids.n_rows;
  arma::Mat<double> centroids = initial_centroids;
  arma::Mat<double> previous_centroids = centroids;
  
  // Run K-Means
  for (arma::uword i = 0; i != max_iter; ++i) {
    // Output progress
    std::cout << "K-Means iteration " << i+1 << "/" << max_iter << std::endl;

    // For each example in X, assign it to the closest centroids
    idx = findClosestCentroids(X, centroids);
    
    // Given memberships, compute new centroids
    centroids = computeCentroids(X, idx, K);
    
    // output progress
    std::cout << "Cost: " << kMeansCost(X, centroids, idx) << std::endl;
  }

  return centroids;
}

//return a random integer in the range [0,n)
int nrand(int n)
{
	if (n <= 0 || n > RAND_MAX)
		throw std::domain_error("Argument to nrand is out of range");

	const int bucket_size = RAND_MAX / n;
	int r;

	do r = std::rand() / bucket_size;
	while (r >= n);

	return r;
}

arma::Mat<double> kMeansInitCentroids (const arma::Mat<double>& X, const arma::uword& K, bool print_seed)
{
  // Initialize values
  arma::uword N = X.n_rows, D = X.n_cols;
  arma::Mat<double> centroids(K, D);

  // Set the seed for the RNG
  int seed = time(NULL);
  srand(seed);
  
  // Pick K random rows of X
  arma::Col<arma::uword> rand_idx(K);
  rand_idx.ones();
  rand_idx *= 999999;
  
  arma::uword i;
  for (arma::uword k = 0; k != K; ++k) {
    i = nrand(N);
    if (k == 0) {
      rand_idx(0) = i;
      continue;
    }
    
    while (arma::any(rand_idx.rows(0, k-1) == i)) {
      i = nrand(N);
    }
    rand_idx(k) = i;
  }

  // set centroids = rows of X given by rand_idx
  for (arma::uword k = 0; k != K; ++k) {
    centroids.row(k) = X.row(rand_idx(k));
  }
  if (print_seed) {
    std::cout << "Seed ID: " << seed << std::endl;
  }

  return centroids;
}
