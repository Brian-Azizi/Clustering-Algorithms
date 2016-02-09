#include <armadillo>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

int main()
{
  // LOAD DATA
  arma::mat X;

  // A) Toy Data
  //char inputFile[] = "../data_files/toyclusters/toyclusters.dat";
  
  // B) X4.dat
  char inputFile[] = "./X4.dat";
  
  
  // INITIALIZE PARAMETERS
  X.load(inputFile);
  const arma::uword N = X.n_rows;
  const arma::uword D = X.n_cols;

  arma::umat ids(N,1);	// needed to shuffle indices later
  arma::umat shuffled_ids(N,1);
  for (arma::uword i = 0; i < N; ++i) {
    ids(i,0) = i;
  }
  int seed = time(NULL);	// set RNG seed to current time
  srand(seed);

  arma::uword K = N;   // initial number of clusters
  
  arma::umat clusters(N,1); // contains cluster assignments for each data point
  for (arma::uword i = 0; i < N; ++i) {
    clusters(i, 0) = i%K;		// initialize as [0,1,...,K-1,0,1,...,K-1,0,...]
  }

  arma::umat cluster_sizes(N,1,arma::fill::zeros); // contains num data points in cluster k
  for (arma::uword i = 0; i < N; ++i) {
    cluster_sizes(clusters(i,0), 0) += 1;
  }

  arma::mat mu(N, D, arma::fill::zeros); // contains cluster mean parameters

  arma::mat filler(D,D,arma::fill::eye);
  std::vector<arma::mat> sigma(N,filler); // contains cluster covariance parameters

  arma::umat uword_dummy(1,1);	// dummy 1x1 matrix;
  // for (arma::uword i = 0; i <N; ++i) {
  //   std::cout << sigma[i] << std::endl;
  // }
  // std::cout << X << std::endl
  // 	    << N << std::endl
  // 	    << D << std::endl
  // 	    << K << std::endl
  // 	    << clusters << std::endl
  // 	    << cluster_sizes << std::endl
  //	    << ids << std::endl;

  // INITIALIZE HYPER PARAMETERS
  // Dirichlet Process concentration parameter is alpha:
  double alpha = 1;
  // Dirichlet Process base distribution (i.e. prior) is 
  // H(mu,sigma) = NIW(mu,Sigma|m_0,k_0,S_0,v_0) = N(mu|m_0,Sigma/k_0)IW(Sigma|S_0,v_0)
  const arma::mat S_0 = arma::cov(X,X,1); // S_xbar / N
  const double v_0 = D + 2;
  const arma::mat m_0 = mean(X).t();
  const double k_0 = 0.01;

  // std::cout << S_0 << std::endl
  // 	    << v_0 << std::endl
  // 	    << m_0 << std::endl
  // 	    << k_0 << std::endl;


  // INITIALIZE SAMPLING PARAMETERS
  arma::uword NUM_SWEEPS = 3;	// number of Gibbs iterations
  bool SAVE_CHAIN = false;	// save output of each Gibbs iteration?
  if (SAVE_CHAIN){
    arma::uword BURN_IN = 2;
    arma::uword CHAINSIZE = NUM_SWEEPS - BURN_IN;
    std::vector<arma::uword> chain_K(CHAINSIZE, K); // Initialize chain variable to initial parameters for convinience
    std::vector<arma::umat> chain_clusters(CHAINSIZE, clusters);
    std::vector<arma::umat> chain_clusterSizes(CHAINSIZE, cluster_sizes);
    std::vector<arma::mat> chain_mu(CHAINSIZE, mu);
    std::vector<std::vector<arma::mat> > chain_sigma(CHAINSIZE, sigma);
    
    for (arma::uword sweep = 0; sweep < CHAINSIZE; ++sweep) {
      std::cout << sweep << " K\n" << chain_K[sweep] << std::endl
    		<< sweep << " clusters\n" << chain_clusters[sweep] << std::endl
    		<< sweep << " sizes\n" << chain_clusterSizes[sweep] << std::endl
    		<< sweep << " mu\n" << chain_mu[sweep] << std::endl;
      for (arma::uword i = 0; i < N; ++i) { 
    	std::cout << sweep << " " << i << " sigma\n" << chain_sigma[sweep][i] << std::endl;
      }
    }
  }

  // START CHAIN
  std::cout << "Starting Algorithm with K = " << K << std::endl;
  for (arma::uword sweep = 0; sweep < NUM_SWEEPS; ++sweep) { 
    // shuffle indices
    shuffled_ids = shuffle(ids);
    // std::cout << shuffled_ids << std::endl;

    // SAMPLE CLUSTERS
    for (arma::uword j = 0; j < N; ++j){
      arma::uword i = shuffled_ids(j);
  
      // Remove i's statistics and any empty clusters
      arma::uword c = clusters(i); // current cluster
      --cluster_sizes(c,0);
      
      if (cluster_sizes(c,0) == 0) { // remove empty cluster
      	cluster_sizes(c,0) = cluster_sizes(K-1,0); // move entries for K onto position c
      	mu.row(c) = mu.row(K-1);
      	sigma[c] = sigma[K-1];
      	uword_dummy(0,0) = c;
      	clusters.rows(find(clusters == K - 1)) = uword_dummy;
	cluster_sizes(K-1,0) = 0;
	mu.row(K-1).fill(-99);
	sigma[K-1].fill(-99);
	--K;
      }
      assert(false);
 
      // Find categorical distribution over clusters
      // p(existing clusters)
      // p(new cluster): find partition function
      // posterior hyperparameters
      // partition = likelihood*prior/posterior
      // sample cluster for i
      // sample parameters for any new-born clusters
    }
    
    // SAMPLE CLUSTER PARAMETERS FROM POSTERIOR
    // cluster data
    // posterior hyperparameters
    // sample fresh parameters
    
    // STORE CHAIN
  }
  // WRITE OUPUT DATA TO FILE
  return 0;
}
