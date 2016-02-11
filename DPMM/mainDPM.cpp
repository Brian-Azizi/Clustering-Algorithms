#include <armadillo>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "DPM.h"

int main()
{
  // LOAD DATA
  arma::mat X;

  // A) Toy Data
  char inputFile[] = "../data_files/toyclusters/toyclusters.dat";
  
  // B) X4.dat
  //char inputFile[] = "./X4.dat";
  
  // C) fisher data
  //char inputFile[] = "./fisher.dat";
  
  // INITIALIZE PARAMETERS
  X.load(inputFile);
  const arma::uword N = X.n_rows;
  const arma::uword D = X.n_cols;

  arma::umat ids(N,1);	// needed to shuffle indices later
  arma::umat shuffled_ids(N,1);
  for (arma::uword i = 0; i < N; ++i) {
    ids(i,0) = i;
  }
  
  arma::arma_rng::set_seed_random(); // set arma rng
  // int seed = time(NULL);	// set RNG seed to current time
  // srand(seed);

  arma::uword initial_K = N;   // initial number of clusters
  arma::uword K = initial_K;
  
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

  if (K < N) {			// set parameters not belonging to any cluster to -999
    mu.rows(K,N-1).fill(-999);
    for (arma::uword k = K; k < N; ++k) {
      sigma[k].fill(-999);
    }
  }

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
  // H(mu,sigma) = NIW(mu,Sigma|m_0,k_0,S_0,nu_0) = N(mu|m_0,Sigma/k_0)IW(Sigma|S_0,nu_0)
  const arma::mat S_0 = arma::cov(X,X,1); // S_xbar / N
  const double nu_0 = D + 2;
  const arma::mat m_0 = mean(X).t();
  const double k_0 = 0.01;

  std::cout << S_0 << std::endl
  	    << nu_0 << std::endl
  	    << m_0 << std::endl
  	    << k_0 << std::endl;


  // INITIALIZE SAMPLING PARAMETERS
  arma::uword NUM_SWEEPS = 1000;	// number of Gibbs iterations
  bool SAVE_CHAIN = true;	// save output of each Gibbs iteration?
  arma::uword BURN_IN = NUM_SWEEPS - 10;
  arma::uword CHAINSIZE = NUM_SWEEPS - BURN_IN;
  std::vector<arma::uword> chain_K(CHAINSIZE, K); // Initialize chain variable to initial parameters for convinience
  std::vector<arma::umat> chain_clusters(CHAINSIZE, clusters);
  std::vector<arma::umat> chain_clusterSizes(CHAINSIZE, cluster_sizes);
  std::vector<arma::mat> chain_mu(CHAINSIZE, mu);
  std::vector<std::vector<arma::mat> > chain_sigma(CHAINSIZE, sigma);
  
  // for (arma::uword sweep = 0; sweep < CHAINSIZE; ++sweep) {
  //   std::cout << sweep << " K\n" << chain_K[sweep] << std::endl
  // 		<< sweep << " clusters\n" << chain_clusters[sweep] << std::endl
  // 		<< sweep << " sizes\n" << chain_clusterSizes[sweep] << std::endl
  // 		<< sweep << " mu\n" << chain_mu[sweep] << std::endl;
  //   for (arma::uword i = 0; i < N; ++i) { 
  // 	std::cout << sweep << " " << i << " sigma\n" << chain_sigma[sweep][i] << std::endl;
  //   }
  // }
  

  // START CHAIN
  std::cout << "Starting Algorithm with K = " << K << std::endl;
  for (arma::uword sweep = 0; sweep < NUM_SWEEPS; ++sweep) { 
    // shuffle indices
    shuffled_ids = shuffle(ids);
    // std::cout << shuffled_ids << std::endl;

    // SAMPLE CLUSTERS
    for (arma::uword j = 0; j < N; ++j){
      arma::uword i = shuffled_ids(j);
      arma::mat x = X.row(i).t(); // current data point
      
      // Remove i's statistics and any empty clusters
      arma::uword c = clusters(i,0); // current cluster
      cluster_sizes(c,0) -= 1;
      
      if (cluster_sizes(c,0) == 0) { // remove empty cluster
      	cluster_sizes(c,0) = cluster_sizes(K-1,0); // move entries for K onto position c
	mu.row(c) = mu.row(K-1);
      	sigma[c] = sigma[K-1];
      	uword_dummy(0,0) = c;
	arma::uvec idx = find(clusters == K - 1);
      	clusters.each_row(idx) = uword_dummy;
	cluster_sizes(K-1,0) = 0;
	mu.row(K-1).fill(-999);
	sigma[K-1].fill(-999);
	--K;
      }

      // quick test of logMvnPdf:
      // arma::mat m_(2,1);
      // arma::mat s_(2,2);
      // arma::mat t_(2,1);
      // m_ << 1 << arma::endr << 2;
      // s_ << 3 << -0.2 << arma::endr << -0.2 << 1;
      // t_ << -3 << arma::endr << -3;
      // double lpdf = logMvnPdf(t_, m_, s_);
      // std::cout << lpdf << std::endl; // śhould be -19.1034 (works)


      
      // Find categorical distribution over clusters (tested)
      arma::mat logP(K+1, 1, arma::fill::zeros);
      
      // quick test of logInvWishPdf
      // arma::mat si_(2,2), s_(2,2);
      // double nu_ = 4;
      // si_ << 1 << 0.5 << arma::endr << 0.5 << 4;
      // s_ << 3 << -0.2 << arma::endr << -0.2 << 1;
      // double lpdf = logInvWishPdf(si_, s_, nu_);
      // std::cout << lpdf << std::endl; // should be -7.4399 (it is)

      // quick test for logNormInvWishPdf
      // arma::mat si_(2,2), s_(2,2);
      // arma :: mat mu_(2,1), m_(2,1);
      // double k_ = 0.5, nu_ = 4;
      // si_ << 1 << 0.5 << arma::endr << 0.5 << 4;
      // s_ << 3 << -0.2 << arma::endr << -0.2 << 1;
      // mu_ << -3 << arma::endr << -3;
      // m_ << 1 << arma::endr << 2;
      // double lp = logNormInvWishPdf(mu_,si_,m_,k_,s_,nu_);
      // std::cout << lp << std::endl; // should equal -15.2318 (it is)
      
      // p(existing clusters) (tested)
      for (arma::uword k = 0; k < K; ++k) {
	arma::mat m_ = mu.row(k).t();
	arma::mat s_ = sigma[k];
	logP(k,0) = log(cluster_sizes(k,0)) - log(N-1+alpha) + logMvnPdf(x,m_,s_);
      }

      // p(new cluster): find partition function (tested)
      arma::mat dummy_mu(D, 1, arma::fill::zeros);
      arma::mat dummy_sigma(D, D, arma::fill::eye);
      double logPrior, logLklihd, logPstr, logPartition;

      // posterior hyperparameters (tested)
      arma::mat m_1(D,1), S_1(D,D);
      double k_1, nu_1;
      k_1 = k_0 + 1;
      nu_1 = nu_0 + 1;
      m_1 = (k_0*m_0 + x) / k_1;
      S_1 = S_0 + x * x.t() + k_0 * (m_0 * m_0.t()) - k_1 * (m_1 * m_1.t());
      // std::cout << k_1 << std::endl
      // 		<< nu_1 << std::endl
      // 		<< m_1 << std::endl
      // 		<< S_1 << std::endl;
      
      // partition = likelihood*prior/posterior (tested)
      // (perhaps a direct computation of the partition function would be better)
      logPrior = logNormInvWishPdf(dummy_mu, dummy_sigma, m_0, k_0, S_0, nu_0);
      logLklihd = logMvnPdf(x, dummy_mu, dummy_sigma);
      logPstr = logNormInvWishPdf(dummy_mu, dummy_sigma, m_1, k_1, S_1, nu_1);
      logPartition = logPrior + logLklihd - logPstr;      
      // std::cout << logPrior << std::endl
      // 		<< logLklihd << std::endl
      // 		<< logPstr << std::endl
      // 		<< logPartition << std::endl;

      // p(new cluster): (tested)
      logP(K,0) = log(alpha) - log(N - 1 + alpha) + logPartition;
      //      std::cout << logP << std::endl;
      //if(i == 49)
      //assert(false);
      // sample cluster for i
      
      arma::uword c_ = logCatRnd(logP);
      clusters(i,0) = c_;

      // quick test for mvnRnd
      // arma::mat mu, si;
      // mu << 1 << arma::endr << 2;
      // si << 1 << 0.9 << arma::endr << 0.9 << 1;
      // arma::mat m = mvnRnd(mu, si);
      // std::cout << m << std::endl;

      // quick test for invWishRnd
      // double df = 4;
      // arma::mat si(2,2);
      // si << 1 << 1 << arma::endr << 1 << 1;
      // arma::mat S = invWishRnd(si,df);
      // std::cout << S << std::endl;

      if (c_ == K) {	// Sample parameters for any new-born clusters from posterior
	cluster_sizes(K, 0) = 1;
	arma::mat si_ = invWishRnd(S_1, nu_1);
	//arma::mat si_ = S_1;
	arma::mat mu_ = mvnRnd(m_1, si_/k_1);
	//arma::mat mu_ = m_1;
	mu.row(K) = mu_.t();
	sigma[K] = si_;
	K += 1;
      } else {
	cluster_sizes(c_,0) += 1;
      }
      // if (j == N-1) {
      // 	std::cout << logP << std::endl;
      // 	std::cout << K << std::endl;
      // 	assert(false);
      // }
    }
    
    // sample CLUSTER PARAMETERS FROM POSTERIOR
    for (arma::uword k = 0; k < K; ++k) {
      // cluster data
      arma::mat Xk = X.rows(find(clusters == k));
      arma::uword Nk = cluster_sizes(k,0);

      // posterior hyperparameters
      arma::mat m_Nk(D,1), S_Nk(D,D);
      double k_Nk, nu_Nk;
      
      arma::mat sum_k = sum(Xk,0).t();
      arma::mat cov_k(D, D, arma::fill::zeros);
      for (arma::uword l = 0; l < Nk; ++l) { 
	cov_k += Xk.row(l).t() * Xk.row(l);
      }
      
      k_Nk = k_0 + Nk;
      nu_Nk = nu_0 + Nk;
      m_Nk = (k_0 * m_0 + sum_k) / k_Nk;
      S_Nk = S_0 + cov_k + k_0 * (m_0 * m_0.t()) - k_Nk * (m_Nk * m_Nk.t());
      
      // sample fresh parameters
      arma::mat si_ = invWishRnd(S_Nk, nu_Nk);
      //arma::mat si_ = S_Nk;
      arma::mat mu_ = mvnRnd(m_Nk, si_/k_Nk);
      //arma::mat mu_ = m_Nk;
      mu.row(k) = mu_.t();
      sigma[k] = si_;
    }
    std::cout << "Iteration " << sweep + 1 << "/" << NUM_SWEEPS<< " done. K = " << K << std::endl;
        
    // STORE CHAIN
    if (SAVE_CHAIN) {
      if (sweep >= BURN_IN) { 
	chain_K[sweep - BURN_IN] = K;
	chain_clusters[sweep - BURN_IN] = clusters;
	chain_clusterSizes[sweep - BURN_IN] = cluster_sizes;
	chain_mu[sweep - BURN_IN] = mu;
	chain_sigma[sweep - BURN_IN] = sigma;
      }
    }
	
  }
  std::cout << "Final cluster sizes: " << std::endl
	    << cluster_sizes.rows(0, K-1) << std::endl;






  
  // WRITE OUPUT DATA TO FILE
  arma::mat MU = mu.rows(0,K-1);
  arma::mat SIGMA(D*K,D);
  for (arma::uword k = 0; k < K; ++k) { 
    SIGMA.rows(k*D,(k+1)*D-1) = sigma[k];
  }
  arma::umat IDX = clusters;

  // A) toycluster data
  // char MuFile[] = "../data_files/toyclusters/dpmMU.out";
  // char SigmaFile[] = "../data_files/toyclusters/dpmSIGMA.out";
  // char IdxFile[] = "../data_files/toyclusters/dpmIDX.out";

  // B) X4.dat
  char MuFile[] = "dpmMU.out";
  char SigmaFile[] = "dpmSIGMA.out";
  char IdxFile[] = "dpmIDX.out";
  
  MU.save(MuFile, arma::raw_ascii);
  SIGMA.save(SigmaFile, arma::raw_ascii);
  IDX.save(IdxFile, arma::raw_ascii);


  if (SAVE_CHAIN) {
    std::ofstream chainKFile("chainK.out");
    std::ofstream chainClustersFile("chainClusters.out");
    std::ofstream chainClusterSizesFile("chainClusterSizes.out");
    std::ofstream chainMuFile("chainMu.out");
    std::ofstream chainSigmaFile("chainSigma.out");
    
    chainKFile << "Dirichlet Process Mixture Model.\nInput: " << inputFile << std::endl
	       << "Number of iterations of Gibbs Sampler: " << NUM_SWEEPS << std::endl
	       << "Burn-In: " << BURN_IN << std::endl
	       << "Initial number of clusters: " << initial_K << std::endl
	       << "Output: Number of cluster (K)\n" << std::endl; 
    chainClustersFile << "Dirichlet Process Mixture Model.\nInput: " << inputFile << std::endl
		      << "Number of iterations of Gibbs Sampler: " << NUM_SWEEPS << std::endl
		      << "Burn-In: " << BURN_IN << std::endl
		      << "Initial number of clusters: " << initial_K << std::endl
		      << "Output: Cluster identities (clusters)\n" << std::endl; 
    chainClusterSizesFile << "Dirichlet Process Mixture Model.\nInput: " << inputFile << std::endl
			  << "Number of iterations of Gibbs Sampler: " << NUM_SWEEPS << std::endl
			  << "Burn-In: " << BURN_IN << std::endl
			  << "Initial number of clusters: " << initial_K << std::endl
			  << "Output: Size of clusters (cluster_sizes)\n" << std::endl; 
    chainMuFile << "Dirichlet Process Mixture Model.\nInput: " << inputFile << std::endl
		<< "Number of iterations of Gibbs Sampler: " << NUM_SWEEPS << std::endl
		<< "Burn-In: " << BURN_IN << std::endl
		<< "Initial number of clusters: " << initial_K << std::endl
		<< "Output: Samples for cluster mean parameters (mu. Note: means stored in rows)\n" << std::endl; 
    chainSigmaFile << "Dirichlet Process Mixture Model.\nInput: " << inputFile << std::endl
		   << "Number of iterations of Gibbs Sampler: " << NUM_SWEEPS << std::endl
		   << "Burn-In " << BURN_IN << std::endl
		   << "Initial number of clusters: " << initial_K << std::endl
		   << "Output: Samples for cluster covariances (sigma)\n" << std::endl; 


    for (arma::uword sweep = 0; sweep < CHAINSIZE; ++sweep) {
      arma::uword K = chain_K[sweep];
      chainKFile << "Sweep #" << BURN_IN + sweep + 1 << "\n" << chain_K[sweep] << std::endl;
      chainClustersFile << "Sweep #" << BURN_IN + sweep + 1 << "\n" << chain_clusters[sweep]  << std::endl;
      chainClusterSizesFile << "Sweep #" << BURN_IN + sweep + 1 << "\n" << chain_clusterSizes[sweep].rows(0, K - 1) << std::endl;
      chainMuFile << "Sweep #" << BURN_IN + sweep + 1 << "\n" << chain_mu[sweep].rows(0, K - 1) << std::endl;
      chainSigmaFile << "Sweep #" << BURN_IN + sweep + 1<< "\n";
      for (arma::uword i = 0; i < K; ++i) { 
	chainSigmaFile << chain_sigma[sweep][i] << std::endl;
      }
      chainSigmaFile << std::endl;
    }
  }

  return 0;
}
