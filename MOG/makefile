GMM : mainGMM.o gmm.o
	g++ mainGMM.o gmm.o -o GMM -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include -larmadillo
#-DARMA_DONT_USE_WRAPPER -DARMA_USE_BLAS -DARMA_USE_LAPACK -lblas -llapack -larmadillo

mainGMM.o : mainGMM.cpp
	g++ -c mainGMM.cpp -o mainGMM.o -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include -larmadillo -O3 -pedantic -Wall -Wextra

gmm.o : gmm.cpp gmm.h
	g++ -c gmm.cpp -o gmm.o -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include -larmadillo -O3 -pedantic -Wall -Wextra

clean :
	rm mainGMM.o gmm.o GMM