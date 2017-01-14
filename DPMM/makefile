DPM : mainDPM.o DPM.o
	g++ mainDPM.o DPM.o -o DPM -larmadillo -std=c++11 

mainDPM.o : mainDPM.cpp
	g++ -c mainDPM.cpp -o mainDPM.o -larmadillo -std=c++11 

DPM.o : DPM.cpp DPM.h
	g++ -c DPM.cpp -o DPM.o -std=c++11 -larmadillo -O3 -pedantic -Wall -Wextra

clean : 
	rm mainDPM.o DPM.o DPM dpmMU.out dpmSIGMA.out dpmIDX.out dpmIDX.dat chainMu.out chainK.out chainClusters.out chainSigma.out chainClusterSizes.out
