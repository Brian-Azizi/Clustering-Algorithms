DPM : mainDPM.o DPM.o
	g++-5.3 mainDPM.o DPM.o -o DPM -std=c++11 -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include -larmadillo

mainDPM.o : mainDPM.cpp
	g++-5.3 -c mainDPM.cpp -o mainDPM.o -std=c++11 -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include -larmadillo -O3 -pedantic -Wall -Wextra

DPM.o : DPM.cpp DPM.h
	g++-5.3 -c DPM.cpp -o DPM.o -std=c++11 -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include -larmadillo -O3 -pedantic -Wall -Wextra

clean : 
	rm mainDPM.o DPM.o DPM dpmMU.out dpmSIGMA.out dpmIDX.out dpmIDX.dat