DPM : mainDPM.o
	g++ mainDPM.o -o DPM -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include -larmadillo

mainDPM.o : mainDPM.cpp
	g++ -c mainDPM.cpp -o mainDPM.o -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include -larmadillo -O3 -pedantic -Wall -Wextra
clean : 
	rm mainDPM.o DPM