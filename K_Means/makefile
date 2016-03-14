kmeans : main.o k_means.o
	g++ -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include main.o k_means.o -o kmeans

k_means.o : k_means.cpp k_means.h
	g++ -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include -c k_means.cpp -o k_means.o -O3 -pedantic -Wall -Wextra

main.o : main.cpp
	g++ -L/lsc/opt/armadillo-6.100/lib -I/lsc/opt/armadillo-6.100/include -c main.cpp -o main.o -O3 -pedantic -Wall -Wextra

clean : 
	rm main.o k_means.o kmeans centroids.out idx.out

dataClear :
	rm ../data_files/toyclusters/centroids.out ../data_files/toyclusters/idx.out ../data_files/toyclusters/idx.dat ../data_files/toyclusters/plot.png