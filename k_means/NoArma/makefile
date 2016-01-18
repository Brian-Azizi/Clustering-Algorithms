full : tests_main.o unit_tests.o utilities.o
	g++ tests_main.o unit_tests.o -o unit_tests

test : tests_main.o unit_tests.o
	g++ tests_main.o unit_tests.o -o unit_tests

tests_main.o : tests_main.cpp
	g++ -c tests_main.cpp -o tests_main.o -O3 -pedantic -Wall -Wextra
	
unit_tests.o : unit_tests.cpp unit_tests.h
	g++ -c unit_tests.cpp -o unit_tests.o -O3 -pedantic -Wall -Wextra

utilities.o : utilities.cpp utilities.h
	g++ -c utilities.cpp -o utilities.h -O3 -pedantic -Wall -Wextra

clean : 
	rm main.o k_means.o kmeans tests_main.o unit_tests.o unit_tests
