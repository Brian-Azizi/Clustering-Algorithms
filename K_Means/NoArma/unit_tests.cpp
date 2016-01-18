#include <iostream>

#include "utilities.h"

void testNumRows()
{
  char datafile[] = "test1.dat";
  std::cout << datafile << std::endl;
  int M = numRows(datafile);
  //int M = 3;
  int actual_M = 3;
  if (M != actual_M) {
    std::cout << "FAILURE! In 'testNumRows'\
when load numRows of test1.dat: Actual M = 3, loaded M = "
	      << M << std::endl;
  }
}

void testDataLoad()
{
  //load Data
  // X = dataLoad("test1.dat");
  // int M = numrows(X);
  // int N = numcols(X);
  //int M = 3;
  //int N = 2;
  
  // actual data:
  int actual_M = 3;
  int actual_N = 2;
  int actual_X[3][2];
  actual_X[0][0] = 3;
  actual_X[0][1] = 4;
  actual_X[1][0] = 2;
  actual_X[1][1] = 1;
  actual_X[2][0] = 0;
  actual_X[2][1] = 7;
  
  // Not sure if these should be here. They should have their own test.
  // if (M != actual_M) {
  // 	std::cerr << "Error in loading test1.dat: M != actual_M" << std::endl;
  // }
  // if (N != actual_N) {
  // 	std::cerr << "Error in loading test1.dat: N != actual_N" << std::endl;
  // }
  
  for (int i = 0; i < actual_M; ++i) {
    for (int j = 0; j < actual_N; ++j) {
      std::cout << actual_X[i][j] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
