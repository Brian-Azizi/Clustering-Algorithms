#include <fstream>
#include <iostream>
#include <string>


// Not working as of yet.
int numRows(char datafile)
{
  std::ifstream inFile(&datafile);
  int M;
  inFile.seekg(0);
  inFile >> M;
  return M;
}
