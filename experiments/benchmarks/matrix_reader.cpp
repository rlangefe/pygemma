#include <Rcpp.h>
#include <fstream>
#include <vector>

// Include io for printing
#include <iostream>

using namespace Rcpp;

# pragma GCC optimize("O3")

/*
Write a function using Rcpp that takes a file path and a list of sampled indices as input. The function will read the space-separated matrix (stored as ascii) at the file path. However, it will only read the entry if both the row index and column index are in the list of sampled indices.  Note that, because this is a very very large matrix, any entry that does not match these criteria should not be read. The function should return a matrix object in R. The input file is also a gz file.

For example, suppose the matrix is in triple backticks:
```
0 1 2
3 4 5
6 7 8
```

Suppose my input is c(1,2). Then after reading, I should get the matrix below:
4 5
7 8
*/

// Function to read and filter a matrix
// [[Rcpp::export]]
NumericMatrix readAndFilterMatrix(std::string filePath, IntegerVector sampledIndices) {
  // Open the file for reading
  std::ifstream file(filePath);

  // Check if the file is open
  if (!file.is_open()) {
    stop("Failed to open the file.");
  }

  // Initialize variables
  std::string line;
  int numCols = sampledIndices.size();
  int numRows = numCols;
  NumericMatrix result(numRows, numCols);
  int i = 0;
  int j = 0; 
  int currenti = 0;
  int currentj = 0;
  double value;

  // Store the sorted sampled indices into a new vector (do not modify the input vector)
  std::vector<int> sortedIndices(sampledIndices.begin(), sampledIndices.end());

  // Sort the sampled indices
  std::sort(sortedIndices.begin(), sortedIndices.end());

  /* 
  Store index from 0 to numCols in the sampled indices vector that would result from sorting
  For example, if sampledIndices = c(2,1,8), then sortedIndices = c(1,2,8) and indexOrder = c(2,1,3)
  */
  std::vector<int> indexOrder(numCols);
  std::iota(indexOrder.begin(), indexOrder.end(), 0);
  std::sort(indexOrder.begin(), indexOrder.end(), [&sampledIndices](int i1, int i2) {return sampledIndices[i1] < sampledIndices[i2];});

  // Read the file line by line
  while (std::getline(file, line)) {
    // Initialize a stringstream to read the line
    std::stringstream ss(line);

    j = 0;
    currentj = 0;

    // Loop over entries in line until next space
    if (currenti == sortedIndices[i]) {
      while (ss >> value) {
        if (currentj == sortedIndices[j]) {
          result(indexOrder[i], indexOrder[j]) = value;
          j++;
        }
        currentj++;

        // Break if we have read all the rows and columns we need
        if (j == numCols) {
          break;
        }
      }
      i++;
    }
    currenti++;

    // Break if we have read all the rows and columns we need
    if (i == numCols) {
      break;
    }
    
  }

  // Close the file
  file.close();


  return result;
}
