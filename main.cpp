#include <iostream>
#include "Matrix.h"

using namespace std;

int main()
{
/*
// 1. Pass the initial inputs (X) into Layer 1
FloatMatrix Z1 = matrixAdd(matrixMultiply(X, layer1.weights), layer1.biases);
FloatMatrix H1 = matrixSigmoid(Z1); // H1 is the output of Layer 1

// 2. Pass Layer 1's outputs (H1) into Layer 2 as its input!
FloatMatrix Z2 = matrixAdd(matrixMultiply(H1, layer2.weights), layer2.biases);
FloatMatrix Yhat = matrixSigmoid(Z2); // Final prediction matrix

*/

    std::vector<float> test_matrix1{1,2,3,4};
    std::vector<float> test_matrix2{5,6,7,8};
    FloatMatrix matrix1{2, 2, test_matrix1};
    FloatMatrix matrix2{2, 2, test_matrix2};
    FloatMatrix result = matrixMultiply(matrix1, matrix2);
    printMatrix(result);
    result = matrixAddScalar(result, 3);
    printMatrix(result);
    return 0;
}
