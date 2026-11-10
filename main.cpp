#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "importdataset.h"

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


std::vector<float> test_matrix1{1,2,3,4};
std::vector<float> test_matrix2{5,6,7,8};
FloatMatrix matrix1{2, 2, test_matrix1};
FloatMatrix matrix2{2, 2, test_matrix2};
FloatMatrix result = matrixMultiply(matrix1, matrix2);
printMatrix(result);
result = matrixAddScalar(result, 3);
printMatrix(result);
load_mnist_dataset();

*/
    size_t input_dimensions = 784;
    size_t hidden_dimensions = 24;
    size_t output_dimensions = 10;
    size_t training_samples = 10000;

    FloatMatrix W1 = initXavierNormal(input_dimensions, hidden_dimensions);
    FloatMatrix B1 = initZeroes(1, hidden_dimensions);
    MatrixLayer layer1{W1, B1, 0.01};

    FloatMatrix W2 = initXavierNormal(hidden_dimensions, output_dimensions);
    FloatMatrix B2 = initZeroes(1, output_dimensions);
    MatrixLayer layer2{W2, B2, 0.01};

    Network mnist_network{layer1, layer2};


    MnistDataset dataset;
    load_mnist_dataset(dataset);
    //const int8_t* img0 = &dataset.images[0 * 784];
    //uint8_t lbl0 = dataset.labels[0];
    std::vector<float> test_matrix1(training_samples*input_dimensions);
    for (size_t i = 0; i<training_samples*input_dimensions; i++){
        test_matrix1[i] = static_cast<float>(static_cast<uint8_t>(dataset.images[i])) / 255.0f;
    }

    std::vector<float> expected_outputs(training_samples * output_dimensions, 0.0f);
    for (size_t i = 0; i<training_samples; i++){
        uint8_t label = static_cast<uint8_t>(dataset.labels[i]);
        if (label < output_dimensions){
            expected_outputs[i * output_dimensions + label] = 1.0f;
        }
    }

    FloatMatrix Y = {training_samples, output_dimensions, expected_outputs};

    FloatMatrix X{training_samples, input_dimensions, test_matrix1};
    FloatMatrix Z1 = matrixAdd(matrixMultiply(X, layer1.weights), layer1.biases);
    FloatMatrix H1 = matrixSigmoid(Z1);

    FloatMatrix Z2 = matrixAdd(matrixMultiply(H1, layer2.weights), layer2.biases);
    FloatMatrix Yhat = matrixSigmoid(Z2);

    trainStep(mnist_network, X, Y);
    return 0;
}
