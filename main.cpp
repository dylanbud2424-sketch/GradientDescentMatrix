#include <iostream>
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "importdataset.h"
#include <sstream>

using namespace std;

int main()
{
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

    /*

    for (size_t i=0; i<1000; i++){
        trainStep(mnist_network, X, Y);
    }

    ofstream network_parameters_layer1_weights_file("network_parameters_layer1_weights.txt");
    writeMatrix(mnist_network.layer1.weights, network_parameters_layer1_weights_file);
    network_parameters_layer1_weights_file.close();
    ofstream network_parameters_layer1_biases_file("network_parameters_layer1_biases.txt");
    writeMatrix(mnist_network.layer1.biases, network_parameters_layer1_biases_file);
    network_parameters_layer1_biases_file.close();
    ofstream network_parameters_layer2_weights_file("network_parameters_layer2_weights.txt");
    writeMatrix(mnist_network.layer2.weights, network_parameters_layer2_weights_file);
    network_parameters_layer2_weights_file.close();
    ofstream network_parameters_layer2_biases_file("network_parameters_layer2_biases.txt");
    writeMatrix(mnist_network.layer2.biases, network_parameters_layer2_biases_file);
    network_parameters_layer2_biases_file.close();

    */

    std::string file_to_open_l1w("network_parameters_layer1_weights.txt");
    fileToMatrix(file_to_open_l1w, mnist_network.layer1.weights);
    printMatrix(mnist_network.layer1.weights);

    std::string file_to_open_l1b("network_parameters_layer1_biases.txt");
    fileToMatrix(file_to_open_l1b, mnist_network.layer1.weights);
    printMatrix(mnist_network.layer1.biases);

    std::string file_to_open_l2w("network_parameters_layer2_weights.txt");
    fileToMatrix(file_to_open_l2w, mnist_network.layer1.weights);
    printMatrix(mnist_network.layer2.weights);

    std::string file_to_open_l2b("network_parameters_layer2_biases.txt");
    fileToMatrix(file_to_open_l2b, mnist_network.layer1.weights);
    printMatrix(mnist_network.layer2.biases);

    return 0;
}
