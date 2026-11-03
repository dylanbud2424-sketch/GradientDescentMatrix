#include "NeuralNetwork.h"

void UpdateLayer(MatrixLayer &layer, FloatMatrix X, FloatMatrix Y){
    FloatMatrix Z = matrixMultiply(X, layer.weights);

    for (size_t i=0; i<Z.get_rows(); i++){
        for (size_t j=0; j < Z.get_columns(); j++){
            Z(i,j) += layer.biases(0,j);
        }
    }

    FloatMatrix Yhat = matrixSigmoid(Z);
    FloatMatrix dZ = matrixSubtract(Yhat, Y);
    FloatMatrix Xt = matrixTranspose(X);
    FloatMatrix dW = matrixMultiply(Xt, dZ);

    FloatMatrix dB = {1, layer.biases.get_columns(), std::vector<float>(layer.biases.get_columns(), 0.0f)};
    for (size_t j =0; j < dZ.get_columns(); j++){
        for (size_t i =0; i < dZ.get_rows(); i++){
            dB(0,j) += dZ(i,j);
        }
    }

    FloatMatrix scaled_dW = matrixMultiplyScalar(dW, layer.learningRate);
    layer.weights = matrixSubtract(layer.weights, scaled_dW);

    FloatMatrix scaled_dB = matrixMultiplyScalar(dB, layer.learningRate);
    layer.biases = matrixSubtract(layer.biases, scaled_dB);


}

void trainStep(Network &net, FloatMatrix X, FloatMatrix Y) {
    FloatMatrix Z1 = matrixMultiply(X, net.layer1.weights);
    for (size_t i = 0; i < Z1.get_rows(); i++) {
        for (size_t j = 0; j < Z1.get_columns(); j++) {
            Z1(i, j) += net.layer1.biases(0, j);
        }
    }
    FloatMatrix H1 = matrixSigmoid(Z1);

    FloatMatrix Z2 = matrixMultiply(H1, net.layer2.weights);
    for (size_t i = 0; i < Z2.get_rows(); i++) {
        for (size_t j = 0; j < Z2.get_columns(); j++) {
            Z2(i, j) += net.layer2.biases(0, j);
        }
    }
    FloatMatrix Yhat = matrixSigmoid(Z2);

    FloatMatrix dZ2 = matrixSubtract(Yhat, Y);

    FloatMatrix H1_t = matrixTranspose(H1);
    FloatMatrix dW2 = matrixMultiply(H1_t, dZ2);

    FloatMatrix dB2 = {1, dZ2.get_columns(), std::vector<float>(dZ2.get_columns(), 0.0f)};
    for (size_t j = 0; j < dZ2.get_columns(); j++) {
        for (size_t i = 0; i < dZ2.get_rows(); i++) {
            dB2(0, j) += dZ2(i, j);
        }
    }

    FloatMatrix W2_t = matrixTranspose(net.layer2.weights);
    FloatMatrix dH1 = matrixMultiply(dZ2, W2_t);

    FloatMatrix dZ1 = {dH1.get_rows(), dH1.get_columns(), std::vector<float>(dH1.get_rows() * dH1.get_columns(), 0.0f)};
    for (size_t i = 0; i < dH1.get_rows(); i++) {
        for (size_t j = 0; j < dH1.get_columns(); j++) {
            dZ1(i, j) = dH1(i, j) * H1(i, j) * (1.0f - H1(i, j));
        }
    }

    FloatMatrix X_t = matrixTranspose(X);
    FloatMatrix dW1 = matrixMultiply(X_t, dZ1);

    FloatMatrix dB1 = {1, dZ1.get_columns(), std::vector<float>(dZ1.get_columns(), 0.0f)};
    for (size_t j = 0; j < dZ1.get_columns(); j++) {
        for (size_t i = 0; i < dZ1.get_rows(); i++) {
            dB1(0, j) += dZ1(i, j);
        }
    }

    net.layer2.weights = matrixSubtract(net.layer2.weights, matrixMultiplyScalar(dW2, net.layer2.learningRate));
    net.layer1.weights = matrixSubtract(net.layer1.weights, matrixMultiplyScalar(dW1, net.layer1.learningRate));

    net.layer2.biases = matrixSubtract(net.layer2.biases, matrixMultiplyScalar(dB2, net.layer2.learningRate));
    net.layer1.biases = matrixSubtract(net.layer1.biases, matrixMultiplyScalar(dB1, net.layer1.learningRate));
}
