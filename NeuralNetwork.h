#ifndef NEURALNETWORK_H_INCLUDED
#define NEURALNETWORK_H_INCLUDED
#include "Matrix.h"


struct MatrixLayer {
    FloatMatrix weights;
    FloatMatrix biases;
    float learningRate;
};


struct Network{
    MatrixLayer layer1;
    MatrixLayer layer2;
};

void UpdateLayer(MatrixLayer &layer, FloatMatrix matrix1, FloatMatrix matrix2);
void trainStep(Network &net, FloatMatrix X, FloatMatrix Y);


#endif // NEURALNETWORK_H_INCLUDED
