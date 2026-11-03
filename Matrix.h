#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <vector>
#include <cstddef>
#include <iostream>
#include <cmath>

class FloatMatrix{
    private:
        int m_rows;
        int m_columns;
        std::vector<float> m_data;
    public:
        FloatMatrix(int rows, int columns, std::vector<float> data)
            : m_rows(rows), m_columns(columns), m_data(data) {}

        float& operator()(std::size_t row, std::size_t column){
            return m_data[row*m_columns+column];
        }
        const std::size_t get_columns(){ return m_columns; }
        const std::size_t get_rows(){ return m_rows; }


};

FloatMatrix matrixMultiply(FloatMatrix matrix1, FloatMatrix matrix2);
FloatMatrix matrixAddScalar(FloatMatrix matrix1, float scalar);
FloatMatrix matrixAdd(FloatMatrix matrix1, FloatMatrix matrix2);
FloatMatrix matrixSubtract(FloatMatrix matrix1, FloatMatrix matrix2);
FloatMatrix matrixMultiplyScalar(FloatMatrix matrix1, float scalar);
FloatMatrix matrixTranspose(FloatMatrix matrix1);
FloatMatrix matrixSigmoid(FloatMatrix matrix);
void printMatrix(FloatMatrix matrix);
#endif // MATRIX_H_INCLUDED
