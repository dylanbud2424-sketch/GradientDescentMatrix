#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

#include <vector>
#include <cstddef>
#include <iostream>
#include <cmath>
#include <random>

class FloatMatrix{
    private:
        std::size_t m_rows;
        std::size_t m_columns;
        std::vector<float> m_data;
    public:
        FloatMatrix(std::size_t rows, std::size_t columns, std::vector<float> data)
            : m_rows(rows), m_columns(columns), m_data(data) {}

        float& operator()(std::size_t row, std::size_t column){
            return m_data[row*m_columns+column];
        }
        std::size_t get_columns() const{ return m_columns; }
        std::size_t get_rows() const{ return m_rows; }


};

FloatMatrix matrixMultiply(FloatMatrix matrix1, FloatMatrix matrix2);
FloatMatrix matrixAddScalar(FloatMatrix matrix1, float scalar);
FloatMatrix matrixAdd(FloatMatrix matrix1, FloatMatrix matrix2);
FloatMatrix matrixSubtract(FloatMatrix matrix1, FloatMatrix matrix2);
FloatMatrix matrixMultiplyScalar(FloatMatrix matrix1, float scalar);
FloatMatrix matrixTranspose(FloatMatrix matrix1);
FloatMatrix matrixSigmoid(FloatMatrix matrix);
FloatMatrix initXavierNormal(size_t rows, size_t columns);
FloatMatrix initZeroes(size_t rows, size_t columns);
void printMatrix(FloatMatrix matrix);
#endif // MATRIX_H_INCLUDED
