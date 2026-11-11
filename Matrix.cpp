#include "Matrix.h"

FloatMatrix matrixMultiply(FloatMatrix matrix1, FloatMatrix matrix2){
    if (matrix1.get_columns() != matrix2.get_rows()){
        std::cout << "These matrices can not be multiplied" << '\n';
    }
    size_t target_rows = matrix1.get_rows();
    size_t target_cols = matrix2.get_columns();
    FloatMatrix result = {target_rows, target_cols, std::vector<float>(target_rows * target_cols, 0.0f)};
    for (size_t i=0; i<target_rows; i++){
        for (size_t j=0; j<target_cols; j++){
            for (size_t k=0; k<matrix1.get_columns(); k++){
                result(i,j) += matrix1(i,k) * matrix2(k,j);
            }
        }
    }
    return result;
}

FloatMatrix matrixAddScalar(FloatMatrix matrix1, float scalar){
    size_t rows = matrix1.get_rows();
    size_t columns = matrix1.get_columns();
    FloatMatrix result = {rows,columns, std::vector<float>(rows*columns, 0.0f)};

    for (size_t i=0; i< rows; i++){
        for (size_t j =0; j< columns; j++){
            result(i,j) = matrix1(i,j) + scalar;
        }
    }
    return result;
}

void printMatrix(FloatMatrix matrix){
    for (size_t i = 0; i<matrix.get_rows(); i++){
        for (size_t j =0; j<matrix.get_columns(); j++){
            std::cout << matrix(i,j) << " ";
        }
        std::cout << '\n';
    }
}

void writeMatrix(FloatMatrix matrix, std::ofstream& filetowrite){
    for (size_t i = 0; i<matrix.get_rows(); i++){
        for (size_t j = 0; j<matrix.get_columns(); j++){
            filetowrite << matrix(i,j) << " ";
        }
        filetowrite << '\n';
    }
}

FloatMatrix matrixAdd(FloatMatrix matrix1, FloatMatrix matrix2){
    size_t matrix1_rows = matrix1.get_rows();
    size_t matrix1_columns = matrix1.get_columns();
    size_t matrix2_columns= matrix2.get_columns();
    size_t matrix2_rows = matrix2.get_rows();
    if (matrix1_rows == matrix2_rows && matrix1_columns == matrix2_columns){
        FloatMatrix result = {matrix1_rows, matrix1_columns, std::vector<float>(matrix1_rows * matrix1_columns, 0.0f)};
        for (size_t i =0; i<matrix1_rows; i++){
            for (size_t j=0; j<matrix1_columns; j++){
                result(i,j) = matrix1(i,j) + matrix2(i,j);
            }
        }
        return result;
    }
    if (matrix2_rows == 1 && matrix1_columns == matrix2_columns){
        FloatMatrix result{matrix1_rows, matrix1_columns, std::vector<float>(matrix1_rows * matrix1_columns, 0.0f)};
        for (size_t i = 0; i < matrix1_rows; ++i){
            for (size_t j = 0; j<matrix1_columns; ++j){
                result(i, j) = matrix1(i,j) + matrix2(0,j);
            }
        }
        return result;
    }
    return FloatMatrix{0,0,{}};
}


FloatMatrix matrixSubtract(FloatMatrix matrix1, FloatMatrix matrix2){
    size_t matrix1_rows = matrix1.get_rows();
    size_t matrix1_columns = matrix1.get_columns();
    FloatMatrix result = {matrix1_rows, matrix1_columns, std::vector<float>(matrix1_rows * matrix1_columns, 0.0f)};
    for (size_t i =0; i<matrix1_rows; i++){
        for (size_t j=0; j<matrix1_columns; j++){
            result(i,j) = matrix1(i,j) - matrix2(i,j);
        }
    }
    return result;
}

FloatMatrix matrixMultiplyScalar(FloatMatrix matrix1, float scalar){
    size_t matrix1_rows = matrix1.get_rows();
    size_t matrix1_columns = matrix1.get_columns();
    FloatMatrix result = {matrix1_rows, matrix1_columns, std::vector<float>(matrix1_rows * matrix1_columns, 0.0f)};
    for (size_t i =0; i<matrix1_rows; i++){
        for (size_t j=0; j<matrix1_columns; j++){
            result(i,j) = matrix1(i,j) * scalar;
        }
    }
    return result;
}

FloatMatrix matrixTranspose(FloatMatrix matrix1){
    size_t matrix1_rows = matrix1.get_rows();
    size_t matrix1_columns = matrix1.get_columns();
    FloatMatrix result = {matrix1_columns, matrix1_rows, std::vector<float>(matrix1_rows * matrix1_columns, 0.0f)};
    for (size_t i =0; i<matrix1_rows; i++){
        for (size_t j=0; j<matrix1_columns; j++){
            result(j,i) = matrix1(i,j);
        }
    }
    return result;
}

FloatMatrix matrixSigmoid(FloatMatrix matrix1){
    size_t matrix1_rows = matrix1.get_rows();
    size_t matrix1_columns = matrix1.get_columns();
    FloatMatrix result = {matrix1_rows, matrix1_columns, std::vector<float>(matrix1_rows * matrix1_columns, 0.0f)};
    for (size_t i =0; i<matrix1_rows; i++){
        for (size_t j=0; j<matrix1_columns; j++){
            result(i,j) = 1.0f / (1.0f + std::exp(-matrix1(i,j)));
        }
    }
    return result;
}

FloatMatrix initXavierNormal(size_t rows, size_t columns) {
    float stddev = std::sqrt(2.0f / static_cast<float>(rows + columns));

    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, stddev);

    std::vector<float> values(rows*columns);
    for (size_t i = 0; i < values.size(); ++i){
        values[i] = dist(gen);
    }
    return FloatMatrix{rows, columns, values};
}

FloatMatrix initZeroes(size_t rows, size_t columns){
    std::vector<float> zeroes(rows * columns, 0.0f);
    return FloatMatrix{rows, columns, zeroes};
}

void fileToMatrix(std::string& filename, FloatMatrix& matrix_to_write){

    std::string columnString;
    std::ifstream MyReadFile(filename);

    for (size_t row = 0; std::getline(MyReadFile, columnString); ++row) {
        std::istringstream iss(columnString);
        std::vector<float> currentRow;
        float value;

        for (size_t column = 0; iss >> value; ++column) {
            matrix_to_write(row, column) = value;
        }
    }
    MyReadFile.close();
}
