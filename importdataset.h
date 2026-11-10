#ifndef IMPORTDATASET_H_INCLUDED
#define IMPORTDATASET_H_INCLUDED
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

struct MnistDataset {
    std::vector<int8_t> images{};
    std::vector<uint8_t> labels{};
    size_t count = 0;
};

MnistDataset load_flat_mnist(const std::string& img_path, const std::string& lbl_path);
void load_mnist_dataset(MnistDataset& dataset);

#endif // IMPORTDATASET_H_INCLUDED
