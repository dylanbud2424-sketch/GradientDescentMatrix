#include "importdataset.h"

MnistDataset load_flat_mnist(const std::string& img_path, const std::string& lbl_path) {
    MnistDataset ds;

    std::ifstream img_file(img_path, std::ios::binary | std::ios::ate);
    if (!img_file.is_open()) throw std::runtime_error("Failed to open image file");

    std::streamsize img_size = img_file.tellg();
    img_file.seekg(0, std::ios::beg);

    ds.count = static_cast<size_t>(img_size) / 784;
    ds.images.resize(static_cast<size_t>(img_size));
    img_file.read(reinterpret_cast<char*>(ds.images.data()), img_size);

    std::ifstream lbl_file(lbl_path, std::ios::binary);
    if (!lbl_file.is_open()) throw std::runtime_error("Failed to open label file");

    ds.labels.resize(ds.count);
    lbl_file.read(reinterpret_cast<char*>(ds.labels.data()), static_cast<std::streamsize>(ds.count));

    return ds;
}

void load_mnist_dataset(MnistDataset& dataset){
    dataset = load_flat_mnist("./data/mnist_images_10k.bin", "./data/mnist_labels_10k.bin");
    std::cout << "Loaded" << dataset.count << " images.\n";
}

