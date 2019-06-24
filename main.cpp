#include <iostream>
#include "PrepareData.hpp"
#include "mnist/mnist_reader.hpp"
#include "Network.h"
#include <map>
#include <iomanip>

Network prepareNet(int input_neurons_num, int hidden_layer_num, int hidden_neurons_num, int output_neurons_num) {
    std::vector<unsigned> topology;

    topology.push_back(input_neurons_num);

    for (int i = 0; i < hidden_layer_num; i++) {
        topology.push_back(hidden_neurons_num);
    }

    topology.push_back(output_neurons_num);

    Network net = Network(topology);

    return net;
}

std::vector<double> prepareImage(std::vector<uint8_t> data);


int main() {
    auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("../data");
    auto data = dataset.training_images;
    auto labels = dataset.training_labels;

    auto prepared_data = Data::prepare(data);
    auto prepared_labels = Data::prepare_labels(labels);

    unsigned input_neurons = dataset.training_images.front().size();

    Network net = Network(std::vector<unsigned int>{input_neurons, 48, 48, 10});

    int traininNum = 300000;
    traininNum = 30000;

    int trainingDataSize = dataset.training_images.size();

    for (int i = 0; i < traininNum; i++) {

        unsigned iNscope = i % trainingDataSize;

        net.feedForward(prepared_data[iNscope]);
        net.backProp(prepared_labels[iNscope]);
    }

    int right = 0;
    int wrong = 0;

    std::map<unsigned, std::map<unsigned, unsigned>> confMatrix;
    prepared_data = Data::prepare(data);

    for (int i = 0; i < dataset.test_images.size(); i++) {
        net.feedForward(prepareImage(dataset.test_images[i]));

        std::vector<double> results = net.getResults();

        double highest = 0;
        unsigned int result = 0;

        for (unsigned int j = 0; j < 10; j++) {
            if (results[j] > highest) {
                highest = results[j];
                result = j;
            }
        }

        if (result == dataset.test_labels[i]) {
            right++;
        } else {
            wrong++;
        }

        confMatrix[result][dataset.test_labels[i]]++;
    }


    for (int row = 0; row < 10; row++) {
        for (int col = 0; col < 10; col++) {
            std::cout << std::setw(10) << confMatrix[row][col];
        }

        std::cout << std::endl;
    }

    double accuracy = double(right) / (right + wrong);
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}


std::vector<double> prepareImage(std::vector<uint8_t> data) {
    std::vector<double> doubleData;

    unsigned int trainingDataSize = data.size();

    for (unsigned int pixel = 0; pixel < trainingDataSize; pixel++) {
        double pixelValue = 2 * (double)data[pixel] / 255.0 - 1.0;
        doubleData.emplace_back(pixelValue);
    }

    return doubleData;
}