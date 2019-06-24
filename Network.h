//
// Created by mathias on 6/5/19.
//

#ifndef NEURAL_NETWORK_MNIST_NETWORK_H
#define NEURAL_NETWORK_MNIST_NETWORK_H

#include <vector>
#include "Neuron.h"

typedef std::vector<Neuron> Layer;

class Network {
public:
    Network(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    std::vector<double> getResults() const;


private:
    std::vector<std::vector<Neuron>> m_layers; //m_layers[layerNum][neuronNum]
    double m_error;


};


#endif //NEURAL_NETWORK_MNIST_NETWORK_H
