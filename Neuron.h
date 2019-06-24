//
// Created by mathias on 6/5/19.
//

#ifndef NEURAL_NETWORK_MNIST_NEURON_H
#define NEURAL_NETWORK_MNIST_NEURON_H

#include <cstdlib>
#include <vector>

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron {

public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVals);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:

    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    // randomWeight: 0 - 1
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    double m_inputVal;
    std::vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
    double learn_rate = 0.01;
    double momentum = 0.3;
};


#endif //NEURAL_NETWORK_MNIST_NEURON_H
