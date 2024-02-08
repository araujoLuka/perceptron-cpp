// Author: Lucas Araujo
// Date: February 2024
// Description: Scalar Perceptron implementation

#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <vector>

namespace ml { // means machine learning

class Perceptron {
public:
    // Constructors
    Perceptron(const float learningRate, const int inputSize);
    Perceptron(const float learningRate, const int inputSize, const int seed);
    virtual ~Perceptron() = default;

    // Getters
    const std::vector<float> getWeights();
    const float getLearningRate();
    const int getInputSize();
    const int getTotalEpochs();

    // Activation function
    virtual const float activation(const std::vector<float>& inputs);

    // Prediction function
    const int predict(const std::vector<float>& inputs);

    // Prediction function with precalculated scalar product
    const int predict(const float scalarProduct);

    // Training function
    virtual void fit(const std::vector<std::vector<float>>& trainingData, const std::vector<int>& labels, const int epochs);

private:
    float learningRate;    // Learning rate for the perceptron
    int inputSize;         // Input size for the perceptron
    int totalEpochs;       // Total number of training epochs
    std::vector<float> weights; // Weights for the perceptron
};

} // namespace ml

#endif // !PERCEPTRON_HPP
