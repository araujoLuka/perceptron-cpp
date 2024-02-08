// Author: Lucas Araujo
// Date: February 2024
// Description: Scalar Perceptron implementation

#include "Perceptron.hpp"

#include <iomanip>
#include <iostream>

namespace ml {

Perceptron::Perceptron(const float learningRate, const int inputSize)
    : learningRate{learningRate}, inputSize{inputSize}, totalEpochs{0} {
    // Initialize weights with random values between 0 and 1
    for (int i{0}; i < inputSize + 1; ++i) 
        this->weights.push_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
}

Perceptron::Perceptron(const float learningRate, const int inputSize, const int seed)
    : learningRate{learningRate}, inputSize{inputSize}, totalEpochs{0} {
    // Initialize weights with random values between 0 and 1 using the provided seed for reproducibility
    srand(seed);
    for (int i{0}; i < inputSize + 1; ++i) 
        this->weights.push_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
}

const std::vector<float> Perceptron::getWeights() { return this->weights; }
const float Perceptron::getLearningRate() { return this->learningRate; }
const int Perceptron::getInputSize() { return this->inputSize; }
const int Perceptron::getTotalEpochs() { return this->totalEpochs; }

const float Perceptron::activation(const std::vector<float>& input) {
    float scalarProduct = this->weights[0]; // Initialize with the bias

    // Calculate the scalar product of inputs and weights
    for (int i{0}; i < this->inputSize; ++i) 
        scalarProduct += input[i] * this->weights[i + 1];

    return scalarProduct;
}

const int Perceptron::predict(const std::vector<float>& input) { 
    return this->activation(input) >= 0 ? 1 : -1; 
}

const int Perceptron::predict(const float scalarProduct) { 
    return scalarProduct >= 0 ? 1 : -1; 
}

void Perceptron::fit(const std::vector<std::vector<float>>& trainingData, const std::vector<int>& labels, const int epochs) {
    for (int epoch{0}; epoch < epochs; ++epoch) {
        // Visualize the training process
        std::cout << "Epoch: " << epoch + 1 << '/' << epochs << " | ";
        std::cout << "Initial weights: ";
        for (int i{0}; i <= this->inputSize; ++i)
            std::cout << std::fixed << std::setw(9) << std::setprecision(5) 
                << this->weights[i] << ' ';
        std::cout << "| ";

        // Count the number of misclassified samples
        // If the count is 0, the perceptron has learned, and we can stop training
        int misclassifiedCount{0};
        for (std::size_t i{0}; i < trainingData.size(); ++i) {
            float scalarProduct{this->activation(trainingData[i])};
            int prediction{this->predict(scalarProduct)};
            int label{labels[i]};
            // Using criterion as suggested by Bishop
            // Only update weights if prediction is wrong
            if (prediction != label) {
                misclassifiedCount++;
                // We can multiply the learning rate by the label, and the operation will be inverted to the correct one
                // If the label is 1, the learning rate will be positive; otherwise, it will be negative
                float learningRate{this->learningRate * label};
                this->weights[0] += learningRate;
                for (int j{0}; j < this->inputSize; ++j)
                    this->weights[j + 1] += learningRate * trainingData[i][j];
            }
        }
        // Visualize the training process
        std::cout << "Misclassified: " << misclassifiedCount << " | ";
        std::cout << "Final weights: ";
        for (int i{0}; i <= this->inputSize; ++i)
            std::cout << std::fixed << std::setw(9) << std::setprecision(5) 
                << this->weights[i] << ' ';
        std::cout << "\n";

        // If the perceptron has learned, we can stop training
        if (misclassifiedCount == 0) {
            this->totalEpochs = epoch + 1; // +1 because the epoch starts at 0
            break;
        }
    }
}

}  // namespace ml

