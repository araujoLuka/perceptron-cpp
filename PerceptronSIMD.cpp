// Author: Lucas Araujo
// Date: February 2024
// Description: SIMD Perceptron implementation

#include "PerceptronSIMD.hpp"

#include <iostream>
#include <iomanip>

namespace ml {

PerceptronSIMD::PerceptronSIMD(const float learningRate, const int inputSize)
    : learningRate{learningRate}, inputSize{inputSize}, totalEpochs{0}, biasWeight{0.f} {
    // Initialize weights with random values between 0 and 1
    this->weightsSIMD = _mm_set_ps((float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX,
                                   (float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX);
}

PerceptronSIMD::PerceptronSIMD(const float learningRate, const int inputSize, const int seed)
    : learningRate{learningRate}, inputSize{inputSize}, totalEpochs{0}, biasWeight{0.f} {
    // Initialize weights with random values between 0 and 1 using a seed for reproducibility
    srand(seed);
    this->weightsSIMD = _mm_set_ps((float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX,
                                   (float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX);
}

// Getters
const float PerceptronSIMD::getLearningRate() { return this->learningRate; }
const int PerceptronSIMD::getInputSize() { return this->inputSize; }
const int PerceptronSIMD::getTotalEpochs() { return this->totalEpochs; }
const float PerceptronSIMD::getBiasWeight() { return this->biasWeight; }
const __m128 PerceptronSIMD::getWeights() { return this->weightsSIMD; }
const std::vector<float> PerceptronSIMD::getWeightsVector() {
    std::vector<float> weightsVector;
    _mm_store_ps(&weightsVector[0], this->weightsSIMD);
    return weightsVector;
}

// Activation functions
const float PerceptronSIMD::activation(const std::vector<float>& input) {
    // Convert input vector to SIMD format
    __m128 inputSIMD{_mm_load_ps(&input[0])};
    return this->activation(inputSIMD);
}

// Private activation function for SIMD processing
const float PerceptronSIMD::activation(const __m128& input) {
    // Perform SIMD multiplication and sum
    __m128 mul{_mm_mul_ps(this->weightsSIMD, input)};
    float scalarProduct{0};
    scalarProduct += mul[0] + mul[1] + mul[2] + mul[3];
    // Add bias weight
    return scalarProduct + this->biasWeight;
}

// Prediction functions
const int PerceptronSIMD::predict(const std::vector<float>& input) { return this->activation(input) >= 0 ? 1 : -1; }

const int PerceptronSIMD::predict(const float scalarProduct) { return scalarProduct >= 0 ? 1 : -1; }

// Training function
void PerceptronSIMD::fit(const std::vector<std::vector<float>>& trainingData, const std::vector<int>& labels,
                         const int epochs) {
    for (int epoch{0}; epoch < epochs; ++epoch) {
        // Visualize the training process
        std::cout << "Epoch: " << epoch + 1 << '/' << epochs << " | ";
        std::cout << "Initial weights: ";
        std::cout << std::fixed << std::setw(9) << std::setprecision(5) 
            << this->biasWeight << ' ';
        for (int i{0}; i < this->inputSize; ++i)
            std::cout << std::fixed << std::setw(9) << std::setprecision(5) 
                << this->weightsSIMD[i] << ' ';
        std::cout << "| ";

        // Count the number of misclassified samples
        // If the count is 0, the perceptron has learned and we can stop training
        int misclassifiedCount{0};

        // Process floats in parallel using SIMD
        for (std::size_t i{0}; i < trainingData.size(); ++i) {
            __m128 input{_mm_load_ps(&trainingData[i][0])};
            float scalarProduct{this->activation(input)};
            int prediction{this->predict(scalarProduct)};
            int label{labels[i]};

            // Update weights only if prediction is wrong
            if (prediction != label) {
                ++misclassifiedCount;
                // Update bias weight and weights using SIMD operations
                float learningRate{this->learningRate * label};
                this->biasWeight += learningRate;
                __m128 lr{_mm_set1_ps(learningRate)};
                this->weightsSIMD = _mm_add_ps(this->weightsSIMD, _mm_mul_ps(lr, input));
            }
        }

        // Visualize the training process
        std::cout << "Misclassified: " << misclassifiedCount << " | ";
        std::cout << "Final weights: ";
        std::cout << std::fixed << std::setw(9) << std::setprecision(5) 
            << this->biasWeight << ' ';
        for (int i{0}; i < this->inputSize; ++i)
            std::cout << std::fixed << std::setw(9) << std::setprecision(5) 
                << this->weightsSIMD[i] << ' ';
        std::cout << '\n';

        // If the perceptron has learned, we can stop training
        if (misclassifiedCount == 0) {
            this->totalEpochs = epoch + 1; // +1 because the epoch starts at 0
            break;
        }
    }
}

}  // namespace ml

