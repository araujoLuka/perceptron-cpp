#include "PerceptronSIMD.hpp"

#include <iostream>
#include <iomanip>

namespace ml {

PerceptronSIMD::PerceptronSIMD(const float learningRate, const int inputSize)
    : learningRate{learningRate}, inputSize{inputSize}, totalEpochs{0}, biasWeight{0.f} {
    this->weightsSIMD = _mm_set_ps((float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX,
                                   (float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX);
}

PerceptronSIMD::PerceptronSIMD(const float learningRate, const int inputSize, const int seed)
    : learningRate{learningRate}, inputSize{inputSize}, totalEpochs{0}, biasWeight{0.f} {
    srand(seed);
    this->weightsSIMD = _mm_set_ps((float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX,
                                   (float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX);
}

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

const float PerceptronSIMD::activation(const std::vector<float>& input) {
    __m128 inputSIMD{_mm_load_ps(&input[0])};
    return this->activation(inputSIMD);
}

// private
const float PerceptronSIMD::activation(const __m128& input) {
    __m128 mul{_mm_mul_ps(this->weightsSIMD, input)};
    float scalarProduct{0};
    scalarProduct += mul[0] + mul[1] + mul[2] + mul[3];
    return scalarProduct + this->biasWeight;
}

const int PerceptronSIMD::predict(const std::vector<float>& input) { return this->activation(input) >= 0 ? 1 : -1; }

const int PerceptronSIMD::predict(const float scalarProduct) { return scalarProduct >= 0 ? 1 : -1; }

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

        // Define the amount of floats to process in parallel
        for (std::size_t i{0}; i < trainingData.size(); ++i) {
            __m128 input{_mm_load_ps(&trainingData[i][0])};
            float scalarProduct{this->activation(input)};
            int prediction{this->predict(scalarProduct)};
            int label{labels[i]};

            // using criterion as suggested by bishop
            // only update weights if prediction is wrong
            if (prediction != label) {
                ++misclassifiedCount;
                // We can multiply the learning rate by the label and the operation will be inverted to the correct one
                // If the label is 1, the learning rate will be positive, otherwise it will be negative
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
