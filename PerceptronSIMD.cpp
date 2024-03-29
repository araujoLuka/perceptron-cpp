// Author: Lucas Araujo
// Date: February 2024
// Description: SIMD Perceptron implementation

#include "PerceptronSIMD.hpp"

namespace ml {

PerceptronSIMD::PerceptronSIMD(const float learningRate, const int inputSize)
    : learningRate{learningRate},
      inputSize{inputSize},
      totalEpochs{0},
      biasWeight{1.f} {
    // Initialize weights with random values between 0 and 1
    this->weightsSIMD = _mm_set_ps(
        (float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX,
        (float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX);
}

PerceptronSIMD::PerceptronSIMD(const float learningRate, const int inputSize,
                               const std::vector<float>& weights)
    : learningRate{learningRate},
      inputSize{inputSize},
      totalEpochs{0},
      biasWeight{1.f} {
    // Initialize weights with the given values
    this->weightsSIMD = _mm_load_ps(&weights[0]);
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
    // Perform SIMD multiplication
    __m128 mul{_mm_mul_ps(this->weightsSIMD, input)};

    // Perform horizontal addition
    __m128 sum1{_mm_hadd_ps(mul, mul)};
    __m128 sum2{_mm_hadd_ps(sum1, sum1)};

    // Extract the result from the sum
    float scalarProduct{_mm_cvtss_f32(sum2)};

    // Add bias weight
    return scalarProduct + this->biasWeight;
}

// Prediction functions
const int PerceptronSIMD::predict(const std::vector<float>& input) {
    return this->activation(input) >= 0 ? 1 : -1;
}

const int PerceptronSIMD::predict(const float scalarProduct) {
    return scalarProduct >= 0 ? 1 : -1;
}

// Training function
void PerceptronSIMD::fit(const std::vector<std::vector<float>>& trainingData,
                         const std::vector<int>& labels, const int epochs) {
    for (int epoch{0}; epoch < epochs; ++epoch) {
        // Count the number of misclassified samples
        // If the count is 0, the perceptron has learned and we can stop
        // training
        int misclassifiedCount{0};

        // Iterate over the training data
        for (std::size_t i{0}; i < trainingData.size(); ++i) {
            // Convert input vector to SIMD format
            __m128 input{_mm_load_ps(&trainingData[i][0])};

            int label{labels[i]};
            float scalarProduct{this->activation(input)};
            int prediction{this->predict(scalarProduct)};

            // Update weights only if prediction is wrong
            if (prediction != label) {
                ++misclassifiedCount;

                float learningRate{this->learningRate * label};

                // Update bias weight
                this->biasWeight += learningRate;

                // Update weights using SIMD
                __m128 lr{_mm_set1_ps(learningRate)};
                this->weightsSIMD =
                    _mm_add_ps(this->weightsSIMD, _mm_mul_ps(lr, input));
            }
        }

        // If the perceptron has learned, we can stop training
        if (misclassifiedCount == 0) {
            this->totalEpochs = epoch + 1;  // +1 because the epoch starts at 0
            break;
        }
    }
}

}  // namespace ml
