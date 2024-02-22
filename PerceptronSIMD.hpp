// Author: Lucas Araujo
// Date: February 2024
// Description: SIMD Perceptron implementation

#ifndef PERCEPTRON_SIMD_HPP
#define PERCEPTRON_SIMD_HPP

#include <vector>
#include <immintrin.h>

namespace ml {

class PerceptronSIMD {
public:
    PerceptronSIMD(const float learningRate, const int inputSize);
    PerceptronSIMD(const float learningRate, const int inputSize, const std::vector<float>& weights);
    virtual ~PerceptronSIMD() = default;

    // Getters
    const float getLearningRate();
    const int getInputSize();
    const int getTotalEpochs();
    const float getBiasWeight();
    const __m128 getWeights();
    const std::vector<float> getWeightsVector();

    // Activation function
    const float activation(const std::vector<float>& inputs);

    // Prediction function
    const int predict(const std::vector<float>& inputs);

    // Prediction function
    const int predict(const float scalarProduct);

    // Training function
    void fit(const std::vector<std::vector<float>>& trainingData, const std::vector<int>& labels, const int epochs);

private:
    // Helper function to calculate activation using SIMD
    const float activation(const __m128& input);

    float learningRate;
    int inputSize;
    int totalEpochs;
    float biasWeight;       // Separated bias weight for SIMD
    __m128 weightsSIMD;     // Using SIMD 4-floats vector
};

} // namespace ml

#endif // !PERCEPTRON_SIMD_HPP

