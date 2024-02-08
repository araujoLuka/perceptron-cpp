// Author: Lucas Araujo
// Date: February 2024
// Description: SIMD Perceptron implementation

#ifndef PERCEPTRON_SIMD_HPP
#define PERCEPTRON_SIMD_HPP

#include "Perceptron.hpp"
#include <immintrin.h>

namespace ml { // means machine learning

class PerceptronSIMD {
public:
    PerceptronSIMD(const float learningRate, const int inputSize);
    PerceptronSIMD(const float learningRate, const int inputSize, const int seed);
    virtual ~PerceptronSIMD() = default;

    /* Getters */
    const float getLearningRate();
    const int getInputSize();
    const int getTotalEpochs();
    const float getBiasWeight();
    const __m128 getWeights();
    const std::vector<float> getWeightsVector();

    /* Activation function
     * > Calculate the scalar product from inputs and weights
     */
    const float activation(const std::vector<float>& inputs);

    /* Prediction function
     * > Uses the activation function to predict the output
     */
    const int predict(const std::vector<float>& inputs);

    /* Prediction function
    * > Uses the activation function with the before calculated scalar product to predict the output
     */
    const int predict(const float scalarProduct);

    /* Training function
     * > Finds the ideal weights for the perceptron to make accurate predictions
     */
    void fit(const std::vector<std::vector<float>>& trainingData, const std::vector<int>& labels, const int epochs);

private:
    const float activation(const __m128& input);

    float learningRate;
    int inputSize;
    int totalEpochs;
    float biasWeight;
    __m128 weightsSIMD;
};

} // namespace ml

#endif // !PERCEPTRON_SIMD_HPP
