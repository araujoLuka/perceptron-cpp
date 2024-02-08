// Author: Lucas Araujo
// Date: February 2024
// Description: Scalar Perceptron implementation

#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <vector>

namespace ml { // means machine learning

class Perceptron {
public:
    Perceptron(const int inputSize, const float leaningRate);
    virtual ~Perceptron() = default;

    /* Getters */
    const std::vector<float> getWeights();
    const float getLearningRate();
    const int getInputSize();
    const int getTotalEpochs();

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
    float learningRate;
    int inputSize;
    int totalEpochs;
    std::vector<float> weights;
};

} // namespace ml

#endif // !PERCEPTRON_HPP
