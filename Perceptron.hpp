#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <vector>

namespace ml { // means machine learning

class Perceptron {
public:
    Perceptron(int inputSize, float leaningRate);
    virtual ~Perceptron() = default;

    /* Getters */
    std::vector<float> getWeights();

    /* Activation function
     * > Calculate the scalar product from inputs and weights
     */
    float activation(std::vector<float> inputs);

    /* Prediction function
     * > Uses the activation function to predict the output
     */
    int predict(std::vector<float> inputs);

    /* Prediction function
    * > Uses the activation function with the before calculated scalar product to predict the output
     */
    int predict(float scalarProduct);

    /* Training function
     * > Finds the ideal weights for the perceptron to make accurate predictions
     */
    void fit(std::vector<std::vector<float>> trainingData, std::vector<int> labels, int epochs);

private:
    static constexpr int BIAS{1};

    float learningRate;
    int inputSize;
    int totalEpochs;
    std::vector<float> weights;
};

} // namespace ml

#endif // !PERCEPTRON_HPP
