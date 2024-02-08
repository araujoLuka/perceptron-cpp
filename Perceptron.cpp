#include "Perceptron.hpp"
#include <iostream>

namespace ml {

Perceptron::Perceptron(int inputSize, float leaningRate)
    : learningRate{leaningRate}, inputSize{inputSize}, totalEpochs{0} {
    srand(time(NULL));
    for (int i{0}; i < inputSize + 1; ++i)
        this->weights.push_back((float)rand() / (float)RAND_MAX);
}

std::vector<float> Perceptron::getWeights() { return this->weights; }

float Perceptron::activation(std::vector<float> input) {
    float scalarProduct{0};

    // Add the bias to scalarProduct
    scalarProduct += Perceptron::BIAS * this->weights[0];

    /* As defined in linear algebra:
     * - Definition. Given two real-valued (column) vectors u, v (both from R^n) the scalar product
     *   is the matrix-matrix multiplication -> u^T * v = SUM(u_i * v_i) from i=1 to n
     */
    for (int i{0}; i < this->inputSize; ++i) 
        scalarProduct += input[i] * this->weights[i + 1];

    return scalarProduct;
}

int Perceptron::predict(std::vector<float> input) { 
    return this->activation(input) >= 0 ? 1 : -1; 
}

int Perceptron::predict(float scalarProduct) {
    return scalarProduct >= 0 ? 1 : -1;
}

void Perceptron::fit(std::vector<std::vector<float>> trainingData, std::vector<int> labels, int epochs) {
    for (int epoch{0}; epoch < epochs; ++epoch) {
        std::cout << "Epoch: " << epoch + 1 << '/' << epochs << '\n';
        std::cout << "Initial weights: ";
        for (int i{0}; i < 5; ++i)
            std::cout << this->weights[i] << " ";
        std::cout << '\n';
        int misclassifiedCount{0};
        for (std::size_t i{0}; i < trainingData.size(); ++i) {
            float scalarProduct{this->activation(trainingData[i])};
            int prediction{this->predict(scalarProduct)};
            int label{labels[i]};
            // using criterion as suggested by bishop
            // only update weights if prediction is wrong
            if (prediction != label) {
                misclassifiedCount++;
                if (label == 1) {
                    this->weights[0] += this->learningRate;
                    for (int j{0}; j < this->inputSize; ++j)
                        this->weights[j + 1] += this->learningRate * trainingData[i][j];
                } else {
                    this->weights[0] -= this->learningRate;
                    for (int j{0}; j < this->inputSize; ++j)
                        this->weights[j + 1] -= this->learningRate * trainingData[i][j];
                }
            }
        }
        std::cout << "Misclassified: " << misclassifiedCount << '\n';
        std::cout << "Final weights: ";
        for (int i{0}; i < 5; ++i)
            std::cout << this->weights[i] << " ";
        std::cout << "\n\n";

        if (misclassifiedCount == 0) {
            this->totalEpochs = epoch + 1;
            break;
        }
    }
}

}  // namespace ml
