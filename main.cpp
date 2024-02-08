#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include "Perceptron.hpp"

static const char *const FILE_NAME{"./iris.data"};

int main (int argc, char *argv[]) {
    std::vector<std::vector<float>> *trainingData{new std::vector<std::vector<float>>};
    std::vector<int> *labels{new std::vector<int>};

    std::ifstream file(FILE_NAME);
    if (!file.is_open()) {
        std::cerr << "Could not open file " << FILE_NAME << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty())
            break;

        std::vector<float> data;
        std::string value;
        std::stringstream ss(line);
        for (int i{0}; i < 4; ++i) {
            std::getline(ss, value, ',');
            data.push_back(std::stof(value));
        }
        std::getline(ss, value);
        if (value == "Iris-setosa")
            labels->push_back(1);
        else if (value == "Iris-versicolor")
            labels->push_back(-1);
        else
            break;
        trainingData->push_back(data);
    }
    file.close();

    std::cout << "Training data: " << trainingData->size() << std::endl;
    std::cout << "Labels: " << labels->size() << std::endl;

    // ml::Perceptron perceptron{4, 0.01}; // 4 inputs, learning rate 0.01
    ml::Perceptron perceptron{4, 1}; // 4 inputs, learning rate 1

    // Measure the time it takes to fit the perceptron
    float time{0};
    perceptron.fit(*trainingData, *labels, 1000); // 1000 epochs
    time = clock() - time;
    std::cout << "Time: " << time / CLOCKS_PER_SEC << "s" << std::endl;

    std::cout << "Weights: ";
    for (int i{0}; i < 5; ++i)
        std::cout << perceptron.getWeights()[i] << " ";

    std::cout << std::endl;

    std::cout << "Prediction: " << perceptron.predict(std::vector<float>{5.1, 3.5, 1.4, 0.2}) << std::endl;
    std::cout << "Correct: 1" << std::endl;
    std::cout << "Prediction: " << perceptron.predict(std::vector<float>{5.9, 3.0, 5.1, 1.8}) << std::endl;
    std::cout << "Correct: -1" << std::endl;

    return 0;
}
