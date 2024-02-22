#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include "Perceptron.hpp"
#include "PerceptronSIMD.hpp"

static const char *const FILE_NAME{"./data/iris.data"};
static constexpr float LEARNING_RATE{0.01};

void printHelp() {
    std::cout << "Usage: ./program [seed]" << std::endl;
    std::cout << "  seed: Optional seed for random number generation. If not provided, default seed is used." << std::endl;
}

int main(int argc, char *argv[]) {
    int seed{static_cast<int>(time(nullptr))};
    if (argc > 1) {
        if (std::string{argv[1]} == "-h" || std::string{argv[1]} == "--help") {
            printHelp();
            return 0;
        }

        try {
            seed = std::stoi(argv[1]);
        } catch (const std::invalid_argument &e) {
            std::cerr << "Invalid seed provided. Please provide a valid seed." << std::endl;
            return -1;
        } catch (const std::out_of_range &e) {
            std::cerr << "Seed out of range. Please provide a valid seed." << std::endl;
            return -2;
        }
    }

    std::vector<std::vector<float>> *trainingData{new std::vector<std::vector<float>>};
    std::vector<int> *labels{new std::vector<int>};
    std::chrono::time_point<std::chrono::system_clock> time;

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

    srand(seed);
    std::vector<float> weights{4, 0};
    for (int i{0}; i < 4; ++i)
        weights[i] = static_cast<float>(rand()) / RAND_MAX;

    ml::Perceptron perceptron{LEARNING_RATE, 4, weights};
    ml::PerceptronSIMD perceptronSIMD{LEARNING_RATE, 4, weights};

    std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "Input size: " << 4 << std::endl;

    std::cout << "Training serial perceptron..." << std::endl;

    // Measure the time it takes to fit the perceptron with chrono
    time = std::chrono::system_clock::now();
    perceptron.fit(*trainingData, *labels, 1000); // 1000 epochs
    int timeP1 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - time).count();

    std::cout << "Time: " << timeP1 << "ns" << std::endl;
    std::cout << "Total epochs to full learn: " << perceptron.getTotalEpochs() << std::endl;
    std::cout << "Weights: ";
    std::cout << perceptron.getBiasWeight() << " ";
    for (int i{0}; i < 4; ++i)
        std::cout << perceptron.getWeights()[i] << " ";

    std::cout << std::endl;

    std::cout << "Prediction: " << perceptron.predict(std::vector<float>{5.1, 3.5, 1.4, 0.2}) << std::endl;
    std::cout << "Correct: 1" << std::endl;
    std::cout << "Prediction: " << perceptron.predict(std::vector<float>{5.9, 3.0, 5.1, 1.8}) << std::endl;
    std::cout << "Correct: -1" << std::endl;

    std::cout << "---" << std::endl;

    std::cout << "Training perceptron SIMD..." << std::endl;
    time = std::chrono::system_clock::now();
    perceptronSIMD.fit(*trainingData, *labels, 1000); // 1000 epochs
    int timeP2 = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - time).count();

    std::cout << "Time: " << timeP2 << "ns" << std::endl;
    std::cout << "Total epochs to full learn: " << perceptronSIMD.getTotalEpochs() << std::endl;
    std::cout << "Weights: ";
    std::cout << perceptronSIMD.getBiasWeight() << " ";
    for (int i{0}; i < 4; ++i)
        std::cout << perceptronSIMD.getWeights()[i] << " ";
    std::cout << std::endl;

    std::cout << "Prediction: " << perceptron.predict(std::vector<float>{5.1, 3.5, 1.4, 0.2}) << std::endl;
    std::cout << "Correct: 1" << std::endl;
    std::cout << "Prediction: " << perceptron.predict(std::vector<float>{5.9, 3.0, 5.1, 1.8}) << std::endl;
    std::cout << "Correct: -1" << std::endl;

    std::cout << "---" << std::endl;

    // Show the time difference between the two perceptrons
    // How much SIMD is faster than the scalar version in percentage
    if (timeP1 > timeP2) {
        std::cout << "Time difference: " << timeP1 - timeP2 << "ns" << std::endl;
        std::cout << "SIMD is " << (float)timeP1 / timeP2 * 100 << "% faster than scalar" << std::endl;
    }
    else {
        std::cout << "Time difference: " << timeP2 - timeP1 << "ns" << std::endl;
        std::cout << "SIMD failed to be faster than scalar" << std::endl;
        std::cout << "Scalar is " << (float)timeP2 / timeP1 * 100 << "% faster than SIMD" << std::endl;
    }

    delete trainingData;
    delete labels;

    return 0;
}
