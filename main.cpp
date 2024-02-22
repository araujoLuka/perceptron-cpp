#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "Perceptron.hpp"
#include "PerceptronSIMD.hpp"

static const char *const FILE_NAME{"./data/iris.data"};
static constexpr float LEARNING_RATE{0.1};

void printHelp() {
    std::cout << "Usage: ./program [seed]" << std::endl;
    std::cout << "  seed: Optional seed for random number generation. If not "
                 "provided, default seed is used."
              << std::endl;
}

void readData(const std::string &fileName,
              std::vector<std::vector<float>> &data, std::vector<int> &labels) {
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Could not open file " << fileName << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) break;
        std::vector<float> d;
        std::string value;
        std::stringstream ss(line);
        for (int i{0}; i < 4; ++i) {
            std::getline(ss, value, ',');
            d.push_back(std::stof(value));
        }
        std::getline(ss, value);
        if (value == "Iris-setosa")
            labels.push_back(1);
        else if (value == "Iris-versicolor")
            labels.push_back(-1);
        else
            break;
        data.push_back(d);
    }
    file.close();
}

void generateWeights(int seed, std::vector<float> &weights) {
    srand(seed);
    for (int i{0}; i < 4; ++i)
        weights[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char *argv[]) {
    int seed{static_cast<int>(time(nullptr))};
    if (argc > 1) {
        for (int i{1}; i < argc; ++i) {
            std::string arg{argv[i]};
            if (arg == "-h" || arg == "--help") {
                printHelp();
                return 0;
            } 
        }
        seed = std::stoi(argv[1]);
    }

    std::vector<std::vector<float>> *trainingData{
        new std::vector<std::vector<float>>};
    std::vector<int> *labels{new std::vector<int>};
    std::chrono::time_point<std::chrono::system_clock> time;

    readData(FILE_NAME, *trainingData, *labels);

    std::cout << "Training data: " << trainingData->size() << std::endl;
    std::cout << "Labels: " << labels->size() << std::endl;

    std::vector<float> weights(4);
    generateWeights(seed, weights);

    ml::Perceptron perceptron{LEARNING_RATE, 4, weights};
    ml::PerceptronSIMD perceptronSIMD{LEARNING_RATE, 4, weights};

    std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "Input size: " << 4 << std::endl;
    std::cout << "---" << std::endl;

    std::cout << "Serial perceptron:" << std::endl;

    // Measure the time it takes to fit the perceptron with chrono
    time = std::chrono::system_clock::now();
    perceptron.fit(*trainingData, *labels, 1000);  // 1000 epochs
    int timeP1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::system_clock::now() - time)
                     .count();

    std::cout << "> Time: " << timeP1 << "ns" << std::endl;
    std::cout << "> Total epochs to full learn: " << perceptron.getTotalEpochs()
              << std::endl;
    std::cout << "> Weights: \n\t";
    std::cout << perceptron.getBiasWeight() << " ";
    for (int i{0}; i < 4; ++i) std::cout << perceptron.getWeights()[i] << " ";
    std::cout << std::endl;

    std::cout << "---" << std::endl;

    std::cout << "Perceptron SIMD:" << std::endl;
    time = std::chrono::system_clock::now();
    perceptronSIMD.fit(*trainingData, *labels, 1000);  // 1000 epochs
    int timeP2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::system_clock::now() - time)
                     .count();

    std::cout << "> Time: " << timeP2 << "ns" << std::endl;
    std::cout << "> Total epochs to full learn: "
              << perceptronSIMD.getTotalEpochs() << std::endl;
    std::cout << "> Weights: \n\t";
    std::cout << perceptronSIMD.getBiasWeight() << " ";
    for (int i{0}; i < 4; ++i) std::cout << perceptronSIMD.getWeights()[i] << " ";
    std::cout << std::endl;

    std::cout << "---" << std::endl;

    // Show the time difference between the two perceptrons
    // How much SIMD is faster than the scalar version in percentage
    if (timeP1 > timeP2) {
        std::cout << "Time difference: " << timeP1 - timeP2 << "ns"
                  << std::endl;
        std::cout << "SIMD is " << (float)timeP1 / timeP2
                  << "x faster than scalar" << std::endl;
    } else {
        std::cout << "Time difference: " << timeP2 - timeP1 << "ns"
                  << std::endl;
        std::cout << "SIMD failed to be faster than scalar" << std::endl;
        std::cout << "Scalar is " << (float)timeP2 / timeP1
                  << "x faster than SIMD" << std::endl;
    }

    delete trainingData;
    delete labels;

    return 0;
}
