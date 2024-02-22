#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Perceptron.hpp"
#include "PerceptronSIMD.hpp"

void printHelp() {
    std::cout << "Usage: ./program <options> [seed]\n\n";
    std::cout << "Options: \n"
              << "  -h, --help: Show this help message and exit.\n"
              << "  --random-weights: Generate random weights for the perceptron "
                 "                    instead of reading from standard input.\n\n";
    std::cout << "Arguments: \n"
              << "  seed: Optional seed for random number generation. If not "
                 "        provided, default seed is used.\n\n";
    std::cout << "Description: \n"
              << "  Reads a dataset from a file and trains a "
                 "  perceptron using the given data.\n";
    std::cout << "\nInputs: \n"
              << "  - Number of features\n"
              << "  - Learning rate\n"
              << "  - File name\n"
              << "  - Weights (optional)\n";
    std::cout << "Outputs: \n"
              << "  - Time to train the perceptron\n"
              << "  - Total epochs to full learn\n"
              << "  - Final weights\n";
}

void readData(std::vector<std::vector<float>> &data, std::vector<int> &labels,
              int features, std::string fileName) {
    std::string label1{""}, label2{""};
    int intances{0};

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
        for (int i{0}; i < features; ++i) {
            std::getline(ss, value, ',');
            d.push_back(std::stof(value));
        }
        std::getline(ss, value);
        if (label1.empty()) {
            label1 = value;
#ifdef DEBUG
            std::cout << "Label 1: " << label1 << std::endl;
#endif
        } else if (label2.empty() && value != label1) {
            label2 = value;
#ifdef DEBUG
            std::cout << "Label 2: " << label2 << std::endl;
#endif
        }
        if (value == label1)
            labels.push_back(1);
        else if (value == label2)
            labels.push_back(-1);
        else
            continue;
        data.push_back(d);
        ++intances;
    }
    file.close();
#ifdef DEBUG
    std::cout << "Read " << intances << " instances\n";
#endif
}

void generateWeights(int seed, std::vector<float> &weights, int features) {
    srand(seed);
    for (int i{0}; i < features; ++i)
        weights[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char *argv[]) {
    int seed{static_cast<int>(time(nullptr))};
    bool randomWeights{false};

    if (argc > 1) {
        for (int i{1}; i < argc; ++i) {
            std::string arg{argv[i]};
            if (arg == "-h" || arg == "--help") {
                printHelp();
                return 0;
            } else if (arg == "--random-weights") {
                randomWeights = true;
            }
        }
        seed = std::stoi(argv[1]);
    }

    std::chrono::time_point<std::chrono::system_clock> time;

    int features;
    float learningRate;
    std::string fileName;

#ifdef DEBUG
    std::cout << "Features: ";
#endif
    std::cin >> features;
#ifdef DEBUG
    std::cout << "Learning rate: ";
#endif
    std::cin >> learningRate;

    std::vector<std::vector<float>> *trainingData =
        new std::vector<std::vector<float>>();
    std::vector<int> *labels = new std::vector<int>();
    std::vector<float> weights(features);

#ifdef DEBUG
    std::cout << "File name: ";
#endif
    std::cin >> fileName;

    readData(*trainingData, *labels, features, fileName);

    if (randomWeights) {
#ifdef DEBUG
        std::cout << "Seed: " << seed << std::endl;
        std::cout << "Generating random weights...\n";
#endif

        generateWeights(seed, weights, features);

#ifdef DEBUG
        std::cout << "Bias weight: " << weights[0] << std::endl;
        std::cout << "Weights: ";
#else
        std::cout << weights[0] << " ";
#endif
        for (int i{1}; i < features; ++i) std::cout << weights[i] << " ";
        std::cout << std::endl;
    } else {
#ifdef DEBUG
        std::cout << "Bias weight: ";
#endif
        std::cin >> weights[0];
#ifdef DEBUG
        std::cout << "Weights (" << features << "): ";
#endif
        for (int i{1}; i < features; ++i) std::cin >> weights[i];
    }

    ml::Perceptron perceptron{learningRate, features, weights};
    ml::PerceptronSIMD perceptronSIMD{learningRate, features, weights};

#ifdef DEBUG
    std::cout << "---\n";
    std::cout << "Serial perceptron:\n";
#endif

    // Measure the time it takes to fit the perceptron with chrono
    time = std::chrono::system_clock::now();
    perceptron.fit(*trainingData, *labels, 1000);  // 1000 epochs
    int timeP1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::system_clock::now() - time)
                     .count();

    std::cout << "> Time: " << timeP1 << "ns\n";
    std::cout << "> Total epochs to full learn: " << perceptron.getTotalEpochs()
              << std::endl;
    std::cout << "> Weights: \n\t";
    std::cout << perceptron.getBiasWeight() << " ";
    for (int i{0}; i < 4; ++i) std::cout << perceptron.getWeights()[i] << " ";
    std::cout << std::endl;

    std::cout << "---\n";

    std::cout << "Perceptron SIMD:\n";
    time = std::chrono::system_clock::now();
    perceptronSIMD.fit(*trainingData, *labels, 1000);  // 1000 epochs
    int timeP2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::system_clock::now() - time)
                     .count();

    std::cout << "> Time: " << timeP2 << "ns\n";
    std::cout << "> Total epochs to full learn: "
              << perceptronSIMD.getTotalEpochs() << std::endl;
    std::cout << "> Weights: \n\t";
    std::cout << perceptronSIMD.getBiasWeight() << " ";
    for (int i{0}; i < 4; ++i)
        std::cout << perceptronSIMD.getWeights()[i] << " ";
    std::cout << std::endl;

    std::cout << "---\n";

    // Show the time difference between the two perceptrons
    // How much SIMD is faster than the scalar version in percentage
    if (timeP1 > timeP2) {
        std::cout << "Time difference: " << timeP1 - timeP2 << "ns"
                  << std::endl;
        std::cout << "SIMD is " << (float)timeP1 / timeP2
                  << "x faster than scalar\n";
    } else {
        std::cout << "Time difference: " << timeP2 - timeP1 << "ns"
                  << std::endl;
        std::cout << "SIMD failed to be faster than scalar\n";
        std::cout << "Scalar is " << (float)timeP2 / timeP1
                  << "x faster than SIMD\n";
    }

    delete trainingData;
    delete labels;

    return 0;
}
