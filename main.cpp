#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Perceptron.hpp"
#include "PerceptronSIMD.hpp"

void printHelp() {
    std::cout << "Usage: ./program [seed]" << std::endl;
    std::cout << "  seed: Optional seed for random number generation. If not "
                 "provided, default seed is used."
              << "\n\n"
              << "Description: " << std::endl
              << "  Reads a dataset from a file and trains a "
                 "  perceptron using the given data. "
              << std::endl;
}

void readData(std::vector<std::vector<float>> &data, std::vector<int> &labels,
              int features) {
    std::string fileName, label1{""}, label2{""};
    int intances{0};

    std::cout << "File name: ";
    std::cin >> fileName;

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
            std::cout << "Label 1: " << label1 << std::endl;
        }
        else if (label2.empty() && value != label1) {
            label2 = value;
            std::cout << "Label 2: " << label2 << std::endl;
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
    std::cout << "Read " << intances << " instances" << std::endl;
}

void generateWeights(int seed, std::vector<float> &weights, int features) {
    srand(seed);
    for (int i{0}; i < features; ++i)
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

    std::chrono::time_point<std::chrono::system_clock> time;

    int features;
    float learningRate;
    std::string label1, label2;

    std::cout << "Features: ";
    std::cin >> features;
    std::cout << "Learning rate: ";
    std::cin >> learningRate;

    std::vector<std::vector<float>> *trainingData =
        new std::vector<std::vector<float>>();
    std::vector<int> *labels = new std::vector<int>();

    readData(*trainingData, *labels, features);

    std::vector<float> weights(features);
    std::cout << "Seed: " << seed << std::endl;
    std::cout << "Generate random weights? [y/n]: ";
    char answer;
    std::cin >> answer;
    if (answer == 'y') {
        generateWeights(seed, weights, features);
        std::cout << "Bias weight: " << weights[0] << std::endl;
        std::cout << "Weights: ";
        for (int i{1}; i < features; ++i) std::cout << weights[i] << " ";
        std::cout << std::endl;
    }
    else {
        std::cout << "Bias weight: ";
        std::cin >> weights[0];
        std::cout << "Enter '" << features << "' weights: ";
        for (int i{1}; i < features; ++i) std::cin >> weights[i];
    }

    ml::Perceptron perceptron{learningRate, features, weights};
    ml::PerceptronSIMD perceptronSIMD{learningRate, features, weights};

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
    for (int i{0}; i < 4; ++i)
        std::cout << perceptronSIMD.getWeights()[i] << " ";
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
