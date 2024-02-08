# Perceptron Implementation

This program implements a Perceptron and a SIMD-optimized Perceptron (PerceptronSIMD) for classification tasks. The implementation uses C++ and leverages vectorized processing for enhanced performance.

## Prerequisites

- C++ compiler that supports C++11 or later.
- Ensure that the necessary dependencies are installed.

## Usage

1. **Clone the repository:**
   ```bash
   $ git clone <repository_url>
   ```

   - Clone the repository using your preferred version control tool.

2. **Navigate to the project directory:**
   ```bash
   $ cd <project_directory>
   ```

   - Use your terminal to go to the project directory.

3. **Compile the program:**
   ```bash
   $ g++ -o perceptron_main perceptron_main.cpp Perceptron.cpp PerceptronSIMD.cpp -std=c++11
   ```

   - Compile the program using a C++ compiler with support for C++11 or later.

4. **Run the executable:**
   ```bash
   $ ./perceptron_main [seed]
   ```

   - Run the executable file, optionally providing a seed as the first command-line argument.

## Purpose

This implementation was developed as part of a "warm-up" exercise for a more in-depth exploration of machine learning concepts in a forthcoming thesis.

## Dataset

- The Iris dataset is used for training. Ensure that the dataset file is available at "./data/iris.data".

## Configuration

- `LEARNING_RATE`: The learning rate for both Perceptrons is set to a constant value (float) in the source code.

## Acknowledgments

- Special thanks to my TCC advisor, Paulo Ricardo Lisboa de Almeida (prlameida.com), for guidance and support throughout the project.

## References

- [YouTube - Neural Networks Demystified](https://www.youtube.com/watch?v=OPizq3YRd0U)
- [SIMD: From Zero to Hero (PDF)](http://const.me/articles/simd/simd.pdf)
- [MIT Course Material on SIMD](http://mitran-lab.amath.unc.edu/courses/MATH547/lessons/Lesson03.pdf)
- [YouTube - Machine Learning 01: Introduction (Stanford)](https://www.youtube.com/watch?v=XiaIbmMGqdg)

### Books

1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Suíça: Springer New York.
2. Duda, R. O., Hart, P. E., Stork, D. G. (2012). Pattern Classification. Alemanha: Wiley.

### Essential Linear Algebra Series

- [YouTube - Essence of Linear Algebra (by 3Blue1Brown)](https://www.youtube.com/watch?v=kjBOesZCoqc&list=PL0-GT3co4r2y2YErbmuJw2L5tW4Ew2O5B)
