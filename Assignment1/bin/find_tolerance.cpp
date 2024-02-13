#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>
#include <ostream>
#include <string>

int main(int argc, char** argv) {
    std::string path_1 = argv[1];
    std::string path_2 = argv[2];

    std::ifstream file_1(path_1, std::ios::binary);
    std::ifstream file_2(path_2, std::ios::binary);

    float largest_diff = 0.0;
    while (true) {
        float num_1, num_2;
        file_1.read((char *) &num_1, sizeof(float));
        file_2.read((char *) &num_2, sizeof(float));

        if (file_1.eof() || file_2.eof()) {
            break;
        }

        largest_diff = std::max(largest_diff, std::abs(num_1 - num_2));
    }

    if (file_1.eof() && file_2.eof()) {
        std::cout << "Largest difference: " << std::fixed << largest_diff << std::endl;
    } else {
        std::cout << "Files differ in length" << std::endl;
    }
}