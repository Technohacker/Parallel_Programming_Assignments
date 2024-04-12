// Reads a file of integers and verifies that the integers are in sorted order,
// while also ensuring that the frequency of each integer is the same as the original file.

#include <array>
#include <cstddef>
#include <iostream>
#include <fstream>

#include "../common.h"

typedef std::array<size_t, ((size_t) 1) << ELEMENT_SIZE * 8> freq_t;

bool check_and_decrement(freq_t* freq, element_t elem) {
    if ((*freq)[elem] == 0) {
        return false;
    }

    (*freq)[elem]--;
    return true;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <original file> <sorted file>" << std::endl;
        return 1;
    }

    std::string original_file = argv[1];
    std::string sorted_file = argv[2];

    std::ifstream original(original_file, std::ios::binary);
    if (!original.is_open()) {
        std::cerr << "Failed to open original file." << std::endl;
        return 1;
    }

    std::ifstream sorted(sorted_file, std::ios::binary);
    if (!sorted.is_open()) {
        std::cerr << "Failed to open sorted file." << std::endl;
        return 1;
    }

    long original_num_elements, sorted_num_elements;
    original.read(reinterpret_cast<char*>(&original_num_elements), sizeof(long));
    sorted.read(reinterpret_cast<char*>(&sorted_num_elements), sizeof(long));

    if (original_num_elements != sorted_num_elements) {
        std::cerr << "Original file has " << original_num_elements << " elements, but sorted file has " << sorted_num_elements << " elements." << std::endl;
        return 1;
    }

    freq_t *histogram = new freq_t();

    element_t elem, last_elem;
    while (original.read(reinterpret_cast<char*>(&elem), ELEMENT_SIZE)) {
        (*histogram)[elem]++;
    }

    bool first = true;
    while (sorted.read(reinterpret_cast<char*>(&elem), ELEMENT_SIZE)) {
        if (sorted.eof()) {
            std::cerr << "Sorted file is shorter than the original file." << std::endl;
            return 1;
        }

        if (first) {
            first = false;
        } else {
            if (elem < last_elem) {
                std::cerr << "Sorted file is not in sorted order. Element " << elem << " is less than " << last_elem << " at position " << sorted.tellg() << "." << std::endl;
                return 1;
            }
        }
        last_elem = elem;

        if (!check_and_decrement(histogram, elem)) {
            std::cerr << "Element " << elem << " is missing from the original file." << std::endl;
            return 1;
        }
    }

    for (size_t i = 0; i < histogram->size(); i++) {
        if ((*histogram)[i] != 0) {
            std::cerr << "Element " << i << " is missing from the sorted file." << std::endl;
            return 1;
        }
    }

    std::cout << "File is correctly sorted." << std::endl;
    delete histogram;

    return 0;
}