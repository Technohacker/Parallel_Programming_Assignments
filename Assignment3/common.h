#pragma once
#include <cstddef>
#include <vector>

// The type of elements in the input file
typedef int element_t;

// A range of elements, [start, end)
// Both are in terms of elements, not bytes
struct range_t {
    size_t start;
    size_t end;
};

// A view into a container, with a specified range
template<typename C>
class view {
private:
    C &container;
    range_t range;
public:
    view(C &container) : container(container), range(range_t { .start = 0, .end = container.size() }) {}
    view(C &container, range_t range) : container(container), range(range) {}

    auto &operator[](size_t i) {
        return container[range.start + i];
    }

    size_t size() {
        return range.end - range.start;
    }

    auto begin() {
        return container.begin() + range.start;
    }

    auto end() {
        return container.begin() + range.end;
    }

    bool from_same_container(view<C> &other) {
        return &container == &other.container;
    }

    view<C> slice(range_t new_range) {
        range_t final_range = {
            .start = range.start + new_range.start,
            .end = range.start + new_range.end,
        };
        return view<C>(container, final_range);
    }
};

template<typename T>
using vector_view = view<std::vector<T>>;

const size_t ELEMENT_SIZE = sizeof(element_t);

const size_t CPU_BLOCK_SIZE = 1024;
const size_t GPU_BLOCK_SIZE = 2048;
