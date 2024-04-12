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
    C *container;
    range_t range;
public:
    view() : container(nullptr), range(range_t { .start = 0, .end = 0 }) {}
    view(C &container) : container(&container), range(range_t { .start = 0, .end = container.size() }) {}
    view(C &container, range_t range) : container(&container), range(range) {}

    auto &operator[](size_t i) {
        assert(container != nullptr);
        return (*container)[range.start + i];
    }

    size_t size() {
        return range.end - range.start;
    }

    auto begin() {
        assert(container != nullptr);
        return (*container).begin() + range.start;
    }

    auto end() {
        assert(container != nullptr);
        return (*container).begin() + range.end;
    }

    bool from_same_container(view<C> &other) {
        return (container == other.container) && (container != nullptr);
    }

    bool is_contiguous(view<C> &other) {
        return from_same_container(other) && range.end == other.range.start;
    }

    view<C> slice(range_t new_range) {
        range_t final_range = {
            .start = range.start + new_range.start,
            .end = range.start + new_range.end,
        };

        view<C> new_view;
        new_view.container = container;
        new_view.range = final_range;
        return new_view;
    }

    view<C> merge(view<C> &other) {
        // Ensure that the views are contiguous and from the same container
        assert(is_contiguous(other));

        range_t final_range = {
            .start = range.start,
            .end = other.range.end,
        };

        view<C> new_view;
        new_view.container = container;
        new_view.range = final_range;
        return new_view;
    }
};

template<typename T>
using vector_view = view<std::vector<T>>;

const size_t ELEMENT_SIZE = sizeof(element_t);

const size_t CPU_BLOCK_SIZE = 1024;
const size_t GPU_BLOCK_SIZE = 2048;
