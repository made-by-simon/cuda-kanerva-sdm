/**
 * Sparse Distributed Memory implementation based on Kanerva (1992).
 *
 * This module implements the fundamental operations of Kanerva's Sparse Distributed
 * Memory (SDM) model, including writing, reading, and erasing memories, based on 
 * Hamming distance activation.
 *
 * Reference:
 *     Pentti Kanerva (1992). Sparse Distributed Memory and Related Models.
 *
 * (c) 2025 Simon Wong
 */

#ifndef KANERVA_SDM_H
#define KANERVA_SDM_H

#include <vector>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <string>

class KanervaSDM {
public:
    /**
     * Initializes the Kanerva SDM.
     *
     * @param address_dimension Length of address vectors (N).
     * @param memory_dimension Length of memory vectors (U).
     * @param num_locations Number of hard locations (M).
     * @param activation_threshold Hamming distance threshold for activation (H).
     * @param random_seed Seed for reproducible random generation of hard locations.
     *
     * @throws std::invalid_argument If any dimension or threshold is non-positive.
     */
    KanervaSDM(int address_dimension, 
               int memory_dimension, 
               int num_locations, 
               int activation_threshold, 
               unsigned int random_seed = 42)
        : address_dimension_(address_dimension),
          memory_dimension_(memory_dimension),
          num_locations_(num_locations),
          activation_threshold_(activation_threshold),
          memory_count_(0) {
        
        if (address_dimension <= 0 || memory_dimension <= 0 || num_locations <= 0) {
            throw std::invalid_argument("All dimensions must be positive integers.");
        }
        if (activation_threshold < 0) {
            throw std::invalid_argument("Activation threshold must be non-negative.");
        }

        // Initialize random number generator.
        std::mt19937 rng(random_seed);
        std::uniform_int_distribution<int> dist(0, 1);

        // Initialize address matrix with random binary values.
        address_matrix_.resize(num_locations_, std::vector<int>(address_dimension_));
        for (int i = 0; i < num_locations_; ++i) {
            for (int j = 0; j < address_dimension_; ++j) {
                address_matrix_[i][j] = dist(rng);
            }
        }

        // Initialize memory matrix with zeros.
        memory_matrix_.resize(num_locations_, std::vector<float>(memory_dimension_, 0.0f));
    }

    /**
     * Writes a memory to an address.
     *
     * @param address Target address vector (x) of size address_dimension.
     * @param memory Memory vector (w) of size memory_dimension.
     *
     * @throws std::invalid_argument If address or memory vectors are invalid.
     */
    void write(const std::vector<int>& address, const std::vector<int>& memory) {
        validate_vector(address, "address", address_dimension_);
        validate_vector(memory, "memory", memory_dimension_);

        std::vector<int> activated_locations = get_activated_locations(address);

        // Convert memory to polar form and update memory matrix.
        for (int loc : activated_locations) {
            for (int i = 0; i < memory_dimension_; ++i) {
                int polar_value = 2 * memory[i] - 1;  // Convert {0,1} to {-1,+1}.
                memory_matrix_[loc][i] += polar_value;
            }
        }

        ++memory_count_;
    }

    /**
     * Reads a memory from an address.
     *
     * @param address Target address vector (x) of size address_dimension.
     *
     * @return Recalled memory vector (z) of size memory_dimension.
     *         Returns all zeros if no locations are activated.
     *
     * @throws std::invalid_argument If address vector is invalid.
     */
    std::vector<int> read(const std::vector<int>& address) {
        validate_vector(address, "address", address_dimension_);

        std::vector<int> activated_locations = get_activated_locations(address);

        // Return zeros if no locations activated.
        if (activated_locations.empty()) {
            return std::vector<int>(memory_dimension_, 0);
        }

        // Sum activated locations.
        std::vector<float> locations_sum(memory_dimension_, 0.0f);
        for (int loc : activated_locations) {
            for (int i = 0; i < memory_dimension_; ++i) {
                locations_sum[i] += memory_matrix_[loc][i];
            }
        }

        // Convert to binary output.
        std::vector<int> result(memory_dimension_);
        for (int i = 0; i < memory_dimension_; ++i) {
            result[i] = (locations_sum[i] >= 0.0f) ? 1 : 0;
        }

        return result;
    }

    /**
     * Erases memory matrix (C), but NOT address matrix (A),
     * so locations are preserved.
     */
    void erase_memory() {
        for (auto& row : memory_matrix_) {
            std::fill(row.begin(), row.end(), 0.0f);
        }
        memory_count_ = 0;
    }

    // Getters for accessing dimensions and count.
    int get_address_dimension() const { return address_dimension_; }
    int get_memory_dimension() const { return memory_dimension_; }
    int get_num_locations() const { return num_locations_; }
    int get_activation_threshold() const { return activation_threshold_; }
    int get_memory_count() const { return memory_count_; }

private:
    int address_dimension_;      // Length of addresses (N).
    int memory_dimension_;       // Length of memories (U).
    int num_locations_;          // Number of locations (M).
    int activation_threshold_;   // Hamming activation threshold (H).
    int memory_count_;           // Number of stored memories (T).

    std::vector<std::vector<int>> address_matrix_;  // Hard locations (A).
    std::vector<std::vector<float>> memory_matrix_; // Memory counters (C).

    /**
     * Finds activated locations based on Hamming distance threshold (H).
     *
     * @param address Target address vector (x) of size address_dimension.
     *
     * @return Vector of indices for activated locations (y).
     *
     * @throws std::invalid_argument If address shape doesn't match address_dimension.
     */
    std::vector<int> get_activated_locations(const std::vector<int>& address) {
        std::vector<int> activated_locations;

        for (int i = 0; i < num_locations_; ++i) {
            // Calculate Hamming distance.
            int hamming_distance = 0;
            for (int j = 0; j < address_dimension_; ++j) {
                if (address_matrix_[i][j] != address[j]) {
                    ++hamming_distance;
                }
            }

            // Check if location is activated.
            if (hamming_distance <= activation_threshold_) {
                activated_locations.push_back(i);
            }
        }

        return activated_locations;
    }

    /**
     * Validates that an address vector or memory vector has the correct dimension
     * and contains only binary values.
     *
     * @param vector Vector to validate.
     * @param vector_name Name of the vector for error message.
     * @param expected_dimension Expected size of the vector.
     *
     * @throws std::invalid_argument If vector dimension is incorrect or contains non-binary values.
     */
    void validate_vector(const std::vector<int>& vector, 
                        const std::string& vector_name, 
                        int expected_dimension) {
        if (static_cast<int>(vector.size()) != expected_dimension) {
            throw std::invalid_argument(
                vector_name + " size " + std::to_string(vector.size()) + 
                " doesn't match expected (" + std::to_string(expected_dimension) + ")"
            );
        }

        for (int val : vector) {
            if (val != 0 && val != 1) {
                throw std::invalid_argument(vector_name + " must contain only 0s and 1s");
            }
        }
    }
};

#endif // KANERVA_SDM_H