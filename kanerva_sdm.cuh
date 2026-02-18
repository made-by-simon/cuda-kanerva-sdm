/**
 * CUDA-accelerated Sparse Distributed Memory (Kanerva SDM).
 * Header-only library with batching support and persistent device memory.
 *
 * (c) 2026 Simon Wong.
 */

#ifndef KANERVA_SDM_CUH
#define KANERVA_SDM_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cstring>

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Computes Hamming distances and identifies activated locations for a batch.
 */
__global__ void compute_activated_locations_batch_kernel(
    const int* address_matrix,
    const int* addresses,
    int* activated_flags,
    int num_locations,
    int address_dimension,
    int hamming_threshold,
    int batch_size
) {
    int batch_idx = blockIdx.y;
    int loc_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && loc_idx < num_locations) {
        int hamming_distance = 0;

        // Compute Hamming distance for this location and batch item.
        const int* current_address = addresses + batch_idx * address_dimension;
        for (int j = 0; j < address_dimension; ++j) {
            int addr_val = address_matrix[loc_idx * address_dimension + j];
            if (addr_val != current_address[j]) {
                ++hamming_distance;
            }
        }

        // Set activation flag.
        activated_flags[batch_idx * num_locations + loc_idx] =
            (hamming_distance <= hamming_threshold) ? 1 : 0;
    }
}

/**
 * Writes batch of memories to activated locations using polar encoding.
 */
__global__ void write_memory_batch_kernel(
    float* memory_matrix,
    const int* memories,
    const int* activated_flags,
    int num_locations,
    int memory_dimension,
    int batch_size
) {
    int batch_idx = blockIdx.y;
    int loc_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && loc_idx < num_locations) {
        if (activated_flags[batch_idx * num_locations + loc_idx] == 1) {
            const int* current_memory = memories + batch_idx * memory_dimension;
            for (int i = 0; i < memory_dimension; ++i) {
                int polar_value = 2 * current_memory[i] - 1;
                atomicAdd(&memory_matrix[loc_idx * memory_dimension + i],
                         static_cast<float>(polar_value));
            }
        }
    }
}

/**
 * Reads batch of memories by summing activated locations.
 */
__global__ void read_memory_batch_kernel(
    const float* memory_matrix,
    const int* activated_flags,
    float* output_sums,
    int num_locations,
    int memory_dimension,
    int batch_size
) {
    int batch_idx = blockIdx.y;
    int mem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && mem_idx < memory_dimension) {
        float sum = 0.0f;

        // Sum across activated locations for this batch item and memory dimension.
        const int* current_flags = activated_flags + batch_idx * num_locations;
        for (int loc = 0; loc < num_locations; ++loc) {
            if (current_flags[loc] == 1) {
                sum += memory_matrix[loc * memory_dimension + mem_idx];
            }
        }

        output_sums[batch_idx * memory_dimension + mem_idx] = sum;
    }
}

/**
 * Converts batch of summed memories to binary outputs.
 */
__global__ void threshold_output_batch_kernel(
    const float* sum_vectors,
    int* outputs,
    int memory_dimension,
    int batch_size
) {
    int batch_idx = blockIdx.y;
    int mem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx < batch_size && mem_idx < memory_dimension) {
        int idx = batch_idx * memory_dimension + mem_idx;
        outputs[idx] = (sum_vectors[idx] >= 0.0f) ? 1 : 0;
    }
}

// ============================================================================
// Kernel Launch Wrappers
// ============================================================================

inline void launch_compute_activated_locations_batch(
    const int* d_address_matrix,
    const int* d_addresses_batch,
    int* d_activated_flags_batch,
    int num_locations,
    int address_dimension,
    int hamming_threshold,
    int batch_size
) {
    int block_size = 256;
    dim3 grid_locations((num_locations + block_size - 1) / block_size, batch_size);

    compute_activated_locations_batch_kernel<<<grid_locations, block_size>>>(
        d_address_matrix,
        d_addresses_batch,
        d_activated_flags_batch,
        num_locations,
        address_dimension,
        hamming_threshold,
        batch_size
    );
}

inline void launch_write_memory_batch(
    float* d_memory_matrix,
    const int* d_memories_batch,
    const int* d_activated_flags_batch,
    int num_locations,
    int memory_dimension,
    int batch_size
) {
    int block_size = 256;
    dim3 grid_locations((num_locations + block_size - 1) / block_size, batch_size);

    write_memory_batch_kernel<<<grid_locations, block_size>>>(
        d_memory_matrix,
        d_memories_batch,
        d_activated_flags_batch,
        num_locations,
        memory_dimension,
        batch_size
    );
}

inline void launch_read_memory_batch(
    const float* d_memory_matrix,
    const int* d_activated_flags_batch,
    float* d_output_sums_batch,
    int num_locations,
    int memory_dimension,
    int batch_size
) {
    int block_size = 256;
    dim3 grid_memory((memory_dimension + block_size - 1) / block_size, batch_size);

    read_memory_batch_kernel<<<grid_memory, block_size>>>(
        d_memory_matrix,
        d_activated_flags_batch,
        d_output_sums_batch,
        num_locations,
        memory_dimension,
        batch_size
    );
}

inline void launch_threshold_output_batch(
    const float* d_sum_vectors,
    int* d_outputs,
    int memory_dimension,
    int batch_size
) {
    int block_size = 256;
    dim3 grid_memory((memory_dimension + block_size - 1) / block_size, batch_size);

    threshold_output_batch_kernel<<<grid_memory, block_size>>>(
        d_sum_vectors,
        d_outputs,
        memory_dimension,
        batch_size
    );
}

// ============================================================================
// KanervaSDMCUDA Class
// ============================================================================

class KanervaSDMCUDA {
public:
    /**
     * Initializes the CUDA-accelerated Kanerva SDM.
     *
     * @param address_dimension Length of address vectors (N).
     * @param memory_dimension Length of memory vectors (U).
     * @param num_locations Number of hard locations (M).
     * @param hamming_threshold Hamming distance threshold for activation (H).
     * @param max_batch_size Maximum batch size for operations (default: 1024).
     * @param random_seed Seed for reproducible random generation of hard locations.
     *
     * @throws std::invalid_argument If any dimension or threshold is non-positive.
     * @throws std::runtime_error If CUDA allocation fails.
     */
    KanervaSDMCUDA(int address_dimension,
                   int memory_dimension,
                   int num_locations,
                   int hamming_threshold,
                   int max_batch_size = 1024,
                   unsigned int random_seed = 42);

    ~KanervaSDMCUDA();

    /**
     * Writes a single memory to an address.
     */
    void write(const std::vector<int>& address,
               const std::vector<int>& memory,
               bool sync_after = true);

    /**
     * Writes a batch of memories to addresses.
     * More efficient than multiple single write calls.
     */
    void write_batch(const std::vector<std::vector<int>>& addresses,
                     const std::vector<std::vector<int>>& memories,
                     bool sync_after = true);

    /**
     * Reads a single memory from an address.
     */
    std::vector<int> read(const std::vector<int>& address,
                          bool sync_before = true);

    /**
     * Reads a batch of memories from addresses.
     * More efficient than multiple single read calls.
     */
    std::vector<std::vector<int>> read_batch(
        const std::vector<std::vector<int>>& addresses,
        bool sync_before = true);

    /**
     * Synchronizes GPU operations.
     */
    void synchronize();

    /**
     * Erases memory matrix, preserving address matrix.
     */
    void erase_memory(bool sync_after = true);

    // Getters.
    int get_address_dimension() const { return address_dimension_; }
    int get_memory_dimension() const { return memory_dimension_; }
    int get_num_locations() const { return num_locations_; }
    int get_hamming_threshold() const { return hamming_threshold_; }
    int get_memory_count() const { return memory_count_; }
    int get_max_batch_size() const { return max_batch_size_; }

private:
    int address_dimension_;
    int memory_dimension_;
    int num_locations_;
    int hamming_threshold_;
    int memory_count_;
    int max_batch_size_;

    // Device pointers (persistent).
    int* d_address_matrix_;
    float* d_memory_matrix_;

    // Device buffers for batching (pre-allocated).
    int* d_addresses_batch_;
    int* d_memories_batch_;
    int* d_activated_flags_batch_;
    float* d_output_sums_batch_;
    int* d_outputs_batch_;

    void validate_vector(const std::vector<int>& vector,
                        const std::string& vector_name,
                        int expected_dimension);

    void validate_batch(const std::vector<std::vector<int>>& batch,
                       const std::string& batch_name,
                       int expected_dimension);

    void check_cuda_error(cudaError_t error, const std::string& message);

    void write_batch_internal(const int* h_addresses,
                             const int* h_memories,
                             int batch_size,
                             bool sync_after);

    void read_batch_internal(const int* h_addresses,
                            int* h_outputs,
                            int batch_size,
                            bool sync_before);
};

// ============================================================================
// KanervaSDMCUDA Implementation
// ============================================================================

inline KanervaSDMCUDA::KanervaSDMCUDA(
    int address_dimension,
    int memory_dimension,
    int num_locations,
    int hamming_threshold,
    int max_batch_size,
    unsigned int random_seed
) : address_dimension_(address_dimension),
    memory_dimension_(memory_dimension),
    num_locations_(num_locations),
    hamming_threshold_(hamming_threshold),
    memory_count_(0),
    max_batch_size_(max_batch_size),
    d_address_matrix_(nullptr),
    d_memory_matrix_(nullptr),
    d_addresses_batch_(nullptr),
    d_memories_batch_(nullptr),
    d_activated_flags_batch_(nullptr),
    d_output_sums_batch_(nullptr),
    d_outputs_batch_(nullptr) {

    // Validate parameters.
    if (address_dimension <= 0) {
        throw std::invalid_argument("Address dimension must be a positive integer.");
    }
    if (memory_dimension <= 0) {
        throw std::invalid_argument("Memory dimension must be a positive integer.");
    }
    if (num_locations <= 0) {
        throw std::invalid_argument("Number of locations must be a positive integer.");
    }
    if (hamming_threshold < 0 || hamming_threshold > address_dimension) {
        throw std::invalid_argument("Hamming threshold must be between zero and the address dimension.");
    }
    if (max_batch_size <= 0) {
        throw std::invalid_argument("Max batch size must be a positive integer.");
    }

    // Generate random address matrix on host.
    std::mt19937 rng(random_seed);
    std::uniform_int_distribution<int> dist(0, 1);

    std::vector<int> h_address_matrix(num_locations * address_dimension);
    for (int i = 0; i < num_locations * address_dimension; ++i) {
        h_address_matrix[i] = dist(rng);
    }

    // Allocate persistent device memory.
    check_cuda_error(
        cudaMalloc(&d_address_matrix_, num_locations * address_dimension * sizeof(int)),
        "Failed to allocate address matrix"
    );
    check_cuda_error(
        cudaMalloc(&d_memory_matrix_, num_locations * memory_dimension * sizeof(float)),
        "Failed to allocate memory matrix"
    );

    // Allocate batch buffers (pre-allocated for max batch size).
    check_cuda_error(
        cudaMalloc(&d_addresses_batch_, max_batch_size * address_dimension * sizeof(int)),
        "Failed to allocate batch address buffer"
    );
    check_cuda_error(
        cudaMalloc(&d_memories_batch_, max_batch_size * memory_dimension * sizeof(int)),
        "Failed to allocate batch memory buffer"
    );
    check_cuda_error(
        cudaMalloc(&d_activated_flags_batch_, max_batch_size * num_locations * sizeof(int)),
        "Failed to allocate batch activation flags"
    );
    check_cuda_error(
        cudaMalloc(&d_output_sums_batch_, max_batch_size * memory_dimension * sizeof(float)),
        "Failed to allocate batch output sum buffer"
    );
    check_cuda_error(
        cudaMalloc(&d_outputs_batch_, max_batch_size * memory_dimension * sizeof(int)),
        "Failed to allocate batch output buffer"
    );

    // Copy address matrix to device.
    check_cuda_error(
        cudaMemcpy(d_address_matrix_, h_address_matrix.data(),
                  num_locations * address_dimension * sizeof(int),
                  cudaMemcpyHostToDevice),
        "Failed to copy address matrix to device"
    );

    // Initialize memory matrix to zeros.
    check_cuda_error(
        cudaMemset(d_memory_matrix_, 0, num_locations * memory_dimension * sizeof(float)),
        "Failed to initialize memory matrix"
    );
}

inline KanervaSDMCUDA::~KanervaSDMCUDA() {
    cudaFree(d_address_matrix_);
    cudaFree(d_memory_matrix_);
    cudaFree(d_addresses_batch_);
    cudaFree(d_memories_batch_);
    cudaFree(d_activated_flags_batch_);
    cudaFree(d_output_sums_batch_);
    cudaFree(d_outputs_batch_);
}

inline void KanervaSDMCUDA::write(
    const std::vector<int>& address,
    const std::vector<int>& memory,
    bool sync_after
) {
    validate_vector(address, "address", address_dimension_);
    validate_vector(memory, "memory", memory_dimension_);

    write_batch_internal(address.data(), memory.data(), 1, sync_after);
    ++memory_count_;
}

inline void KanervaSDMCUDA::write_batch(
    const std::vector<std::vector<int>>& addresses,
    const std::vector<std::vector<int>>& memories,
    bool sync_after
) {
    if (addresses.size() != memories.size()) {
        throw std::invalid_argument("Address and memory batch sizes must match");
    }
    if (addresses.empty()) {
        throw std::invalid_argument("Batch cannot be empty");
    }
    if (static_cast<int>(addresses.size()) > max_batch_size_) {
        throw std::invalid_argument(
            "Batch size " + std::to_string(addresses.size()) +
            " exceeds maximum " + std::to_string(max_batch_size_)
        );
    }

    validate_batch(addresses, "addresses", address_dimension_);
    validate_batch(memories, "memories", memory_dimension_);

    int batch_size = static_cast<int>(addresses.size());

    // Flatten batches into contiguous host arrays.
    std::vector<int> h_addresses_flat(batch_size * address_dimension_);
    std::vector<int> h_memories_flat(batch_size * memory_dimension_);

    for (int i = 0; i < batch_size; ++i) {
        std::memcpy(&h_addresses_flat[i * address_dimension_],
                   addresses[i].data(),
                   address_dimension_ * sizeof(int));
        std::memcpy(&h_memories_flat[i * memory_dimension_],
                   memories[i].data(),
                   memory_dimension_ * sizeof(int));
    }

    write_batch_internal(h_addresses_flat.data(), h_memories_flat.data(),
                        batch_size, sync_after);
    memory_count_ += batch_size;
}

inline std::vector<int> KanervaSDMCUDA::read(
    const std::vector<int>& address,
    bool sync_before
) {
    validate_vector(address, "address", address_dimension_);

    std::vector<int> result(memory_dimension_);
    read_batch_internal(address.data(), result.data(), 1, sync_before);

    return result;
}

inline std::vector<std::vector<int>> KanervaSDMCUDA::read_batch(
    const std::vector<std::vector<int>>& addresses,
    bool sync_before
) {
    if (addresses.empty()) {
        throw std::invalid_argument("Batch cannot be empty");
    }
    if (static_cast<int>(addresses.size()) > max_batch_size_) {
        throw std::invalid_argument(
            "Batch size " + std::to_string(addresses.size()) +
            " exceeds maximum " + std::to_string(max_batch_size_)
        );
    }

    validate_batch(addresses, "addresses", address_dimension_);

    int batch_size = static_cast<int>(addresses.size());

    // Flatten addresses into contiguous host array.
    std::vector<int> h_addresses_flat(batch_size * address_dimension_);
    for (int i = 0; i < batch_size; ++i) {
        std::memcpy(&h_addresses_flat[i * address_dimension_],
                   addresses[i].data(),
                   address_dimension_ * sizeof(int));
    }

    // Allocate output buffer.
    std::vector<int> h_outputs_flat(batch_size * memory_dimension_);
    read_batch_internal(h_addresses_flat.data(), h_outputs_flat.data(),
                       batch_size, sync_before);

    // Unflatten results.
    std::vector<std::vector<int>> results(batch_size,
                                          std::vector<int>(memory_dimension_));
    for (int i = 0; i < batch_size; ++i) {
        std::memcpy(results[i].data(),
                   &h_outputs_flat[i * memory_dimension_],
                   memory_dimension_ * sizeof(int));
    }

    return results;
}

inline void KanervaSDMCUDA::synchronize() {
    check_cuda_error(cudaDeviceSynchronize(), "Failed to synchronize device");
}

inline void KanervaSDMCUDA::erase_memory(bool sync_after) {
    check_cuda_error(
        cudaMemset(d_memory_matrix_, 0, num_locations_ * memory_dimension_ * sizeof(float)),
        "Failed to erase memory matrix"
    );
    memory_count_ = 0;

    if (sync_after) {
        synchronize();
    }
}

inline void KanervaSDMCUDA::write_batch_internal(
    const int* h_addresses,
    const int* h_memories,
    int batch_size,
    bool sync_after
) {
    // Copy batch to device.
    check_cuda_error(
        cudaMemcpy(d_addresses_batch_, h_addresses,
                  batch_size * address_dimension_ * sizeof(int),
                  cudaMemcpyHostToDevice),
        "Failed to copy addresses to device"
    );
    check_cuda_error(
        cudaMemcpy(d_memories_batch_, h_memories,
                  batch_size * memory_dimension_ * sizeof(int),
                  cudaMemcpyHostToDevice),
        "Failed to copy memories to device"
    );

    // Compute activated locations.
    launch_compute_activated_locations_batch(
        d_address_matrix_,
        d_addresses_batch_,
        d_activated_flags_batch_,
        num_locations_,
        address_dimension_,
        hamming_threshold_,
        batch_size
    );
    check_cuda_error(cudaGetLastError(), "Kernel launch failed (activation batch)");

    // Write memory to activated locations.
    launch_write_memory_batch(
        d_memory_matrix_,
        d_memories_batch_,
        d_activated_flags_batch_,
        num_locations_,
        memory_dimension_,
        batch_size
    );
    check_cuda_error(cudaGetLastError(), "Kernel launch failed (write batch)");

    if (sync_after) {
        synchronize();
    }
}

inline void KanervaSDMCUDA::read_batch_internal(
    const int* h_addresses,
    int* h_outputs,
    int batch_size,
    bool sync_before
) {
    if (sync_before) {
        synchronize();
    }

    // Copy addresses to device.
    check_cuda_error(
        cudaMemcpy(d_addresses_batch_, h_addresses,
                  batch_size * address_dimension_ * sizeof(int),
                  cudaMemcpyHostToDevice),
        "Failed to copy addresses to device"
    );

    // Compute activated locations.
    launch_compute_activated_locations_batch(
        d_address_matrix_,
        d_addresses_batch_,
        d_activated_flags_batch_,
        num_locations_,
        address_dimension_,
        hamming_threshold_,
        batch_size
    );
    check_cuda_error(cudaGetLastError(), "Kernel launch failed (activation batch)");

    // Read memory from activated locations.
    launch_read_memory_batch(
        d_memory_matrix_,
        d_activated_flags_batch_,
        d_output_sums_batch_,
        num_locations_,
        memory_dimension_,
        batch_size
    );
    check_cuda_error(cudaGetLastError(), "Kernel launch failed (read batch)");

    // Threshold output.
    launch_threshold_output_batch(
        d_output_sums_batch_,
        d_outputs_batch_,
        memory_dimension_,
        batch_size
    );
    check_cuda_error(cudaGetLastError(), "Kernel launch failed (threshold batch)");

    // Copy results back to host.
    check_cuda_error(
        cudaMemcpy(h_outputs, d_outputs_batch_,
                  batch_size * memory_dimension_ * sizeof(int),
                  cudaMemcpyDeviceToHost),
        "Failed to copy results from device"
    );
}

inline void KanervaSDMCUDA::validate_vector(
    const std::vector<int>& vector,
    const std::string& vector_name,
    int expected_dimension
) {
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

inline void KanervaSDMCUDA::validate_batch(
    const std::vector<std::vector<int>>& batch,
    const std::string& batch_name,
    int expected_dimension
) {
    for (size_t i = 0; i < batch.size(); ++i) {
        validate_vector(batch[i],
                       batch_name + "[" + std::to_string(i) + "]",
                       expected_dimension);
    }
}

inline void KanervaSDMCUDA::check_cuda_error(cudaError_t error, const std::string& message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(message + ": " + cudaGetErrorString(error));
    }
}

#endif // KANERVA_SDM_CUH
