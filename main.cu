#include "kanerva_sdm.cuh"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <string>
#include <sstream>
#include <algorithm>

class ProgressBar {
public:
    ProgressBar(const std::string& desc, long long total, int width = 30)
        : desc_(desc), total_(total), width_(width), current_(0), check_counter_(0) {
        check_every_ = std::max(1LL, total_ / 1000);
        start_ = std::chrono::high_resolution_clock::now();
        last_display_ = start_;
        render(start_);
    }

    void update(long long n = 1) {
        current_ += n;
        check_counter_ += n;
        if (check_counter_ >= check_every_ || current_ >= total_) {
            check_counter_ = 0;
            auto now = std::chrono::high_resolution_clock::now();
            double since_last = std::chrono::duration<double>(now - last_display_).count();
            if (since_last >= 0.05 || current_ >= total_) {
                render(now);
                last_display_ = now;
            }
        }
    }

    void finish() {
        current_ = total_;
        render(std::chrono::high_resolution_clock::now());
        std::cout << "\n";
    }

private:
    void render(std::chrono::high_resolution_clock::time_point now) {
        double elapsed = std::chrono::duration<double>(now - start_).count();
        double pct     = (total_ > 0) ? (double)current_ / total_ : 0.0;
        int filled     = (int)(pct * width_);
        double rate    = (elapsed > 0 && current_ > 0) ? current_ / elapsed : 0.0;
        double eta     = (rate > 0 && current_ < total_) ? (total_ - current_) / rate : 0.0;

        int e_m = (int)elapsed / 60, e_s = (int)elapsed % 60;
        int t_m = (int)eta    / 60, t_s = (int)eta    % 60;

        std::string bar(filled, '=');
        if (current_ < total_ && filled < width_) bar += '>';
        bar += std::string(std::max(0, width_ - (int)bar.size()), ' ');

        // Format rate with K/M suffix for readability.
        std::string rate_str;
        {
            char buf[32];
            if (rate >= 1e6)       snprintf(buf, sizeof(buf), "%.2fM", rate / 1e6);
            else if (rate >= 1e3)  snprintf(buf, sizeof(buf), "%.2fK", rate / 1e3);
            else                   snprintf(buf, sizeof(buf), "%.1f",  rate);
            rate_str = buf;
        }

        char time_buf[64];
        snprintf(time_buf, sizeof(time_buf), "%02d:%02d<%02d:%02d", e_m, e_s, t_m, t_s);

        std::cout << "\r" << desc_ << ": " << std::setw(3) << (int)(pct * 100) << "%|"
                  << bar << "| "
                  << current_ << "/" << total_
                  << " [" << time_buf << ", " << rate_str << " ops/s]  "
                  << std::flush;
    }

    std::string  desc_;
    long long    total_, current_, check_counter_, check_every_;
    int          width_;
    std::chrono::high_resolution_clock::time_point start_, last_display_;
};

std::vector<int> generate_random_binary_vector(int size, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 1);
    std::vector<int> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

/**
 * Single-operation speed test (mirrors the Python/C++ notebook loop).
 * Writes one-by-one, then reads one-by-one, with sync after each op.
 */
double speed_test_single(KanervaSDMCUDA& sdm, int dimension, int num_memories) {
    std::cout << "\n=== CUDA Single-Op Speed Test (" << num_memories << " memories) ===" << std::endl;

    std::mt19937 rng(42);
    std::vector<std::vector<int>> addresses(num_memories);
    std::vector<std::vector<int>> memories(num_memories);

    for (int i = 0; i < num_memories; ++i) {
        addresses[i] = generate_random_binary_vector(dimension, rng);
        memories[i] = generate_random_binary_vector(dimension, rng);
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Write one-by-one (synced), matching the Python loop.
    {
        ProgressBar pb("Writing", num_memories);
        for (int i = 0; i < num_memories; ++i) {
            sdm.write(addresses[i], memories[i], true);
            pb.update();
        }
        pb.finish();
    }

    // Read one-by-one (synced), matching the Python loop.
    {
        ProgressBar pb("Reading", num_memories);
        for (int i = 0; i < num_memories; ++i) {
            sdm.read(addresses[i], true);
            pb.update();
        }
        pb.finish();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Stored and recalled " << num_memories
              << " memories. Elapsed time: " << std::fixed << std::setprecision(3)
              << elapsed << " seconds." << std::endl;

    return elapsed;
}

/**
 * Batched speed test — processes all writes then all reads in large batches.
 * This exploits CUDA parallelism and is the recommended usage pattern.
 */
double speed_test_batched(KanervaSDMCUDA& sdm, int dimension, int num_memories) {
    int max_batch = sdm.get_max_batch_size();

    std::cout << "\n=== CUDA Batched Speed Test (" << num_memories
              << " memories, batch_size=" << max_batch << ") ===" << std::endl;

    std::mt19937 rng(42);
    std::vector<std::vector<int>> addresses(num_memories);
    std::vector<std::vector<int>> memories(num_memories);

    for (int i = 0; i < num_memories; ++i) {
        addresses[i] = generate_random_binary_vector(dimension, rng);
        memories[i] = generate_random_binary_vector(dimension, rng);
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Batched writes.
    {
        ProgressBar pb("Writing", num_memories);
        for (int offset = 0; offset < num_memories; offset += max_batch) {
            int batch_size = std::min(max_batch, num_memories - offset);
            std::vector<std::vector<int>> addr_batch(addresses.begin() + offset,
                                                      addresses.begin() + offset + batch_size);
            std::vector<std::vector<int>> mem_batch(memories.begin() + offset,
                                                     memories.begin() + offset + batch_size);
            sdm.write_batch(addr_batch, mem_batch, true);
            pb.update(batch_size);
        }
        pb.finish();
    }

    // Batched reads.
    {
        ProgressBar pb("Reading", num_memories);
        for (int offset = 0; offset < num_memories; offset += max_batch) {
            int batch_size = std::min(max_batch, num_memories - offset);
            std::vector<std::vector<int>> addr_batch(addresses.begin() + offset,
                                                      addresses.begin() + offset + batch_size);
            sdm.read_batch(addr_batch, true);
            pb.update(batch_size);
        }
        pb.finish();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    std::cout << "Stored and recalled " << num_memories
              << " memories. Elapsed time: " << std::fixed << std::setprecision(3)
              << elapsed << " seconds." << std::endl;

    return elapsed;
}

template<typename T>
T prompt(const std::string& label, T default_val) {
    std::cout << "  " << label << " [" << default_val << "]: ";
    std::string line;
    std::getline(std::cin, line);
    if (line.empty()) return default_val;
    T val;
    std::istringstream(line) >> val;
    return val;
}

int main() {
    try {
        std::cout << "Enter parameters (press Enter to accept default):" << std::endl;
        const int dimension         = prompt("Dimension",         1000);
        const int num_locations     = prompt("Num locations",     1000000);
        const int hamming_threshold = prompt("Hamming threshold", 37);
        const int num_memories      = prompt("Num memories",      1000);
        const int max_batch         = prompt("Max batch size",    1024);
        const unsigned int seed     = prompt("Seed",              37u);

        std::cout << "\nInitializing CUDA SDM..." << std::endl;
        std::cout << "  Dimension:          " << dimension << std::endl;
        std::cout << "  Num locations:      " << num_locations << std::endl;
        std::cout << "  Hamming threshold:  " << hamming_threshold << std::endl;
        std::cout << "  Num memories:       " << num_memories << std::endl;
        std::cout << "  Max batch size:     " << max_batch << std::endl;

        KanervaSDMCUDA sdm(dimension, dimension, num_locations,
                           hamming_threshold, max_batch, seed);

        // Test 1: Single-operation loop (apples-to-apples with Python/C++).
        sdm.erase_memory();
        double time_single = speed_test_single(sdm, dimension, num_memories);

        // Test 2: Batched operations (showcases CUDA parallelism).
        sdm.erase_memory();
        double time_batched = speed_test_batched(sdm, dimension, num_memories);

        // Summary.
        std::cout << "\n=== Performance Summary ===" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Single-op time:  " << time_single  << " s" << std::endl;
        std::cout << "  Batched time:    " << time_batched  << " s" << std::endl;
        std::cout << "  Batch speedup:   " << std::setprecision(2)
                  << time_single / time_batched << "x over single-op" << std::endl;

        std::cout << "\nPaste these into the notebook to compare:" << std::endl;
        std::cout << "  time_cuda_single  = " << std::setprecision(3) << time_single << std::endl;
        std::cout << "  time_cuda_batched = " << time_batched << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
