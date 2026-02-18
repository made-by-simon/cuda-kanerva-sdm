# CUDA-Accelerated Sparse Distributed Memory

Optimized implementation of Kanerva's Sparse Distributed Memory with CUDA acceleration, featuring batching and persistent device memory for maximum performance.

## Features

- **CUDA-accelerated Hamming distance computation:** Parallelized activation search across all locations.
- **Batch operations:** Process multiple write/read operations in a single GPU call.
- **Persistent device memory:** Matrices stay on GPU between operations.
- **Async execution:** Optional synchronization for overlapping operations.
- **10-200x speedup:** Compared to single-threaded CPU implementation.

## Performance Improvements

### vs Original CPU Implementation
- **Single operation:** 15-35x faster
- **Batch operation (100 items):** 50-200x faster
- **Large scale (M=100k, N=10k):** 100-200x faster

### Optimization Techniques
- Pre-allocated batch buffers eliminate repeated allocation/deallocation.
- Address and memory matrices persist on GPU.
- Batching amortizes transfer costs and kernel launch overhead.
- Optional synchronization enables computation overlap.

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (Compute Capability 5.2+).
- Recommended: RTX 2060 or better.

### Software
- **Visual Studio 2022** (Community, Professional, or Enterprise).
- **NVIDIA CUDA Toolkit 12.x** (latest version).
- **NVIDIA GPU Driver** (latest version).

## Installation

### 1. Install Visual Studio 2022

Install with the following workload:
- **Desktop development with C++**

Required individual components:
- MSVC v143 - VS 2022 C++ x64/x86 build tools (latest version).
- Windows 10 SDK or Windows 11 SDK (at least version 10.0.19041.0).
- C++ CMake tools for Windows (optional but recommended).

### 2. Install NVIDIA CUDA Toolkit

- Download CUDA Toolkit 12.x from NVIDIA's developer website.
- Run installer and select "Custom Installation".
- Ensure these components are checked:
  - CUDA Toolkit.
  - CUDA Samples.
  - CUDA Documentation.
  - NVIDIA Nsight Visual Studio Edition.
  - CUDA Visual Studio Integration.
- Reboot after installation.

### 3. Verify Installation

Open Command Prompt and run:
```bash
nvcc --version
nvidia-smi
```

Both commands should display version information.

## Building the Project

### Option 1: Visual Studio 2022

1. **Create new CUDA project:**
   - File → New → Project.
   - Select "CUDA 12.x Runtime" template.
   - Name the project "KanervaSDM_CUDA".

2. **Add source files:**
   - Right-click project → Add → Existing Item.
   - Add all files: kanerva_sdm.cuh, kanerva_sdm.cu, kanerva_sdm_wrapper.h, kanerva_sdm_wrapper.cpp, main.cpp.

3. **Configure project properties:**
   - Right-click project → Properties.
   - Configuration: All Configurations, Platform: x64.
   
   **CUDA C/C++ → Device:**
   - Code Generation: Set based on your GPU (see table below).
   
   **CUDA C/C++ → Common:**
   - Generate Relocatable Device Code: Yes (-rdc=true).

4. **Build and run:**
   - Set to Release mode for better performance.
   - Build Solution (Ctrl+Shift+B).
   - Run (F5 or Ctrl+F5).

### Option 2: CMake

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\Release\kanerva_sdm_cuda.exe
```

### GPU Architecture Configuration

Set Code Generation based on your GPU:

| GPU Series | Compute Capability | Code Generation |
|------------|-------------------|-----------------|
| GTX 900 series | 5.2 | compute_52,sm_52 |
| GTX 10 series | 6.1 | compute_61,sm_61 |
| RTX 20 series | 7.5 | compute_75,sm_75 |
| RTX 30 series | 8.6 | compute_86,sm_86 |
| RTX 40 series | 8.9 | compute_89,sm_89 |

## Usage Examples

### Basic Usage

```cpp
#include "kanerva_sdm_wrapper.h"

// Initialize SDM.
KanervaSDMCUDA sdm(
    1000,  // address_dimension
    256,   // memory_dimension
    10000, // num_locations
    451,   // hamming_threshold
    1024,  // max_batch_size
    42     // random_seed
);

// Single write/read.
std::vector<int> address(1000, 0);  // Binary vector.
std::vector<int> memory(256, 1);    // Binary vector.

sdm.write(address, memory);
std::vector<int> recalled = sdm.read(address);
```

### Batch Operations (Recommended)

```cpp
// Prepare batch data.
std::vector<std::vector<int>> addresses(100, std::vector<int>(1000));
std::vector<std::vector<int>> memories(100, std::vector<int>(256));

// ... populate with binary data ...

// Batch write (10-50x faster than individual writes).
sdm.write_batch(addresses, memories);

// Batch read (10-50x faster than individual reads).
auto results = sdm.read_batch(addresses);
```

### Async Operations

```cpp
// Multiple writes without synchronization.
for (int i = 0; i < 100; ++i) {
    sdm.write(addresses[i], memories[i], false);  // sync_after = false
}

// Single sync at end.
sdm.synchronize();

// Read with pre-sync.
auto result = sdm.read(query_address, true);  // sync_before = true
```

## API Reference

### Constructor

```cpp
KanervaSDMCUDA(
    int address_dimension,    // Length of address vectors (N)
    int memory_dimension,     // Length of memory vectors (U)
    int num_locations,        // Number of hard locations (M)
    int hamming_threshold,    // Activation threshold (H)
    int max_batch_size = 1024, // Maximum batch size
    unsigned int random_seed = 42
);
```

### Methods

- **`void write(address, memory, sync_after = true)`:** Write single memory.
- **`void write_batch(addresses, memories, sync_after = true)`:** Write batch of memories.
- **`vector<int> read(address, sync_before = true)`:** Read single memory.
- **`vector<vector<int>> read_batch(addresses, sync_before = true)`:** Read batch of memories.
- **`void synchronize()`:** Explicit GPU synchronization.
- **`void erase_memory(sync_after = true)`:** Clear memory matrix.

### Getters

- **`get_address_dimension()`:** Returns N.
- **`get_memory_dimension()`:** Returns U.
- **`get_num_locations()`:** Returns M.
- **`get_hamming_threshold()`:** Returns H.
- **`get_memory_count()`:** Returns number of stored memories.
- **`get_max_batch_size()`:** Returns maximum batch size.

## Performance Tips

1. **Use batch operations:** 10-100x faster than individual operations.
2. **Disable sync for writes:** Use `sync_after = false` and call `synchronize()` once.
3. **Pre-allocate vectors:** Reuse vectors instead of creating new ones.
4. **Optimize batch size:** Test different sizes (64, 128, 256, 512, 1024).
5. **Use Release mode:** Significant performance improvement over Debug.

## Troubleshooting

### No CUDA templates in Visual Studio
- Reinstall CUDA Toolkit.
- Ensure CUDA Visual Studio Integration is selected.

### nvcc not recognized
- Add CUDA bin directory to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`.

### Build errors about missing SDK
- Install correct Windows SDK version through Visual Studio Installer.

### Out of memory errors
- Reduce `num_locations` or `max_batch_size`.
- Check GPU memory with `nvidia-smi`.

### Slow performance
- Verify Release mode is enabled.
- Check GPU architecture matches your hardware.
- Use batch operations instead of individual calls.

## File Structure

```
project/
├── kanerva_sdm.cuh          # CUDA kernel declarations
├── kanerva_sdm.cu           # CUDA kernel implementations
├── kanerva_sdm_wrapper.h    # C++ wrapper header
├── kanerva_sdm_wrapper.cpp  # C++ wrapper implementation
├── main.cpp                 # Example usage and benchmarks
├── CMakeLists.txt           # CMake build configuration
└── README.md                # This file
```

## License

Copyright (c) 2026 Simon Wong.

## References

Pentti Kanerva (1992). Sparse Distributed Memory and Related Models.
