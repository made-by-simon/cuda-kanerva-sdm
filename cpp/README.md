# Kanerva Sparse Distributed Memory — Python & C++ Implementations

Python and C++ (with pybind11 bindings) implementations of Pentti Kanerva's Sparse Distributed Memory (SDM).

## Prerequisites

- **Python 3.8+**
- **C++ compiler** with C++17 support
  - **Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (select "Desktop development with C++" workload)
  - **macOS:** `xcode-select --install`
  - **Linux:** `sudo apt install build-essential` (Debian/Ubuntu) or equivalent

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Windows (cmd)
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies and build the C++ extension

```bash
pip install -e ".[dev]"
```

This single command:
- Installs **pybind11** (needed to compile the C++ bindings)
- Compiles `src/bindings.cpp` and `src/kanerva_sdm.h` into a native Python module (`sdm_cpp`)
- Installs **numpy** and **pytest** (dev dependencies)

> **Tip:** If you only want to run the pure-Python implementation, `pip install numpy` is sufficient — the C++ extension is optional.

### 3. Verify the build

```bash
python -c "import sdm_cpp; print('C++ module loaded OK')"
```

## Running the tests

```bash
python test_sdm.py
```

This runs both implementations through the same test suite with shared parameters (address dimension, memory dimension, number of hard locations, Hamming threshold, etc.) and prints a side-by-side comparison of accuracy and performance.

If the C++ module is not installed, the script will skip the C++ tests and run only the Python ones.

## Project structure

```
cpp/
├── kanerva_sdm.py       # Pure-Python SDM implementation
├── src/
│   ├── kanerva_sdm.h    # C++ SDM implementation (header-only)
│   └── bindings.cpp     # pybind11 bindings exposing C++ class to Python
├── test_sdm.py          # Consolidated test script (Python + C++)
├── setup.py             # Build configuration for the C++ extension
├── pyproject.toml       # Project metadata and dependencies
└── README.md
```

## Rebuilding after C++ changes

If you edit `src/kanerva_sdm.h` or `src/bindings.cpp`, rebuild with:

```bash
pip install -e ".[dev]"
```
