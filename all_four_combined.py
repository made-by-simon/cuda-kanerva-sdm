"""
Unified benchmark: Python, C++, Sequential CUDA, Batched CUDA.

Fixed : dimension=100, num_memories=100
Varies: num_hard_locations = [10, 100, 1000, 10000, 100000]

Outputs: combined_results.csv + combined_results.png
"""

import subprocess
import csv
import re
import time
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import kanerva_sdm as sdm_py
import sdm_cpp

# ── fixed parameters ──────────────────────────────────────────────────────────
DIMENSION             = 100
NUM_MEMORIES          = 100
SEED                  = 42
HAMMING_THRESHOLD     = int(0.5*DIMENSION)
NUM_HARD_LOCATIONS_LIST = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

EXE_PATH   = os.path.join("build", "Debug", "kanerva_sdm_cuda.exe")
OUTPUT_CSV = "combined_results.csv"
OUTPUT_PNG = "combined_results.png"


# ── helpers ───────────────────────────────────────────────────────────────────

def _max_batch(num_locations: int) -> int:
    """Keep activated-flags buffer under ~32 MB; cap at 1024."""
    return min(1024, max(1, (32 * 1024 * 1024) // (num_locations * 4)))


def run_python(num_hard_locations: int, addresses, memories) -> float:
    sdm = sdm_py.KanervaSDM(
        ADDRESS_DIMENSION=DIMENSION,
        MEMORY_DIMENSION=DIMENSION,
        NUM_LOCATIONS=num_hard_locations,
        HAMMING_THRESHOLD=HAMMING_THRESHOLD,
        RANDOM_SEED=SEED,
    )
    start = time.perf_counter()
    for i in range(NUM_MEMORIES):
        sdm.write(addresses[i], memories[i])
    for i in range(NUM_MEMORIES):
        sdm.read(addresses[i])
    return time.perf_counter() - start


def run_cpp(num_hard_locations: int, addresses, memories) -> float:
    sdm = sdm_cpp.KanervaSDM(
        address_dimension=DIMENSION,
        memory_dimension=DIMENSION,
        num_locations=num_hard_locations,
        activation_threshold=HAMMING_THRESHOLD,
    )
    start = time.perf_counter()
    for i in range(NUM_MEMORIES):
        sdm.write(addresses[i], memories[i])
    for i in range(NUM_MEMORIES):
        sdm.read(addresses[i])
    return time.perf_counter() - start


def run_cuda(num_hard_locations: int):
    """
    Invoke the CUDA exe via stdin and return (sequential_time, batched_time).
    Stdin order: dimension / num_locations / hamming / num_memories / max_batch / seed
    """
    max_batch   = _max_batch(num_hard_locations)
    stdin_input = "\n".join([
        str(DIMENSION),
        str(num_hard_locations),
        str(HAMMING_THRESHOLD),
        str(NUM_MEMORIES),
        str(max_batch),
        str(SEED),
        "",   # trailing newline so the last getline() completes
    ])

    result = subprocess.run(
        [EXE_PATH],
        input=stdin_input,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"CUDA exe exited {result.returncode}: {result.stderr[:300]}"
        )

    single_match = re.search(r"Single-op time:\s*([\d.]+)\s*s", result.stdout)
    batched_match = re.search(r"Batched time:\s*([\d.]+)\s*s",  result.stdout)

    sequential = float(single_match.group(1)) if single_match else None
    batched    = float(batched_match.group(1)) if batched_match else None

    if sequential is None or batched is None:
        raise RuntimeError(
            f"Could not parse CUDA timings from output:\n{result.stdout[-500:]}"
        )

    return sequential, batched


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(EXE_PATH):
        print(f"[ERROR] CUDA executable not found: {EXE_PATH}")
        print("Build first:  cmake --build build --config Debug")
        sys.exit(1)

    rng       = np.random.default_rng(SEED)
    addresses = rng.integers(0, 2, (NUM_MEMORIES, DIMENSION), dtype=np.int8)
    memories  = rng.integers(0, 2, (NUM_MEMORIES, DIMENSION), dtype=np.int8)

    results = []

    for num_locs in tqdm(NUM_HARD_LOCATIONS_LIST, desc="num_hard_locations"):
        tqdm.write(f"\nnum_hard_locations = {num_locs:,}")

        py_time = run_python(num_locs, addresses, memories)
        tqdm.write(f"  Python:          {py_time:.4f} s")

        cpp_time = run_cpp(num_locs, addresses, memories)
        tqdm.write(f"  C++:             {cpp_time:.4f} s")

        seq_time, par_time = run_cuda(num_locs)
        tqdm.write(f"  Sequential CUDA: {seq_time:.4f} s")
        tqdm.write(f"  Parallel CUDA:   {par_time:.4f} s")

        base = {"num_memories": NUM_MEMORIES, "dimension": DIMENSION,
                "num_hard_locations": num_locs}
        results += [
            {**base, "implementation": "Python",          "time_s": py_time},
            {**base, "implementation": "C++",             "time_s": cpp_time},
            {**base, "implementation": "Sequential CUDA", "time_s": seq_time},
            {**base, "implementation": "Parallel CUDA",   "time_s": par_time},
        ]

    # ── CSV ───────────────────────────────────────────────────────────────────
    fieldnames = ["num_memories", "dimension", "num_hard_locations",
                  "implementation", "time_s"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    implementations = ["Python", "C++", "Sequential CUDA", "Parallel CUDA"]
    _, ax = plt.subplots(figsize=(10, 6))

    for impl in implementations:
        rows = [r for r in results if r["implementation"] == impl]
        xs   = [r["num_hard_locations"] for r in rows]
        ys   = [r["time_s"]             for r in rows]
        ax.plot(xs, ys, marker="o", label=impl)

    ax.set_xscale("log")
    #ax.set_yscale("log")
    ax.set_xlabel("Number of Hard Locations")
    ax.set_ylabel("Time (s)")
    ax.set_title(
        f"KanervaSDM Performance  —  dim={DIMENSION}, memories={NUM_MEMORIES}"
    )
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.show()
    print(f"Plot saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
