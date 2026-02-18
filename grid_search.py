"""
Grid search over dimension x num_locations for KanervaSDM CUDA performance.

Runs all 9 combinations of:
  dimension      : [10, 100, 1000]
  num_locations  : [100, 10_000, 1_000_000]
  memory_count   : 1000 (fixed)

Results saved to grid_search_results_cuda.csv.
"""

import subprocess
import csv
import re
import itertools
import os
import sys

EXE_PATH = os.path.join("build", "Debug", "kanerva_sdm_cuda.exe")
OUTPUT_CSV = "grid_search_cuda.csv"

DIMENSIONS = [10, 100, 1000]
NUM_LOCATIONS_LIST = [100, 10_000, 1_000_000]
MEMORY_COUNT = 1000
SEED = 37


def hamming_threshold_for(dimension):
    """
    Scale the Hamming threshold proportionally from the default of 37 @ dim=1000.
    Must satisfy 0 <= threshold <= dimension.
    """
    return max(1, round(37 * dimension / 1000))


def max_batch_for(num_locations):
    """
    Limit batch size so the activated-flags buffer (max_batch * num_locations * 4 bytes)
    stays under ~32 MB on the GPU.  Also capped at 1024.
    """
    target_bytes = 32 * 1024 * 1024  # 32 MB
    return min(1024, max(1, target_bytes // (num_locations * 4)))


def run_sdm(dimension, num_locations, memory_count):
    """
    Launch the CUDA exe with the given parameters via stdin and parse timings.
    Returns (single_op_time, batched_time, status_str).
    """
    hamming = hamming_threshold_for(dimension)
    max_batch = max_batch_for(num_locations)

    # Each line answers one interactive prompt in order:
    #   Dimension / Num locations / Hamming threshold / Num memories / Max batch size / Seed
    stdin_input = "\n".join([
        str(dimension),
        str(num_locations),
        str(hamming),
        str(memory_count),
        str(max_batch),
        str(SEED),
        "",  # trailing newline so the last getline() completes
    ])

    print(f"\n{'='*64}")
    print(f"  dim={dimension:>7}, locs={num_locations:>9}, "
          f"hamming={hamming}, max_batch={max_batch}")
    print(f"{'='*64}")
    sys.stdout.flush()

    try:
        result = subprocess.run(
            [EXE_PATH],
            input=stdin_input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,  # 10-minute timeout per combination
        )

        stdout = result.stdout
        stderr = result.stderr

        # Echo stdout (strip carriage-return noise from progress bars).
        for line in stdout.replace("\r", "\n").splitlines():
            stripped = line.strip()
            if stripped:
                print(" ", stripped)
        sys.stdout.flush()

        if result.returncode != 0:
            error_msg = f"exit_code_{result.returncode}"
            print(f"  [ERROR] Process exited with code {result.returncode}")
            if stderr:
                print(f"  [STDERR] {stderr[:500]}")
            return None, None, error_msg

        # Parse from the Performance Summary block.
        single_match = re.search(r"Single-op time:\s*([\d.]+)\s*s", stdout)
        batched_match = re.search(r"Batched time:\s*([\d.]+)\s*s", stdout)

        single_time = float(single_match.group(1)) if single_match else None
        batched_time = float(batched_match.group(1)) if batched_match else None

        if single_time is None or batched_time is None:
            return single_time, batched_time, "parse_error"

        return single_time, batched_time, "ok"

    except subprocess.TimeoutExpired:
        print("  [ERROR] Timed out after 600 s")
        return None, None, "timeout"
    except Exception as exc:
        print(f"  [ERROR] {exc}")
        return None, None, str(exc)[:80]


def main():
    if not os.path.exists(EXE_PATH):
        print(f"[ERROR] Executable not found: {EXE_PATH}")
        print("Build the project first:  cmake --build build --config Debug")
        sys.exit(1)

    combos = list(itertools.product(DIMENSIONS, NUM_LOCATIONS_LIST))

    print(f"Grid search: {len(combos)} combinations")
    print(f"  memory_count = {MEMORY_COUNT} (fixed)")
    print(f"  Executable   : {EXE_PATH}")
    print(f"  Output CSV   : {OUTPUT_CSV}")

    results = []

    for i, (dimension, num_locations) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}]  dimension={dimension}, num_locations={num_locations}")

        single_time, batched_time, status = run_sdm(dimension, num_locations, MEMORY_COUNT)

        results.append({
            "dimension": dimension,
            "num_locations": num_locations,
            "memory_count": MEMORY_COUNT,
            "hamming_threshold": hamming_threshold_for(dimension),
            "max_batch": max_batch_for(num_locations),
            "single_op_time_s": single_time,
            "batched_time_s": batched_time,
        })

        print(f"  -> single={single_time}s, batched={batched_time}s  [{status}]")

    # Write CSV.
    fieldnames = [
        "dimension", "num_locations", "memory_count", "hamming_threshold",
        "max_batch", "single_op_time_s", "batched_time_s",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*64}")
    print(f"Results saved to {OUTPUT_CSV}")
    print(f"{'='*64}")

    # Summary table.
    print(f"\n{'dim':>6} {'num_locs':>10} {'single_s':>10} {'batched_s':>10}")
    print("-" * 40)
    for r in results:
        s = f"{r['single_op_time_s']:.3f}" if r["single_op_time_s"] is not None else "N/A"
        b = f"{r['batched_time_s']:.3f}"  if r["batched_time_s"]  is not None else "N/A"
        print(f"{r['dimension']:>6} {r['num_locations']:>10} {s:>10} {b:>10}")


if __name__ == "__main__":
    main()
