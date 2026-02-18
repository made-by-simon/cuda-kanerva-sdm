"""Generate the performance plot from a saved combined_results.csv."""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

INPUT_CSV  = "combined_results.csv"
OUTPUT_PNG = "combined_results.png"

if not os.path.exists(INPUT_CSV):
    print(f"[ERROR] {INPUT_CSV} not found. Run all_four_combined.py first.")
    sys.exit(1)

df = pd.read_csv(INPUT_CSV)

dimension    = df["dimension"].iloc[0]
num_memories = df["num_memories"].iloc[0]

for _path in font_manager.findSystemFonts():
    if "Inter" in _path:
        font_manager.fontManager.addfont(_path)

sns.set_theme(style="whitegrid", font="Inter")
_, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(
    data      = df,
    x         = "num_hard_locations",
    y         = "time_s",
    hue       = "implementation",
    hue_order = ["Python", "C++", "Sequential CUDA", "Parallel CUDA"],
    marker    = "o",
    ax        = ax,
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of Hard Locations")
ax.set_ylabel("Time (s)")
ax.set_title(f"Performance of Different KanervaSDM Implementations (Dimension = 100 and Memory Count = 100)")
ax.legend()

implementations = ["Python", "C++", "Sequential CUDA", "Parallel CUDA"]
colors = {line.get_label(): line.get_color() for line in ax.get_lines()}

for impl in implementations:
    row = df[df["implementation"] == impl].sort_values("num_hard_locations").iloc[-1]
    ax.annotate(
        f"{row['time_s']:.3f}s",
        xy=(row["num_hard_locations"], row["time_s"]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=12,
        color=colors.get(impl, "black"),
    )
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
plt.show()
print(f"Plot saved to {OUTPUT_PNG}")
