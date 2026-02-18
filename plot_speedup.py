"""Plot Python / Parallel CUDA speedup ratio from combined_results.csv."""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

INPUT_CSV  = "combined_results.csv"
OUTPUT_PNG = "speedup.png"

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

py   = df[df["implementation"] == "Python"].sort_values("num_hard_locations").reset_index(drop=True)
cuda = df[df["implementation"] == "Parallel CUDA"].sort_values("num_hard_locations").reset_index(drop=True)

speedup = pd.DataFrame({
    "num_hard_locations": py["num_hard_locations"],
    "speedup": py["time_s"].values / cuda["time_s"].values,
})

_, ax = plt.subplots(figsize=(10, 6))

color = sns.color_palette()[0]
ax.plot(speedup["num_hard_locations"], speedup["speedup"], marker="o", color=color)

# Label each point with its speedup value
for _, row in speedup.iterrows():
    ax.annotate(
        f"{row['speedup']:.1f}×",
        xy=(row["num_hard_locations"], row["speedup"]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=12,
        color=color,
    )

ax.set_xscale("log")
ax.set_xlabel("Number of Hard Locations")
ax.set_ylabel("Speedup (×)")
ax.set_title(
    "Speedup Ratio of Python and CUDA KanervaSDM Implementations (Dimension = 100 and Memory Count = 100)"
)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
plt.show()
print(f"Plot saved to {OUTPUT_PNG}")
