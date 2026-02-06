import os
import re
from PIL import Image

# Folder containing your PNGs
folder = "gif_plots"

# Regex to capture the scientific-notation number in the filename
pattern = re.compile(r"final_plot_y_eff_(.*)\.png")

# Collect (value, path) pairs
files = []
for f in os.listdir(folder):
    match = pattern.match(f)
    if match:
        val = float(match.group(1))   # e.g. "1.00e+19" → 1e19
        files.append((val, os.path.join(folder, f)))

# Sort by the numeric value
files.sort(key=lambda x: x[0])

# Load images in order
images = [Image.open(path) for _, path in files]

# Save GIF (looping forever, 200 ms per frame)
gif_path = "final.gif"
images[0].save(
    gif_path,
    save_all=True,
    append_images=images[1:],
    duration=200,     # ms per frame; adjust as needed
    loop=0
)

print(f"Saved GIF as: {gif_path}")
