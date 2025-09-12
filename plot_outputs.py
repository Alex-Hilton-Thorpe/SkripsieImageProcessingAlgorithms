# plot_outputs.py

import matplotlib.pyplot as plt

# Path to your output file
filepath = "outputs.txt"

# Lists to hold x and y values
x_vals = []
y_vals = []

# Read the file
with open(filepath, "r") as f:
    for line in f:
        line = line.strip()
        if not line:  # skip empty lines
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue  # skip malformed lines
        try:
            x = float(parts[0])
            y = 1-float(parts[1])
        except ValueError:
            continue  # skip lines that can't be converted
        x_vals.append(x)
        y_vals.append(y)

# Create the plot
plt.figure(figsize=(8, 6))
if x_vals and y_vals:
    # Plot first point as red
    plt.scatter(x_vals[0], y_vals[0], c='red', marker='o', label='First Point')

    # Plot the rest as blue
    if len(x_vals) > 1:
        plt.scatter(x_vals[1:], y_vals[1:], c='blue', marker='o', label='Other Points')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Ball Tracking Output")
plt.grid(True)
plt.axis('equal')  # optional: equal scaling for x and y
plt.show()
