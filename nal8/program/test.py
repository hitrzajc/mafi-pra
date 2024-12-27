import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib import cm

# Create some data and a colormap
data = np.random.rand(10, 10)
cmap = cm.viridis

# Create a figure and axis
fig, ax = plt.subplots()

# Display the image
cax = ax.imshow(data, cmap=cmap)

# Add the color bar
cbar = fig.colorbar(cax)

# Create a dashed line to overlay on the color bar
line = mlines.Line2D([0.1, 0.9], [0.2, 0.2], color='black', linestyle='--', linewidth=0.5, transform=cbar.ax.transAxes)
cbar.ax.add_line(line)

# Show the plot
plt.show()
