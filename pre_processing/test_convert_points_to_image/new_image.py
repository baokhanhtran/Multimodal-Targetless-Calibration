import cv2
import numpy as np

# Read coordinates from the first text file
with open('output_2.txt', 'r') as file:
    coordinates = [tuple(map(int, line.split())) for line in file]

# Read RGB values from the second text file
with open('rgb_2.txt', 'r') as file:
    rgb_values = [tuple(map(int, line.split())) for line in file]

# Create a black image with width 1830 and height 400
image = np.zeros((400, 1840, 3), dtype=np.uint8)

# Set RGB values to each pixel in the image based on coordinates
for coord, rgb in zip(coordinates, rgb_values):
    x, y = coord
    image[y, x] = rgb

# Save the new image
cv2.imwrite('output_image.png', image)

