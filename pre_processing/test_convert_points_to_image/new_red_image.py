import cv2
import numpy as np

# Read coordinates from the first text file
with open('output_2.txt', 'r') as file:
    coordinates = [tuple(map(int, line.split())) for line in file]

# Read RGB values from the second text file
with open('rgb_2.txt', 'r') as file:
    red_values = [int(line.split()[2]) for line in file]

# Create a black image with width 1830 and height 400
image = np.zeros((400, 1840, 3), dtype=np.uint8)

# Set red values to each pixel in the image based on coordinates
# Set green and blue values to 0
for coord, red in zip(coordinates, red_values):
    x, y = coord
    image[y, x] = [0, 0, red]  # Set red value, and green and blue to 0

# Save the new image
cv2.imwrite('output_red_image.png', image)
