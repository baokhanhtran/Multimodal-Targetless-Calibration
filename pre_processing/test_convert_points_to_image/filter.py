import cv2
import numpy as np

# Load the image
src = cv2.imread("0000000016.png", cv2.IMREAD_COLOR)

# Split the image into individual color channels
b, g, r = cv2.split(src)

# Create blank images with the same size and stack the individual channels to create RGB images
blue_image = np.zeros_like(src)
blue_image[:, :, 0] = b

green_image = np.zeros_like(src)
green_image[:, :, 1] = g

red_image = np.zeros_like(src)
red_image[:, :, 2] = r

# Save each channel as a separate image
cv2.imwrite("blue.png", blue_image)
cv2.imwrite("green.png", green_image)
cv2.imwrite("red.png", red_image)
