import cv2
import numpy as np

# Load the three images
image1 = cv2.imread("output_blue_image.png")
image2 = cv2.imread("output_green_image.png")
image3 = cv2.imread("output_red_image.png")

# Ensure all images have the same size


# Add the images together
result_image = image1 + image2 + image3

# Save the result image
cv2.imwrite("result_image.png", result_image)

