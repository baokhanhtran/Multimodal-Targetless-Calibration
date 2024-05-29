import cv2
import numpy as np

# Load the PNG image
image = cv2.imread('0000000016.png')  # Replace with the path to your PNG image

# Read coordinates from the text file
with open('output_2.txt', 'r') as file:
    coordinates = [tuple(map(int, line.split())) for line in file]

# Function to get RGB values from coordinates
# def get_rgb_values(img, coords):
#     rgb_values = []
#     for coord in coords:
#         x, y = coord
#         if x == 0 and y == 70:
#             rgb = (255, 255, 255)
#         else:
#             rgb = img[y, x]  # OpenCV uses (y, x) indexing
#         rgb_values.append(tuple(rgb))
#     return rgb_values

def get_rgb_values(img, coords):
    rgb_values = []
    for coord in coords:
        x, y = coord
        rgb = img[y, x]  # OpenCV uses (y, x) indexing
        rgb_values.append(tuple(rgb))
    return rgb_values

# Get RGB values for the coordinates
rgb_values = get_rgb_values(image, coordinates)

# Write RGB values and coordinates to the output file
with open('rgb_2.txt', 'w') as output_file:
    for coord, rgb in zip(coordinates, rgb_values):
        # Convert the RGB values to the desired format
        rgb_str = " ".join(map(str, rgb))
        output_file.write(f"{rgb_str}\n")

