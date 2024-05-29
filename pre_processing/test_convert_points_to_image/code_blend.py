import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images
image1 = cv2.imread('0000000000.png')  # Replace with the path to your first image
image2 = cv2.imread('0000000016.png')  # Replace with the path to your second image

# Find the dimensions of each image
height1, width1, _ = image1.shape
height2, width2, _ = image2.shape

# Find the maximum dimensions
max_height = max(height1 + 70, height2)
max_width = max(width1, width2)

# Create a blank background image
background = np.zeros((max_height, max_width, 3), dtype=np.uint8)

# Paste the first image onto the background starting from the 85th pixel
background[70:70+height1, :width1] = image1

# Blend the second image onto the background
alpha = 0.8  # Adjust the blending strength
background[:height2, :width2] = cv2.addWeighted(image2, 0.5, background[:height2, :width2], alpha, 0)

# Display the result
plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.title('Blended Images')
plt.axis('off')
plt.show()

# Save the blended result
cv2.imwrite('blended_result.png', background)
