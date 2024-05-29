import cv2

# Read the two images
image1 = cv2.imread('0000000016.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('0000000000.png', cv2.IMREAD_GRAYSCALE)

# Manually adjust contrast and brightness
alpha = 0.7  # Contrast control (1.0 means no change)
beta = 30    # Brightness control (0-100, where 0 is black)

# Apply the adjustment to both images
adjusted_image1 = cv2.convertScaleAbs(image1, alpha=alpha, beta=beta)
adjusted_image2 = cv2.convertScaleAbs(image2, alpha=alpha, beta=beta)

# Display the original and adjusted images
cv2.imwrite('fix_part_1.png', adjusted_image1)
cv2.imwrite('fix_part_2.png', adjusted_image2)
cv2.imshow('Adjusted Image 1', adjusted_image1)
cv2.imshow('Adjusted Image 2', adjusted_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
