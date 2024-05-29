import os
import cv2
import numpy as np

def calib(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            # Read the image
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            # Undistort the image
            h, w = img.shape[:2]
            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
            undistorted_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

            # Save the result
            output_path = os.path.join(output_folder, f'undistorted_{filename}')
            cv2.imwrite(output_path, undistorted_img)

    print('Distortion correction completed.')

# Loop through all images in the input folder
def warp_image(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg")):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Warp the image
            _, warped_image = warper.warp(
                image,
                K,
                R,
                cv2.INTER_LINEAR,
                cv2.BORDER_REFLECT,
            )

            # Save the warped image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, warped_image)
    print("Warping completed.")

# Loop through all images in the input folder
def crop_image(input_folder, output_folder, top_cut, bottom_cut, left_cut, right_cut):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg")):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Crop the image
            cropped_image = image[top_cut:image.shape[0] - bottom_cut, left_cut:image.shape[1] - right_cut]

            # Save the cropped image to the new folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cropped_image)
    print("Cropping and saving to a new folder completed.")

# Loop through all images in the input folder
def new_name(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    # List all files in the source folder
    file_list = sorted(os.listdir(input_folder))  # Sort the files to maintain order
    # Loop through each file
    for sequence_number, file_name in enumerate(file_list, start=0):
        # Check if the file is an image (you may want to add more checks)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load the image
            image = cv2.imread(os.path.join(input_folder, file_name))

            # Save the image to the destination folder with the new name format
            new_name = '{0:010d}.png'.format(sequence_number)
            output_path = os.path.join(output_folder, new_name)
            cv2.imwrite(output_path, image)
    print("Renewing and saving to a new folder completed.")

# Loop through each file
def  split_image(source_folder, left_destination_folder, right_destination_folder):
    # List all files in the source folder
    os.makedirs(left_destination_folder, exist_ok=True)
    os.makedirs(right_destination_folder, exist_ok=True)
    file_list = os.listdir(source_folder)   
    for file_name in file_list:
        # Check if the file is an image (you may want to add more checks)
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load the image
            image = cv2.imread(os.path.join(source_folder, file_name))

            # Get image width
            width = image.shape[1]

            # Split the image into left and right halves
            left_half = image[:, :width // 2]
            right_half = image[:, width // 2:]

            # Save the left half to the left destination folder
            left_output_path = os.path.join(left_destination_folder, file_name)
            cv2.imwrite(left_output_path, left_half)

            # Save the right half to the right destination folder
            right_output_path = os.path.join(right_destination_folder, file_name)
            cv2.imwrite(right_output_path, right_half)
    print("Finish splitting")

def concat_image(folder_paths, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    # Number of images in each folder
    num_images = 40  # Change this to the actual number of images
    # Iterate over the image numbers
    for image_num in range(0, num_images + 1):
        # Initialize an empty list to store images for concatenation
        images_to_concat = []

        # Iterate over the five folders
        for folder_path in folder_paths:
            # Construct the file name based on the image number and folder
            file_name = '{0:010d}.png'.format(image_num)

            # Read the image
            image = cv2.imread(os.path.join(folder_path, file_name))

            # Check if the image was successfully loaded
            if image is not None:
                # Resize the image if needed
                # (adjust the dimensions based on your requirements)
                height = 340  # Change this to the desired height
                image = cv2.resize(image, (int(height * image.shape[1] / image.shape[0]), height))

                # Append the image to the list
                images_to_concat.append(image)

        # Check if at least one image was successfully loaded
        if images_to_concat:
            # Concatenate the images horizontally
            result = np.hstack(images_to_concat)

            # Save the concatenated image with the specified format
            output_file = os.path.join(output_folder, '{0:010d}.png'.format(image_num))
            cv2.imwrite(output_file, result)
    print("Concatenation completed.")

if __name__ == "__main__":
    # Input and output folders
    folder_pairs_calib = [
        ('cam0', 'cam0_convert'),
        ('cam1', 'cam1_convert'),
        ('cam2', 'cam2_convert'),
        ('cam3', 'cam3_convert'),
        ('cam4', 'cam4_convert'),
        ('cam5', 'cam5_convert')
    ]

    mtx = np.array([[498.5665, 0, 319.7037],
                    [0, 499.2976, 243.6018],
                    [0, 0, 1]])
    dist = np.array([-0.3688, 0.1143, 0, 0, 0])

    for input_folder_calib, output_folder_calib in folder_pairs_calib:
        calib(input_folder_calib, output_folder_calib)
# Input and output folder paths

    folder_pairs_warp = [
        ("cam0_convert", "cam0_convert"),
        ("cam1_convert", "cam1_convert"),
        ("cam2_convert", "cam2_convert"),
        ("cam3_convert", "cam3_convert"),
        ("cam4_convert", "cam4_convert"),
        ("cam5_convert", "cam5_convert")
    ]

    K = np.array([[418.05525585857526, 0., 320.],
                [0., 418.05525585857526, 256.],
                [0., 0., 1.]], dtype=np.float32)

    R = np.array([[1, 0, 0.1],
                [0, 1, 0],
                [-0.1, 0, 1]], dtype=np.float32)

    warper = cv2.PyRotationWarper("spherical", 418.05525585857526)

    for input_folder_warp, output_folder_warp in folder_pairs_warp:
        warp_image(input_folder_warp, output_folder_warp)

    # Folder path containing images
    folder_cuts_crop = [
        ("cam0_convert", "cam0_convert", 71, 71, 59, 56),
        ("cam1_convert", "cam1_convert", 71, 84, 68, 47),
        ("cam2_convert", "cam2_convert", 85, 75, 59, 56),
        ("cam3_convert", "cam3_convert", 75, 75, 49, 66),
        ("cam4_convert", "cam4_convert", 85, 75, 57, 58),
        ("cam5_convert", "cam5_convert", 81, 74, 64, 51)
    ]

    for input_folder_crop, output_folder_crop, left_cut, right_cut, top_cut, bottom_cut in folder_cuts_crop:
        crop_image(input_folder_crop, output_folder_crop, top_cut, bottom_cut, left_cut, right_cut)

    # Source and destination folders
    source_folder = "cam3_trans"
    left_destination_folder = "cam3_name_left"
    right_destination_folder = "cam3_name_right"

    split_image(source_folder, left_destination_folder, right_destination_folder)

    # Path to the five folders
    folder_paths = ["cam3_name_right", "cam4_trans", "cam5_trans", "cam0_trans", "cam1_trans", "cam2_trans", "cam3_name_left"]

    # Output folder to save the concatenated images
    output_folder = "stitch_2"

    concat_image(folder_paths, output_folder) 

