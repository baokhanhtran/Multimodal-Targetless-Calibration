import os
import cv2
import numpy as np

def convert_rgb_to_bgr(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Adjust the file extensions accordingly
            # Read the RGB image
            rgb_image_path = os.path.join(input_folder, file_name)
            rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)

            if rgb_image is not None:
                # Convert RGB to BGR
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                # Save the BGR image to the output folder
                output_file_path = os.path.join(output_folder, file_name)
                cv2.imwrite(output_file_path, bgr_image)
                print(f"Converted and saved: {output_file_path}")
            else:
                print(f"Failed to read RGB image: {rgb_image_path}")

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
                height = 400  # Change this to the desired height
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
    folder_pairs = [
        ("cam0", "cam0_convert"),
        ("cam1_trans_1", "cam1_convert"),
        ("cam2", "cam2_convert"),
        ("cam3", "cam3_convert"),
        ("cam4", "cam4_convert")
    ]

    for input_folder, output_folder in folder_pairs:
        convert_rgb_to_bgr(input_folder, output_folder)

# Input and output folder paths
    folder_pairs_warp = [
        ("cam0_convert", "cam0_convert"),
        ("cam1_convert", "cam1_convert"),
        ("cam2_convert", "cam2_convert"),
        ("cam3_convert", "cam3_convert"),
        ("cam4_convert", "cam4_convert")
    ]

    K = np.array([[268.45972, 0., 312.],
                [0., 268.45972, 312.],
                [0., 0., 1.]], dtype=np.float32)

    R = np.array([[7.9853165e-01, 5.8726855e-03, 6.0192424e-01],
                [-5.0797407e-03, 9.9998260e-01, -3.0173990e-03],
                [-6.0193139e-01, -6.4813346e-04, 7.9854763e-01]], dtype=np.float32)

    warper = cv2.PyRotationWarper("spherical", 266.8302856436542)

    for input_folder_warp, output_folder_warp in folder_pairs_warp:
        warp_image(input_folder_warp, output_folder_warp)
    # Folder path containing images
    folder_cuts = [
        ("cam0_convert", "cam0_convert", 58, 58, 58, 30),
        ("cam1_convert", "cam1_convert", 58, 58, 55, 33),
        ("cam2_convert", "cam2_convert", 58, 55, 56, 32),
        ("cam3_convert", "cam3_convert", 67, 58, 60, 28),
        ("cam4_convert", "cam4_convert", 58, 58, 62, 26)
    ]

    for input_folder_crop, output_folder_crop, left_cut, right_cut, top_cut, bottom_cut in folder_cuts:
        crop_image(input_folder_crop, output_folder_crop, top_cut, bottom_cut, left_cut, right_cut)

    # Path to the five folders
    folder_paths = ["cam3_trans", "cam4_trans", "cam0_trans", "cam1_trans", "cam2_trans"]

    # Output folder to save the concatenated images
    output_folder = "stitch_2"

    concat_image(folder_paths, output_folder) 
