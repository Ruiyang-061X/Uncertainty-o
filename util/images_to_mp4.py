import cv2
import os

# Parent directory containing all the image folders
parent_dir = '/data/lab/yan/huzhang/huzhang1/data/OpenEQA/'

# List all folders in the parent directory
image_folders = [
        os.path.join(parent_dir, folder) for folder in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, folder))
]

# Iterate over each folder and process images
for folder in image_folders:
    print(folder)
    images = sorted([img for img in os.listdir(folder) if img.endswith('.png')])

    if len(images) == 0:
        print(f"No images found in folder: {folder}")
        continue

    # Read the first image to get the frame size
    first_image = cv2.imread(os.path.join(folder, images[0]))
    height, width, _ = first_image.shape

    # Define the video output file
    output_video = os.path.join(folder + '.mp4')

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
    fps = 10  # Set FPS to a lower value for a slower video (e.g., 5 frames per second)
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write each image to the video file
    for image_name in images:
        image_path = os.path.join(folder, image_name)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release the video writer and move to the next folder
    video_writer.release()
    print(f"Video saved: {output_video}")
