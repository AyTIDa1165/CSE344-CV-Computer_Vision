import cv2
import os

# Folder containing images
image_folder = "CamVid/test_images"
output_video = "output_video.mp4"

# Get all image files and sort them to maintain sequence
images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])

# Read the first image to get frame size
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Change to "XVID" for AVI
fps = 30  # Frames per second
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Add each image to video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release video writer
video.release()
cv2.destroyAllWindows()
