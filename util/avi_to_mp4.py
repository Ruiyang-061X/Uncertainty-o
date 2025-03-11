from moviepy.editor import VideoFileClip

# Load the .avi file
video = VideoFileClip("/data/lab/yan/huzhang/huzhang1/data/MSVDQA/video/5-8z5U-o4O4_0_3.avi")

# Write the video to a new .mp4 file
video.write_videofile("/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/output.mp4", codec="libx264")