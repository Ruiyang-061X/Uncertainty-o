import random
import numpy as np
from PIL import Image, ImageFilter
from moviepy.editor import VideoFileClip, ImageSequenceClip, vfx, concatenate_videoclips
from moviepy.video.fx.all import mirror_x, mirror_y


def play_speed(video, speed):
    try:
        video_perturbed = video.fx(vfx.speedx, speed)
        return video_perturbed
    except:
        return video


def temporal_shifting(video, factor):
    try:
        shift_duration = abs(factor) * video.duration
        if factor < 0:
            shift_video = video.subclip(0, shift_duration)
            main_video = video.subclip(shift_duration)
            video_perturbed = concatenate_videoclips([main_video, shift_video])
        else:
            shift_video = video.subclip(video.duration - shift_duration)
            main_video = video.subclip(0, video.duration - shift_duration)
            video_perturbed = concatenate_videoclips([shift_video, main_video])
        return video_perturbed
    except:
        return video


def temporal_random_cropping(video, scale):
    try:
        crop_duration = video.duration * scale
        max_start_time = video.duration - crop_duration
        start_time = random.uniform(0, max_start_time)
        video_perturbed = video.subclip(start_time, start_time + crop_duration)
        return video_perturbed
    except:
        return video


def frame_dropping(video, p):
    try:
        frames_selected = []
        for frame in video.iter_frames(fps=video.fps, dtype='uint8'):
            if random.random() > p:
                frames_selected.append(frame)
        video_perturbed = ImageSequenceClip(frames_selected, fps=video.fps)
        return video_perturbed
    except:
        return video


def video_blurring(video, factor):
    try:
        def blur_frame(frame):
            frame_image = Image.fromarray(frame)
            blurred_image = frame_image.filter(ImageFilter.GaussianBlur(radius=factor))
            return np.array(blurred_image)
        video_perturbed = video.fl_image(blur_frame)
        return video_perturbed
    except:
        return video

    
def video_brightness_adjustment(video, brightness):
    try:
        video_perturbed = video.fx(vfx.colorx, brightness)
        return video_perturbed
    except:
        return video


def video_contrast_adjustment(video, contrast):
    try:
        video_perturbed = video.fx(vfx.colorx, contrast)
        return video_perturbed
    except:
        return video


def video_rotatation(video, angle):
    try:
        video_perturbed = video.rotate(angle)
        return video_perturbed
    except:
        return video


def video_flipping(video, dir):
    try:
        if dir == "horizontal":
            video_perturbed = video.fx(mirror_x)
        elif dir == "vertical":
            video_perturbed = video.fx(mirror_y)
        return video_perturbed
    except:
        return video
    

def video_random_cropping(video, scale):
    try:
        crop_width = int(video.w * scale)
        crop_height = int(video.h * scale)
        x_start = np.random.randint(0, video.w - crop_width + 1)
        y_start = np.random.randint(0, video.h - crop_height + 1)
        video_perturbed = video.crop(x1=x_start, y1=y_start, width=crop_width, height=crop_height)
        return video_perturbed
    except:
        return video


def read_video(path):
    video = VideoFileClip(path)
    return video


def save_vide(video, path):
    video.write_videofile(path, codec="libx264", audio_codec="aac")


def perturbation_of_video_prompt(args, idx, video):
    video = read_video(video)
    perturbed_video_list = []
    if args.video_perturbation == 'play_speed':
        for i in [0.5, 0.75, 1.25, 1.5, 2]:
            video_perturbed = play_speed(video, i)
            video_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/video/{idx}_play_speed_{i}.mp4"
            save_vide(video_perturbed, video_perturbed_path)
            perturbed_video_list.append(video_perturbed_path)
    elif args.video_perturbation == 'temporal_shifting':
        for i in [-0.4, -0.2, 0.1, 0.3, 0.5]:
            video_perturbed = temporal_shifting(video, i)
            video_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/video/{idx}_temporal_shifting_{i}.mp4"
            save_vide(video_perturbed, video_perturbed_path)
            perturbed_video_list.append(video_perturbed_path)
    elif args.video_perturbation == 'temporal_random_cropping':
        for i in [0.9, 0.8, 0.7, 0.6, 0.5]:
            video_perturbed = temporal_random_cropping(video, i)
            video_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/video/{idx}_temporal_random_cropping_{i}.mp4"
            save_vide(video_perturbed, video_perturbed_path)
            perturbed_video_list.append(video_perturbed_path)
    elif args.video_perturbation == 'frame_dropping':
        for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
            video_perturbed = frame_dropping(video, i)
            video_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/video/{idx}_frame_dropping_{i}.mp4"
            save_vide(video_perturbed, video_perturbed_path)
            perturbed_video_list.append(video_perturbed_path)
    elif args.video_perturbation == 'video_blurring':
        for i in [1, 2, 3, 4, 5]:
            video_perturbed = video_blurring(video, i)
            video_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/video/{idx}_video_blurring_{i}.mp4"
            save_vide(video_perturbed, video_perturbed_path)
            perturbed_video_list.append(video_perturbed_path)
    elif args.video_perturbation == 'video_brightness_adjustment':
        for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
            video_perturbed = video_brightness_adjustment(video, i)
            video_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/video/{idx}_video_brightness_adjustment_{i}.mp4"
            save_vide(video_perturbed, video_perturbed_path)
            perturbed_video_list.append(video_perturbed_path)
    elif args.video_perturbation == 'video_contrast_adjustment':
        for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
            video_perturbed = video_contrast_adjustment(video, i)
            video_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/video/{idx}_video_contrast_adjustment_{i}.mp4"
            save_vide(video_perturbed, video_perturbed_path)
            perturbed_video_list.append(video_perturbed_path)
    elif args.video_perturbation == 'video_rotatation':
        for i in [-20, -10, 10, 20, 40]:
            video_perturbed = video_rotatation(video, i)
            video_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/video/{idx}_video_rotatation_{i}.mp4"
            save_vide(video_perturbed, video_perturbed_path)
            perturbed_video_list.append(video_perturbed_path)
    elif args.video_perturbation == 'video_flipping':
        for i in ['horizontal', 'vertical', 'horizontal', 'vertical', 'horizontal']:
            video_perturbed = video_flipping(video, i)
            video_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/video/{idx}_video_flipping_{i}.mp4"
            save_vide(video_perturbed, video_perturbed_path)
            perturbed_video_list.append(video_perturbed_path)
    elif args.video_perturbation == 'video_random_cropping':
        for i in [0.9, 0.8, 0.7, 0.6, 0.5]:
            video_perturbed = video_random_cropping(video, i)
            video_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/video/{idx}_video_random_cropping_{i}.mp4"
            save_vide(video_perturbed, video_perturbed_path)
            perturbed_video_list.append(video_perturbed_path)
    return perturbed_video_list


if __name__ == "__main__":
    video = VideoFileClip('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/1.mp4')

    for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
        video_perturbed = play_speed(video, i)
        video_perturbed.write_videofile(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/video/1_play_speed_{i}.mp4', codec="libx264", audio_codec="aac")

    for i in [-0.4, -0.2, 0.1, 0.3, 0.5]:
        video_perturbed = temporal_shifting(video, i)
        video_perturbed.write_videofile(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/video/1_temporal_shifting_{i}.mp4', codec="libx264", audio_codec="aac")

    for i in [0.9, 0.8, 0.7, 0.6, 0.5]:
        video_perturbed = temporal_random_cropping(video, i)
        video_perturbed.write_videofile(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/video/1_temporal_random_cropping_{i}.mp4', codec="libx264", audio_codec="aac")

    for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
        video_perturbed = frame_dropping(video, i)
        video_perturbed.write_videofile(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/video/1_frame_dropping_{i}.mp4', codec="libx264", audio_codec="aac")

    for i in [1, 2, 3, 4, 5]:
        video_perturbed = video_blurring(video, i)
        video_perturbed.write_videofile(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/video/1_video_blurring_{i}.mp4', codec="libx264", audio_codec="aac")

    for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
        video_perturbed = video_brightness_adjustment(video, i)
        video_perturbed.write_videofile(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/video/1_video_brightness_adjustment_{i}.mp4', codec="libx264", audio_codec="aac")
        
    for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
        video_perturbed = video_contrast_adjustment(video, i)
        video_perturbed.write_videofile(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/video/1_video_contrast_adjustment_{i}.mp4', codec="libx264", audio_codec="aac")

    for i in [-20, -10, 10, 20, 40]:
        video_perturbed = video_rotatation(video, i)
        video_perturbed.write_videofile(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/video/1_video_rotatation_adjustment_{i}.mp4', codec="libx264", audio_codec="aac")

    for i in ['horizontal', 'vertical', 'horizontal', 'vertical', 'horizontal']:
        video_perturbed = video_flipping(video, i)
        video_perturbed.write_videofile(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/video/1_video_flipping_{i}.mp4', codec="libx264", audio_codec="aac")

    for i in [0.9, 0.8, 0.7, 0.6, 0.5]:
        video_perturbed = video_random_cropping(video, i)
        video_perturbed.write_videofile(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/video/1_video_random_cropping_{i}.mp4', codec="libx264", audio_codec="aac")