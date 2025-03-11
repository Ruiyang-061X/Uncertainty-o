import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt


def time_stretch(audio, rate):
    try:
        audio_perturbed = librosa.effects.time_stretch(audio, rate=rate)
        return audio_perturbed
    except:
        return audio


def time_shift(audio, shift_factor):
    try:
        audio_perturbed = np.roll(audio, int(shift_factor * len(audio)))
        return audio_perturbed
    except:
        return audio


def adjust_volume(audio, degree):
    try:
        audio_perturbed = audio * degree
        return audio_perturbed
    except:
        return audio


def pitch_shift(audio, sr, n_steps):
    try:
        audio_perturbed = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        return audio_perturbed
    except:
        return audio


def add_noise(audio, noise_level):
    try:
        audio_perturbed = audio + noise_level * np.random.randn(len(audio))
        return audio_perturbed
    except:
        return audio


def high_pass_filter(audio, sr, cutoff):
    try:
        sos = butter(10, cutoff, btype='high', fs=sr, output='sos')
        audio_perturbed = sosfilt(sos, audio)
        return audio_perturbed
    except:
        return audio


def low_pass_filter(audio, sr, cutoff):
    try:
        sos = butter(10, cutoff, btype='low', fs=sr, output='sos')
        audio_perturbed = sosfilt(sos, audio)
        return audio_perturbed
    except:
        return audio


def band_pass_filter(audio, sr, lowcut, highcut):
    try:
        sos = butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
        audio_perturbed = sosfilt(sos, audio)
        return audio_perturbed
    except:
        return audio


def read_audio(path):
    audio, sr = librosa.load(path)
    return audio, sr


def save_vide(audio, sr, path):
    sf.write(path, audio, sr)


def perturbation_of_audio_prompt(args, idx, audio):
    audio, sr = read_audio(audio)
    perturbed_audio_list = []
    if args.audio_perturbation == 'time_stretch':
        for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
            audio_perturbed = time_stretch(audio, i)
            audio_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/audio/{idx}_time_stretch_{i}.wav"
            save_vide(audio_perturbed, sr, audio_perturbed_path)
            perturbed_audio_list.append(audio_perturbed_path)
    elif args.audio_perturbation == 'time_shift':
        for i in [-0.4, -0.2, 0.1, 0.3, 0.5]:
            audio_perturbed = time_shift(audio, i)
            audio_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/audio/{idx}_time_shift_{i}.wav"
            save_vide(audio_perturbed, sr, audio_perturbed_path)
            perturbed_audio_list.append(audio_perturbed_path)
    if args.audio_perturbation == 'adjust_volume':
        for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
            audio_perturbed = adjust_volume(audio, i)
            audio_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/audio/{idx}_adjust_volume_{i}.wav"
            save_vide(audio_perturbed, sr, audio_perturbed_path)
            perturbed_audio_list.append(audio_perturbed_path)
    elif args.audio_perturbation == 'pitch_shift':
        for i in [-2, -1, 1, 2, 3]:
            audio_perturbed = pitch_shift(audio, sr, i)
            audio_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/audio/{idx}_pitch_shift_{i}.wav"
            save_vide(audio_perturbed, sr, audio_perturbed_path)
            perturbed_audio_list.append(audio_perturbed_path)
    elif args.audio_perturbation == 'add_noise':
        for i in [0.002, 0.004, 0.006, 0.008, 0.010]:
            audio_perturbed = add_noise(audio, i)
            audio_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/audio/{idx}_add_noise_{i}.wav"
            save_vide(audio_perturbed, sr, audio_perturbed_path)
            perturbed_audio_list.append(audio_perturbed_path)
    elif args.audio_perturbation == 'high_pass_filter':
        for i in [500, 1000, 1500, 2000, 2500]:
            audio_perturbed = high_pass_filter(audio, sr, i)
            audio_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/audio/{idx}_high_pass_filter_{i}.wav"
            save_vide(audio_perturbed, sr, audio_perturbed_path)
            perturbed_audio_list.append(audio_perturbed_path)
    elif args.audio_perturbation == 'low_pass_filter':
        for i in [200, 400, 600, 800, 1000]:
            audio_perturbed = low_pass_filter(audio, sr, i)
            audio_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/audio/{idx}_low_pass_filter_{i}.wav"
            save_vide(audio_perturbed, sr, audio_perturbed_path)
            perturbed_audio_list.append(audio_perturbed_path)
    elif args.audio_perturbation == 'band_pass_filter':
        for (i, j) in zip([100, 200, 300, 400, 500], [2500, 2000, 1500, 1000, 800]):
            audio_perturbed = band_pass_filter(audio, sr, i, j)
            audio_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/audio/{idx}_band_pass_filter_{i}.wav"
            save_vide(audio_perturbed, sr, audio_perturbed_path)
            perturbed_audio_list.append(audio_perturbed_path)
    return perturbed_audio_list


if __name__ == "__main__":
    audio, sr = librosa.load('/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/1.wav')

    for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
        audio_perturbed = time_stretch(audio, i)
        sf.write(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1_time_stretch_{i}.wav', audio_perturbed, sr)

    for i in [-0.4, -0.2, 0.1, 0.3, 0.5]:
        audio_perturbed = time_shift(audio, i)
        sf.write(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1_time_shift_{i}.wav', audio_perturbed, sr)

    for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
        audio_perturbed = adjust_volume(audio, i)
        sf.write(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1_adjust_volume_{i}.wav', audio_perturbed, sr)

    for i in [-2, -1, 1, 2, 3]:
        audio_perturbed = pitch_shift(audio, sr, i)
        sf.write(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1_pitch_shift_{i}.wav', audio_perturbed, sr)

    for i in [0.002, 0.004, 0.006, 0.008, 0.010]:
        audio_perturbed = add_noise(audio, i)
        sf.write(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1_add_noise_{i}.wav', audio_perturbed, sr)

    for i in [500, 1000, 1500, 2000, 2500]:
        audio_perturbed = high_pass_filter(audio, sr, i)
        sf.write(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1_high_pass_filter_{i}.wav', audio_perturbed, sr)

    for i in [200, 400, 600, 800, 1000]:
        audio_perturbed = low_pass_filter(audio, sr, i)
        sf.write(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1_low_pass_filter_{i}.wav', audio_perturbed, sr)

    for (i, j) in zip([100, 200, 300, 400, 500], [2500, 2000, 1500, 1000, 800]):
        audio_perturbed = band_pass_filter(audio, sr, i, j)
        sf.write(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/1_band_pass_filter_{i}_{j}.wav', audio_perturbed, sr)