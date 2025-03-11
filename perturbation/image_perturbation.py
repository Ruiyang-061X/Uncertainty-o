import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def image_blurring(image, blur_radius):
    try:
        image_perturbed = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return image_perturbed
    except:
        return image


def image_rotation(image, angle):
    try:
        image_perturbed = image.rotate(angle)
        return image_perturbed
    except:
        return image


def image_flipping(image, direction):
    try:
        if direction == "horizontal":
            image_perturbed = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == "vertical":
            image_perturbed = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image_perturbed
    except:
        return image


def image_shifting(image, direction, length):
    try:
        w, h = image.size
        if direction == 'up':
            translation = (0, length)
        elif direction == 'down':
            translation = (0, -length)
        elif direction == 'left':
            translation = (length, 0)
        elif direction == 'right':
            translation = (-length, 0)        
        image_perturbed = image.transform(
            (w, h), 
            Image.AFFINE, 
            (1, 0, translation[0], 0, 1, translation[1]),
            fillcolor=(0, 0, 0)
        )
        return image_perturbed
    except:
        return image


def image_cropping(image, scale):
    try:
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)
        right = left + new_w
        bottom = top + new_h
        image_perturbed = image.crop((left, top, right, bottom))
        return image_perturbed
    except:
        return image


def image_erasing(image, length):
    try:
        w, h = image.size
        length = min(length, h)
        length = min(length, w)
        top_left_x = random.randint(0, w - length)
        top_left_y = random.randint(0, h - length)
        image_perturbed = image.copy()
        erase_area = Image.new("RGB", (length, length), (0, 0, 0))
        image_perturbed.paste(erase_area, (top_left_x, top_left_y))
        return image_perturbed
    except:
        return image


def adjust_brightness(image, factor):
    try:
        image_perturbed = ImageEnhance.Brightness(image).enhance(factor)
        return image_perturbed
    except:
        return image


def adjust_contrast(image, factor):
    try:
        image_perturbed = ImageEnhance.Contrast(image).enhance(factor)
        return image_perturbed
    except:
        return image


def image_sharpen(image, degree):
    try:
        image_perturbed = ImageEnhance.Sharpness(image).enhance(degree)
        return image_perturbed
    except:
        return image


def gaussian_noise(image, degree):
    try:
        image_array = np.array(image)
        mean = 0
        std_dev = degree * 255
        noise = np.random.normal(mean, std_dev, image_array.shape)
        noisy_image = image_array + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        image_perturbed = Image.fromarray(noisy_image)
        return image_perturbed
    except:
        return image


def dropout(image, p):
    try:
        image_array = np.array(image)
        mask = np.random.rand(*image_array.shape[:2]) > p
        mask = np.expand_dims(mask, axis=-1)
        dropped_image = image_array * mask
        dropped_image = dropped_image.astype(np.uint8)
        image_perturbed = Image.fromarray(dropped_image)
        return image_perturbed
    except:
        return image


def salt_and_pepper(image, p):
    try:
        image_array = np.array(image)
        noise = np.random.rand(image_array.shape[0], image_array.shape[1])
        salt_mask = noise < (p / 2)
        pepper_mask = noise > 1 - (p / 2)
        salt_mask = np.expand_dims(salt_mask, axis=-1).repeat(3, axis=-1)
        pepper_mask = np.expand_dims(pepper_mask, axis=-1).repeat(3, axis=-1)
        image_array[salt_mask] = 255
        image_array[pepper_mask] = 0
        image_perturbed = Image.fromarray(image_array)
        return image_perturbed
    except:
        return image
    

def read_image(image):
    if isinstance(image, str):
        image = Image.open(image)
    return image


def save_image(image, path):
    image.save(path)


def perturbation_of_image_prompt(args, idx, image):
    image = read_image(image)
    image_perturbed_list = []
    if args.image_perturbation == 'image_blurring':
        for i in [1, 2, 3, 4, 5]:
            image_perturbed = image_blurring(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_image_blurring_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'image_rotation':
        for i in [-20, -10, 10, 20, 40]:
            image_perturbed = image_rotation(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_image_rotation_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'image_flipping':
        for i in ['horizontal', 'vertical', 'horizontal', 'vertical', 'horizontal']:
            image_perturbed = image_flipping(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_image_flipping_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'image_shifting':
        for (i, j) in zip(['up', 'down', 'left', 'right', 'up'], [50, 50, 50, 50, 100]):
            image_perturbed = image_shifting(image, i, j)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_image_shifting_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'image_cropping':
        for i in [0.9, 0.8, 0.7, 0.6, 0.5]:
            image_perturbed = image_cropping(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_image_cropping_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'image_erasing':
        for i in [50, 100, 150, 200, 250]:
            image_perturbed = image_erasing(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_image_erasing_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'adjust_brightness':
        for i in [0.6, 0.8, 1.2, 1.4, 1.6]:
            image_perturbed = adjust_brightness(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_adjust_brightness_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'adjust_contrast':
        for i in [0.6, 0.8, 1.2, 1.4, 1.6]:
            image_perturbed = adjust_contrast(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_adjust_contrast_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'image_sharpen':
        for i in [1, 2, 3, 4, 5]:
            image_perturbed = image_sharpen(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_image_sharpen_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'gaussian_noise':
        for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
            image_perturbed = gaussian_noise(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_gaussian_noise_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'dropout':
        for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
            image_perturbed = dropout(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_dropout_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    elif args.image_perturbation == 'salt_and_pepper':
        for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
            image_perturbed = salt_and_pepper(image, i)
            image_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/{args.benchmark}_{idx}_salt_and_pepper_{i}.png"
            save_image(image_perturbed, image_perturbed_path)
            image_perturbed_list.append(image_perturbed_path)
    return image_perturbed_list


if __name__ == "__main__":
    image_path = '/data/lab/yan/huzhang/huzhang1/data/CoCoCap/image/val2014/COCO_val2014_000000001700.jpg'
    image = Image.open(image_path)

    for i in [1, 2, 3, 4, 5]:
        image_perturbed = image_blurring(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_blurring_{i}.png')

    for i in [-20, -10, 10, 20, 40]:
        image_perturbed = image_rotation(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_rotation_{i}.png')

    for i in ['horizontal', 'vertical', 'horizontal', 'vertical', 'horizontal']:
        image_perturbed = image_flipping(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_flipping_{i}.png')

    for (i, j) in zip(['up', 'down', 'left', 'right', 'up'], [50, 50, 50, 50, 100]):
        image_perturbed = image_shifting(image, i, j)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_shifting_{i}_{j}.png')

    for i in [0.9, 0.8, 0.7, 0.6, 0.5]:
        image_perturbed = image_cropping(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_cropping_{i}.png')

    for i in [50, 100, 150, 200, 250]:
        image_perturbed = image_erasing(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_erasing_{i}.png')

    for i in [0.6, 0.8, 1.2, 1.4, 1.6]:
        image_perturbed = adjust_brightness(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_adjust_brightness_{i}.png')

    for i in [0.6, 0.8, 1.2, 1.4, 1.6]:
        image_perturbed = adjust_contrast(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_adjust_contrast_{i}.png')

    for i in [1, 2, 3, 4, 5]:
        image_perturbed = image_sharpen(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_sharpen_{i}.png')

    for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
        image_perturbed = gaussian_noise(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_gaussian_noise_{i}.png')

    for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
        image_perturbed = dropout(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_dropout_{i}.png')

    for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
        image_perturbed = salt_and_pepper(image, i)
        image_perturbed.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/image/COCO_val2014_000000001700_salt_and_pepper_{i}.png')