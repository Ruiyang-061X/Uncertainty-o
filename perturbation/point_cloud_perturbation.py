import numpy as np


def rotation(point_cloud, angle):
    try:
        point_cloud_perturbed = point_cloud.copy()
        angle = np.radians(angle)
        rot_matrix = np.array([[np.cos(angle),  -np.sin(angle),     0],
                               [np.sin(angle),   np.cos(angle),     0],
                               [            0,               0,     1]])
        point_cloud_perturbed[:, :3] = np.dot(point_cloud[:, :3], rot_matrix.T)
        return point_cloud_perturbed
    except:
        return point_cloud


def reflecting(point_cloud, axis):
    try:
        point_cloud_perturbed = point_cloud.copy()
        point_cloud_perturbed[:, :3][:, axis] = -point_cloud_perturbed[:, :3][:, axis]
        return point_cloud_perturbed
    except:
        return point_cloud


def scaling(point_cloud, factor):
    try:
        point_cloud_perturbed = point_cloud.copy()
        point_cloud_perturbed[:, :3] *= factor
        return point_cloud_perturbed
    except:
        return point_cloud


def translation(point_cloud, shift):
    try:
        point_cloud_perturbed = point_cloud.copy()
        point_cloud_perturbed[:, :3] += np.array(shift)
        return point_cloud_perturbed
    except:
        return point_cloud


def random_sampling(point_cloud, rate):
    try:
        point_cloud_perturbed = point_cloud.copy()
        indices = np.random.choice(point_cloud.shape[0], int(point_cloud.shape[0] * rate), replace=False)
        point_cloud_perturbed = point_cloud_perturbed[indices]
        return point_cloud_perturbed
    except:
        return point_cloud


def jittering(point_cloud, sigma, clip):
    try:
        point_cloud_perturbed = point_cloud.copy()
        jitter = np.clip(sigma * np.random.randn(*point_cloud_perturbed[:, :3].shape), -clip, clip)
        point_cloud_perturbed[:, :3] += jitter
        return point_cloud_perturbed
    except:
        return point_cloud


def shearing(point_cloud, shear_factor):
    try:
        point_cloud_perturbed = point_cloud.copy()
        shear_matrix = np.array([[1, shear_factor,      0],
                                 [0,            1,      0],
                                 [0,            0,      1]])
        point_cloud_perturbed[:, :3] = np.dot(point_cloud_perturbed[:, :3], shear_matrix.T)
        return point_cloud_perturbed
    except:
        return point_cloud


def outlier_injection(point_cloud, num_outliers, spread):
    try:
        point_cloud_perturbed = point_cloud.copy()
        outliers = (np.random.rand(num_outliers, 3) - 0.5) * 2 * spread + point_cloud_perturbed[:, :3].mean(axis=0)
        black_color = np.zeros((num_outliers, point_cloud_perturbed.shape[1] - 3))
        outliers = np.hstack([outliers, black_color])
        point_cloud_perturbed = np.vstack([point_cloud_perturbed, outliers])
        return point_cloud_perturbed
    except:
        return point_cloud


def to_8192(point_cloud):
    try:
        num_points = point_cloud.shape[0]
        if num_points < 8192:
            padding = np.zeros((8192 - num_points, 6))
            point_cloud_8192 = np.vstack((point_cloud, padding))
        elif num_points > 8192:
            indices = np.random.choice(num_points, 8192, replace=False)
            point_cloud_8192 = point_cloud[indices]
        else:
            point_cloud_8192 = point_cloud
        return point_cloud_8192
    except:
        return point_cloud


def read_point_cloud(point_cloud):
    if isinstance(point_cloud, str):
        point_cloud = np.load(point_cloud)
    return point_cloud


def save_point_cloud(point_cloud, path):
    np.save(path, point_cloud)


def perturbation_of_point_cloud_prompt(args, idx, point_cloud):
    point_cloud = read_point_cloud(point_cloud)
    perturbed_point_cloud_list = []
    if args.point_cloud_perturbation == 'rotation':
        for i in [-20, -10, 10, 20, 40]:
            point_cloud_perturbed = rotation(point_cloud, i)
            point_cloud_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/{idx}_rotation_{i}.npy"
            save_point_cloud(point_cloud_perturbed, point_cloud_perturbed_path)
            perturbed_point_cloud_list.append(point_cloud_perturbed_path)
    elif args.point_cloud_perturbation == 'reflecting':
        for i in [0, 1, 2, 0, 1]:
            point_cloud_perturbed = reflecting(point_cloud, i)
            point_cloud_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/{idx}_reflecting_{i}.npy"
            save_point_cloud(point_cloud_perturbed, point_cloud_perturbed_path)
            perturbed_point_cloud_list.append(point_cloud_perturbed_path)
    elif args.point_cloud_perturbation == 'scaling':
        for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
            point_cloud_perturbed = scaling(point_cloud, i)
            point_cloud_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/{idx}_scaling_{i}.npy"
            save_point_cloud(point_cloud_perturbed, point_cloud_perturbed_path)
            perturbed_point_cloud_list.append(point_cloud_perturbed_path)
    elif args.point_cloud_perturbation == 'translation':
        for i in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]]:
            point_cloud_perturbed = translation(point_cloud, i)
            point_cloud_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/{idx}_translation_{'_'.join(i)}.npy"
            save_point_cloud(point_cloud_perturbed, point_cloud_perturbed_path)
            perturbed_point_cloud_list.append(point_cloud_perturbed_path)
    if args.point_cloud_perturbation == 'random_sampling':
        for i in [0.9, 0.8, 0.7, 0.6, 0.5]:
            point_cloud_perturbed = random_sampling(point_cloud, i)
            point_cloud_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/{idx}_random_sampling_{i}.npy"
            save_point_cloud(point_cloud_perturbed, point_cloud_perturbed_path)
            perturbed_point_cloud_list.append(point_cloud_perturbed_path)
    elif args.point_cloud_perturbation == 'jittering':
        for i, j in [(0.005, 0.01), (0.01, 0.02), (0.01, 0.05), (0.05, 0.05), (0.05, 0.1)]:
            point_cloud_perturbed = jittering(point_cloud, i, j)
            point_cloud_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/{idx}_jittering_{i}_{j}.npy"
            save_point_cloud(point_cloud_perturbed, point_cloud_perturbed_path)
            perturbed_point_cloud_list.append(point_cloud_perturbed_path)
    elif args.point_cloud_perturbation == 'shearing':
        for i in [0.5, 1.0, 1.5, 2.0, 2.5]:
            point_cloud_perturbed = shearing(point_cloud, i)
            point_cloud_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/{idx}_shearing_{i}.npy"
            save_point_cloud(point_cloud_perturbed, point_cloud_perturbed_path)
            perturbed_point_cloud_list.append(point_cloud_perturbed_path)
    elif args.point_cloud_perturbation == 'outlier_injection':
        for i, j in [(100, 0.5), (200, 1.0), (300, 1.5), (400, 2.0), (500, 2.5)]:
            point_cloud_perturbed = outlier_injection(point_cloud, i, j)
            point_cloud_perturbed_path = f"/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/{idx}_outlier_injection_{i}_{j}.npy"
            save_point_cloud(point_cloud_perturbed, point_cloud_perturbed_path)
            perturbed_point_cloud_list.append(point_cloud_perturbed_path)
    return perturbed_point_cloud_list


if __name__ == "__main__":
    point_cloud_path = '/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/.asset/point_cloud_duck.npy'
    point_cloud = np.load(point_cloud_path)

    for i in [-20, -10, 10, 20, 40]:
        point_cloud_perturbed = rotation(point_cloud, i)
        np.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/point_cloud_duck_rotation_{i}.npy', point_cloud_perturbed)

    for i in [0, 1, 2, 0, 1]:
        point_cloud_perturbed = reflecting(point_cloud, i)
        np.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/point_cloud_duck_reflecting_{i}.npy', point_cloud_perturbed)

    for i in [0.5, 0.75, 1.25, 1.5, 2.0]:
        point_cloud_perturbed = scaling(point_cloud, i)
        np.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/point_cloud_duck_scaling_{i}.npy', point_cloud_perturbed)

    for i in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0]]:
        point_cloud_perturbed = translation(point_cloud, i)
        np.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/point_cloud_duck_translation_{"_".join(map(str, i))}.npy', point_cloud_perturbed)

    for i in [0.9, 0.8, 0.7, 0.6, 0.5]:
        point_cloud_perturbed = random_sampling(point_cloud, i)
        point_cloud_perturbed = to_8192(point_cloud_perturbed)
        np.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/point_cloud_duck_random_sampling_{i}.npy', point_cloud_perturbed)

    for i, j in [(0.005, 0.01), (0.01, 0.02), (0.01, 0.05), (0.05, 0.05), (0.05, 0.1)]:
        point_cloud_perturbed = jittering(point_cloud, i, j)
        np.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/point_cloud_duck_jittering_{i}_{j}.npy', point_cloud_perturbed)

    for i in [0.5, 1.0, 1.5, 2.0, 2.5]:
        point_cloud_perturbed = shearing(point_cloud, i)
        np.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/point_cloud_duck_shearing_{i}.npy', point_cloud_perturbed)

    for i, j in [(100, 0.5), (200, 1.0), (300, 1.5), (400, 2.0), (500, 2.5)]:
        point_cloud_perturbed = outlier_injection(point_cloud, i, [j, j, j])
        point_cloud_perturbed = to_8192(point_cloud_perturbed)
        np.save(f'/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud/point_cloud_duck_outlier_injection_{i}_{j}.npy', point_cloud_perturbed)