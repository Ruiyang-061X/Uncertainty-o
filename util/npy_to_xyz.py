import os
import numpy as np


def npy_to_xyz(dir_path):
    npy_files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
    for npy_file in npy_files:
        npy_file_path = os.path.join(dir_path, npy_file)
        xyz_file_name = os.path.splitext(npy_file)[0] + '.xyz'
        xyz_file_path = os.path.join(dir_path, xyz_file_name)

        point_cloud = np.load(npy_file_path)
        with open(xyz_file_path, 'w') as f:
            for point in point_cloud:
                if point.shape[0] == 6:
                    f.write(f"{point[0]} {point[1]} {point[2]} {int(255 * point[3])} {int(255 * point[4])} {int(255 * point[5])}\n")
                else:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")


if __name__ == "__main__":
    dir_path = "/data/lab/yan/huzhang/huzhang1/code/Uncertainty-o1/output/perturbation/point_cloud"
    npy_to_xyz(dir_path)