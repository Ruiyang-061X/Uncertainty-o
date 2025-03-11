import numpy as np
from datetime import datetime
import trimesh


def get_cur_time():
    return datetime.today().strftime('%Y_%m_%d_%H_%M_%S')


def custom_serializer(obj):
    return obj.__class__.__name__


def read_obj_file(obj_path):
        points = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    x, y, z = map(float, parts[1:4])
                    points.append([x, y, z])
        return np.array(points)


def ply_to_npy(ply_path):
    mesh = trimesh.load(ply_path)
    npy_data = np.array(mesh.vertices)
    npy_save_path = ply_path.replace('.ply', '.npy')
    np.save(npy_save_path, npy_data)
    return npy_save_path