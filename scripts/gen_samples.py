import argparse
import os
import pickle
import time

import cv2
import numpy as np
import open3d as o3d


def plot_colored_pcd(pc, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(color)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, origin_frame])


def plot_pcd(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, origin_frame])


def plot_colored_pcd_pair(pcd1,  pcd2, color1, color2):
    points = np.vstack([pcd1, pcd2])
    colors = np.vstack([color1, color2])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, origin_frame])


def load_point_cloud(pcd_file, dims):
    pc = np.fromfile(pcd_file, dtype=np.float32)
    pc = pc.reshape(pc.shape[0]//dims, dims)
    return pc


def back_project_depth(depth, K):
    Kinv = np.linalg.inv(K)
    height, width = depth.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    k = np.ones(shape=(height, width), dtype=np.float32)
    temp = np.stack([i, j, k], axis=-1)
    temp = np.expand_dims(temp, -1)
    temp = np.squeeze(np.matmul(Kinv, temp))
    depth = np.expand_dims(depth, axis=-1)
    pcd = temp * depth
    return pcd


def create_sampling_grid(pcd, pose, K):
    height = pcd.shape[0]
    width = pcd.shape[1]
    rot = np.matmul(K, pose[:3, :3])
    trans = np.matmul(K, pose[:3, 3].reshape(3, 1))
    pcd = np.expand_dims(pcd, axis=-1)
    grid = np.matmul(rot, pcd) + trans
    grid[:, :, :2] = grid[:, :, :2] / grid[:, :, 2:]
    idx = (grid[:, :, 0] >= 0) & (grid[:, :, 0] < width) & (
        grid[:, :, 1] >= 0) & (grid[:, :, 1] < height)
    return idx, np.squeeze(grid)


def sample_points_zbuffer(depth_src, color_src, idx, grid):
    shape = (grid.shape[0], grid.shape[1])
    depth_out = np.ones(shape=shape) * 100000
    color_out = np.zeros_like(grid)
    corresp = np.zeros(shape=shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not idx[i, j]:
                continue
            x, y, z = grid[i, j, :]
            u = int(x)
            v = int(y)
            d = depth_src[v, u]
            if z < depth_out[i, j] and abs(d-z) < 0.1:
                depth_out[i, j] = z
                color_out[i, j, :] = color_src[v, u, :]
                corresp[i, j] = 1
    return corresp, color_out


def gen_training_samples(args):
    files = os.listdir(args.folder)
    training_sample = dict()
    count = 0
    for file_dir in files:
        print(file_dir)
        abs_path = os.path.abspath(os.path.join(args.folder, file_dir))
        rgb_path = os.path.join(abs_path, 'rgb')
        depth_path = os.path.join(abs_path, 'depth')
        pose_path = os.path.join(abs_path, 'pose')
        model_path = os.path.join(abs_path, 'model')
        intr_file = os.path.join(abs_path, 'intrinsics.txt')
        num_models = len(os.listdir(rgb_path))
        K = np.loadtxt(intr_file)
        for i in range(num_models):
            rgb1 = cv2.imread(os.path.join(
                rgb_path, '{}.jpg'.format(i)), -1)
            cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB, rgb1)
            rgb1 = rgb1/255.0
            depth1 = cv2.imread(os.path.join(
                depth_path, '{}.png'.format(i)), -1) / 5000.0
            pose1 = np.loadtxt(os.path.join(
                pose_path, '{}.txt'.format(i))).reshape(4, 4)
            # cv2.imshow("anchor", rgb1)

            item = dict()
            item['dir'] = file_dir
            item['query'] = i
            item['positives'] = []
            item['nonnegatives'] = []

            for j in range(num_models):
                if i == j:
                    continue
                print('working on {}/{}'.format(i, j))
                rgb2 = cv2.imread(os.path.join(
                    rgb_path, '{}.jpg'.format(j)), -1)
                cv2.cvtColor(rgb2, cv2.COLOR_BGR2RGB, rgb2)
                rgb2 = rgb2/255.0
                depth2 = cv2.imread(os.path.join(
                    depth_path, '{}.png'.format(j)), -1) / 5000.0
                pose2 = np.loadtxt(os.path.join(
                    pose_path, '{}.txt'.format(j))).reshape(4, 4)
                pcd = back_project_depth(depth2, K)
                pose2to1 = np.matmul(np.linalg.inv(pose1), pose2)
                idx, grid = create_sampling_grid(pcd, pose2to1, K)
                start = time.time()
                corresp, sample = sample_points_zbuffer(
                    depth1, rgb1, idx, grid)
                end = time.time()
                print(end - start)
                num_smaples = np.count_nonzero(corresp)
                print(num_smaples, num_smaples/(640*480))
                ratio = num_smaples / (640 * 480)

                if ratio >= 0.4:
                    item['positives'].append(j)
                    # cv2.imshow('samples', sample)
                    # cv2.waitKey(1)
                elif ratio >= 0.25:
                    item['nonnegatives'].append(j)

            print(item)
            if len(item['positives']) != 0:
                training_sample[count] = item
                count += 1

    print(training_sample)
    with open('training_samples.pickle', 'wb') as f:
        pickle.dump(training_sample, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='./out2')
    gen_training_samples(parser.parse_args())
