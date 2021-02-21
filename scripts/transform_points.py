import argparse
import os

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--force', action='store_true')

    args = parser.parse_args()
    if args.force:
        args.out_file = args.in_file

    dirs = [f for f in os.listdir(args.in_file)
            if os.path.isdir(os.path.join(args.in_file, f))]
    for folder in dirs:
        dir_out = os.path.join(args.out_file, folder)
        model_path = os.path.join('data', folder, 'model')
        pose_path = os.path.join('data', folder, 'pose')
        if args.in_file != args.out_file:
            os.mkdir(os.path.join(dir_out))
            os.mkdir(os.path.join(dir_out, 'model'))
            os.mkdir(os.path.join(dir_out, 'pose'))
        num_models = len(os.listdir(model_path))
        for i in range(len(os.listdir(model_path))):
            model_path_out = os.path.join(
                dir_out, 'model', '{}.bin'.format(i))
            pose_path_out = os.path.join(dir_out, 'pose', '{}.txt'.format(i))
            model = np.fromfile(os.path.join(
                model_path, '{}.bin'.format(i)), dtype=np.float32)
            model = model.reshape((model.shape[0]//6, 6))
            pose = np.loadtxt(os.path.join(
                pose_path, '{}.txt'.format(i))).reshape(4, 4)
            rot = pose[:3, :3]
            trans = pose[:3, 3].reshape(3, 1)
            model = np.expand_dims(model, axis=-1)
            model[:, 0:3, :] = np.matmul(rot, model[:, 0:3, :]) + trans
            model = np.squeeze(model)
            model = model.reshape((model.shape[0]*model.shape[1],))
            model.tofile(model_path_out)
