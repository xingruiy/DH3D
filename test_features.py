import numpy as np
import os
import imageio
import cv2
import argparse
import matplotlib.pyplot as plt

root_path = '/home/xingrui/Downloads/rgbd_dataset_freiburg3_long_office_household'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('anchor', type=int)
    parser.add_argument('--path', type=str, default=root_path)
    args = parser.parse_args()

    with open(os.path.join(args.path, 'associated.txt'), 'r') as f:
        rgb_filenames = np.array([q.strip().split(' ')
                                  for q in f.readlines()])[::10, 1]

    features = np.loadtxt('features.txt')
    anchor = args.anchor

    array = []
    ret = -1
    second_ret = -1
    min_dist = 100
    second_min_dist = 100
    for i in range(features.shape[0]):
        if i != anchor:
            anchor_feat = features[anchor, :]
            target_feat = features[i, :]
            dist = np.linalg.norm(anchor_feat-target_feat)
            if dist < min_dist:
                min_dist = dist
                ret = i
            elif dist < second_min_dist:
                second_min_dist = dist
                second_ret = i
            array += [dist]

    print('target found: {}'.format(ret))
    print('candidate target found: {}'.format(second_ret))
    anchor_img = imageio.imread(os.path.join(args.path, rgb_filenames[anchor]))
    ret_img = imageio.imread(os.path.join(args.path, rgb_filenames[ret]))
    ret_img2 = imageio.imread(os.path.join(
        args.path, rgb_filenames[second_ret]))
    cv2.imshow('anchor', anchor_img)
    cv2.imshow('ret', ret_img)
    cv2.imshow('ret2', ret_img2)
    cv2.waitKey(0)

    # plt.plot(array)
    # plt.plot(anchor, 0, 'r+')
    # plt.plot(ret, 0, 'b+')
    # plt.plot(second_ret, 0, 'g+')
    # plt.show()
