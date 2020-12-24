import numpy as np
import os
import imageio
import random
import cv2
import matplotlib.pyplot as plt

root_path = '/home/xingrui/Downloads/rgbd_dataset_freiburg3_long_office_household'
valid_path = '/home/xingrui/Downloads/rgbd_dataset_freiburg3_long_office_household_validation'


def find_knn(src, dst, k=4):
    dist = np.linalg.norm(src-dst, axis=-1)
    dist_idx = np.argsort(dist)
    return dist_idx[:k]


if __name__ == '__main__':

    with open(os.path.join(root_path, 'associated.txt'), 'r') as f:
        root_files = np.array([q.strip().split(' ')
                               for q in f.readlines()])[::10, 1]
    with open(os.path.join(valid_path, 'associated.txt'), 'r') as f:
        valid_files = np.array([q.strip().split(' ')
                                for q in f.readlines()])[::10, 1]

    features = np.loadtxt('features.txt')
    features_valid = np.loadtxt('features_valid.txt')
    final = []
    anchors = []
    candidates = list(range(len(valid_files)))
    for _ in range(6):
        anchors += [random.choice(candidates)]
    # print(candidates)
    # anchors = [0, 55, 120, 150, 233]

    for anchor in anchors:
        knn_idx = find_knn(features_valid[anchor, :], features, 4)
        anchor_img = imageio.imread(
            os.path.join(valid_path, valid_files[anchor]))
        anchor_img = cv2.resize(anchor_img, (160, 120))
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(anchor_img, '{}'.format(anchor), (10, 110), font, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)

        out = anchor_img

        for idx in knn_idx:
            ret_img = imageio.imread(os.path.join(root_path, root_files[idx]))
            ret_img = cv2.resize(ret_img, (160, 120))
            ret_img = cv2.cvtColor(ret_img, cv2.COLOR_RGB2BGR)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(ret_img, '{}'.format(idx), (10, 110), font, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            out = np.concatenate((out, ret_img), axis=1)

        if final == []:
            final = out
        else:
            final = np.concatenate((final, out), axis=0)

    cv2.imshow('out', final)
    cv2.waitKey(0)
    # array = []
    # ret = -1
    # second_ret = -1
    # min_dist = 100
    # second_min_dist = 100
    # for i in range(features_valid.shape[0]):
    #     if i != anchor:
    #         anchor_feat = features_valid[anchor, :]
    #         target_feat = features[i, :]
    #         dist = np.linalg.norm(anchor_feat-target_feat)
    #         if dist < min_dist:
    #             min_dist = dist
    #             ret = i
    #         elif dist < second_min_dist:
    #             second_min_dist = dist
    #             second_ret = i
    #         array += [dist]

    # print('target found: {}'.format(ret))
    # print('candidate target found: {}'.format(second_ret))
    # anchor_img = imageio.imread(os.path.join(args.path, valid_files[anchor]))
    # ret_img = imageio.imread(os.path.join(args.path, root_files[ret]))
    # ret_img2 = imageio.imread(os.path.join(
    #     args.path, root_files[second_ret]))
    # anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_RGB2BGR)
    # ret_img = cv2.cvtColor(ret_img, cv2.COLOR_RGB2BGR)
    # ret_img2 = cv2.cvtColor(ret_img2, cv2.COLOR_RGB2BGR)

    # anchor_img = cv2.resize(anchor_img, (160, 120))
    # ret_img = cv2.resize(ret_img, (160, 120))
    # ret_img2 = cv2.resize(ret_img2, (160, 120))

    # out = np.concatenate((anchor_img, ret_img, ret_img2), axis=1)
    # cv2.imshow('out', out)
    # cv2.waitKey(0)

    # cv2.imshow('anchor', anchor_img)
    # cv2.imshow('ret', ret_img)
    # cv2.imshow('ret2', ret_img2)
    # cv2.waitKey(0)

    # plt.plot(array)
    # plt.plot(anchor, 0, 'r+')
    # plt.plot(ret, 0, 'b+')
    # plt.plot(second_ret, 0, 'g+')
    # plt.show()
