# Copyright (C) 2020 Juan Du (Technical University of Munich)
# For more information see <https://vision.in.tum.de/research/vslam/dh3d>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import os
import shutil
import sys

import cv2
import imageio
import numpy as np
import tensorflow as tf
from core.configs import dotdict
from core.datasets import Global_test_dataset
from core.model import DH3D
from core.utils import mkdir_p
from tensorpack.dataflow import BatchData
from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils import get_model_loader
from tfutils.function.imgwarp import get_point_cloud_template

from .evaluation_retrieval import GlobalDesc_eval

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(CURRENT_DIR)))


def get_eval_global_testdata(cfg, data_path, ref_gt_file):
    querybatch = cfg.batch_size
    pos = cfg.num_pos
    neg = cfg.num_neg
    other_neg = 1 if cfg.other_neg else 0
    totalbatch = querybatch * (pos + neg + other_neg + 1)

    df = Global_test_dataset(
        basedir=data_path, test_file=os.path.join(data_path, ref_gt_file))
    df = BatchData(df, totalbatch, remainder=True)

    df.reset_state()
    return df, totalbatch


def get_model_config(Model_Path):
    model_base = os.path.dirname(Model_Path)
    model_config_json = os.path.join(model_base, 'config.json')

    assert os.path.exists(model_config_json)
    with open(model_config_json) as f:
        model_config = dotdict(json.load(f))
    return model_config


# def eval_retrieval(evalargs):
#     # Set up evaluation:
#     save_dir = evalargs.save_dir
#     mkdir_p(evalargs.save_dir)

#     model_configs = get_model_config(evalargs.ModelPath)

#     # Set up graph:
#     pred_config = PredictConfig(
#         model=DH3D(model_configs),
#         session_init=get_model_loader(evalargs.ModelPath),
#         input_names=['pointclouds'],
#         output_names=['globaldesc'],  # ['globaldesc'], output_weights
#     )
#     predictor = OfflinePredictor(pred_config)

#     # Data:
#     df, totalbatch = get_eval_global_testdata(
#         model_configs, evalargs.data_path, evalargs.ref_gt_file)

#     # Predict:
#     pcdnum = 0
#     for [pcds, names] in df:  # pcds is a list, batchsize x numpts x 3
#         batch = pcds.shape[0]
#         if totalbatch > batch:
#             numpts = pcds.shape[1]
#             pcddim = pcds.shape[2]
#             padzeros = np.zeros(
#                 [totalbatch - batch, numpts, pcddim], dtype=np.float32)
#             pcds = np.vstack([pcds, padzeros])
#         results = predictor(pcds)
#         global_feats = results[0]

#         for i in range(batch):
#             pcdnum += 1
#             globaldesc = global_feats[i]
#             name = names[i]
#             savename = os.path.join(evalargs.save_dir, name)
#             basedir = os.path.dirname(savename)
#             mkdir_p(basedir)
#             globaldesc.tofile(savename)

#     print('predicted {} poitnclouds \n'.format(pcdnum))

#     # Evaluation recall:
#     if evalargs.eval_recall:
#         evaluator = GlobalDesc_eval(result_savedir='./', desc_dir=save_dir,
#                                     database_file=os.path.join(
#                                         evalargs.data_path, evalargs.ref_gt_file),
#                                     query_file=os.path.join(
#                                         evalargs.data_path, evalargs.qry_gt_file),
#                                     max_num_nn=25)
#         evaluator.evaluate()
#         print("evaluation finished!\n")

#     if evalargs.delete_tmp:
#         # delete all the descriptors
#         descdirs = [os.path.join(save_dir, f) for f in os.listdir(save_dir)]
#         descdirs = [d for d in descdirs if os.path.isdir(d)]
#         for d in descdirs:
#             shutil.rmtree(d)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--gpu', help='comma separated list of GPU(s) to use.', default='0')
#     parser.add_argument('--save_dir', type=str,
#                         default='./demo_data/res_global')

#     # for evaluation
#     parser.add_argument('--ModelPath', type=str, help='Model to load (for evaluation)',
#                         default='models/global/globalmodel')
#     # parser.add_argument('--data_path', type=str, default="../data/oxford_test_global")
#     parser.add_argument('--data_path', type=str,
#                         default="evaluate/global_eval/demo_data/")
#     parser.add_argument('--ref_gt_file', type=str,
#                         default='global_ref_demo.pickle')
#     parser.add_argument('--qry_gt_file', type=str,
#                         default='global_query_demo.pickle')

#     parser.add_argument('--eval_recall', action='store_true', default=False)
#     parser.add_argument('--delete_tmp', action='store_true', default=False)

#     evalargs = parser.parse_args()
#     eval_retrieval(evalargs)

def downsample_points(pcd):
    pcd = pcd[::5, ::5, :]
    # cv2.imshow('pcd', pcd.numpy()/np.max(pcd.numpy()))
    # cv2.waitKey(0)
    return pcd


def get_eval_data_tum(root_path, batch_size=1):
    def __generator(root_path):
        K = np.array([580, 0, 320, 0, 580, 240, 0, 0, 1],
                     np.float32).reshape((3, 3))
        pcd_temp = get_point_cloud_template(640, 480, K)

        with open(os.path.join(root_path, 'associated.txt'), 'r') as f:
            depth_filenames = np.array([q.strip().split(' ')
                                        for q in f.readlines()])[::10, 3]

        for depth_filename in depth_filenames:
            depth = imageio.imread(os.path.join(
                root_path, depth_filename)) / 5000.0
            depth = tf.reshape(depth, shape=(480, 640, 1))
            depth = tf.cast(depth, dtype=tf.float32)
            pcd = pcd_temp * depth
            pcd = downsample_points(pcd)
            pcd = tf.reshape(pcd, shape=(pcd.shape[0]*pcd.shape[1], 3))
            # tf.print(pcd.shape)
            pcd = pcd[pcd[:, 2] != 0]
            print(pcd.shape)
            yield pcd[:8192, :]
            # pcd = tf.reshape(pcd, shape=(640*480, 3))
            # yield pcd[pcd != (0, 0, 0)]

    dataset = tf.data.Dataset.from_generator(
        lambda: __generator(root_path), output_types=(tf.float32))
    return dataset.batch(batch_size)


def eval_tum_rgbd(evalargs):
    # Set up evaluation:
    save_dir = evalargs.save_dir
    mkdir_p(evalargs.save_dir)

    model_configs = get_model_config(evalargs.ModelPath)

    # Set up graph:
    pred_config = PredictConfig(
        model=DH3D(model_configs),
        session_init=get_model_loader(evalargs.ModelPath),
        input_names=['pointclouds'],
        output_names=['globaldesc'],  # ['globaldesc'], output_weights
    )
    predictor = OfflinePredictor(pred_config)

    totalbatch = 24
    df = get_eval_data_tum(
        '/home/xingrui/Downloads/rgbd_dataset_freiburg3_long_office_household', totalbatch)
    features = []
    for pcds in df:
        batch = pcds.shape[0]
        if totalbatch > batch:
            numpts = pcds.shape[1]
            pcddim = pcds.shape[2]
            padzeros = np.zeros(
                [totalbatch - batch, numpts, pcddim], dtype=np.float32)
            pcds = tf.convert_to_tensor(np.vstack([pcds, padzeros]))
        results = predictor(pcds.numpy())
        global_feats = results[0]
        features += list(global_feats)
    features = np.array(features)
    print(features.shape)
    np.savetxt('features.txt', features)

    # Data:
    # df, totalbatch = get_eval_global_testdata(
    #     model_configs, evalargs.data_path, evalargs.ref_gt_file)

    # Predict:
    # pcdnum = 0
    # for [pcds, names] in df:  # pcds is a list, batchsize x numpts x 3
    #     batch = pcds.shape[0]
    #     if totalbatch > batch:
    #         numpts = pcds.shape[1]
    #         pcddim = pcds.shape[2]
    #         padzeros = np.zeros(
    #             [totalbatch - batch, numpts, b], dtype=np.float32)
    #         pcds = np.vstack([pcds, padzeros])
    #     results = predictor(pcds)
    #     global_feats = results[0]

    #     for i in range(batch):
    #         pcdnum += 1
    #         globaldesc = global_feats[i]
    #         name = names[i]
    #         savename = os.path.join(evalargs.save_dir, name)
    #         basedir = os.path.dirname(savename)
    #         mkdir_p(basedir)
    #         globaldesc.tofile(savename)

    # print('predicted {} poitnclouds \n'.format(pcdnum))


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print('no gpu device found!')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', help='comma separated list of GPU(s) to use.', default='0')
    parser.add_argument('--save_dir', type=str,
                        default='./demo_data/res_global')

    # for evaluation
    parser.add_argument('--ModelPath', type=str, help='Model to load (for evaluation)',
                        default='models/global/globalmodel')
    # parser.add_argument('--data_path', type=str, default="../data/oxford_test_global")
    parser.add_argument('--data_path', type=str,
                        default="evaluate/global_eval/demo_data/")
    parser.add_argument('--ref_gt_file', type=str,
                        default='global_ref_demo.pickle')
    parser.add_argument('--qry_gt_file', type=str,
                        default='global_query_demo.pickle')

    parser.add_argument('--eval_recall', action='store_true', default=False)
    parser.add_argument('--delete_tmp', action='store_true', default=False)

    evalargs = parser.parse_args()
    eval_tum_rgbd(evalargs)
