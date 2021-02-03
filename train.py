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
import tensorflow as tf
from tensorpack import *
from core.datasets import *
from core.model import DH3D
from core.configs import ConfigFactory
from core.utils import log_config_info


def get_data(cfg={}):
    if cfg.training_local:
        return get_train_local_selfpair(cfg)
    else:
        return get_train_global_triplet(cfg)


def get_config(model, config):

    callbacks = [
        PeriodicTrigger(ModelSaver(max_to_keep=100),
                        every_k_steps=config.savemodel_every_k_steps),
        ModelSaver(),
    ]

    train_configs = TrainConfig(
        model=model(config),
        dataflow=get_data(cfg=config),
        callbacks=callbacks,
        extra_callbacks=[
            MovingAverageSummary(),
            MergeAllSummaries(),
            ProgressBar(['total_cost']),
            RunUpdateOps()
        ],
        max_epoch=config.max_epochs
    )
    if config.loadpath is not None:
        train_configs.session_init = SmartInit(
            configs.loadpath, ignore_mismatch=True)

    return train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', help='comma separated list of GPU(s) to use.', type=str, default='0')
    parser.add_argument(
        '--logdir', help='log directory', default='logs')
    parser.add_argument(
        '--logact', type=str, help='action to log directory', default='k')
    parser.add_argument(
        '--cfg', type=str, default='basic_config')
    parser.add_argument('--load', type=int, default=-1)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        print('no gpu device found!')

    configs = ConfigFactory(args.cfg).getconfig()

    if args.load >= 0:
        configs.loadpath = 'logs/model-{}'.format(args.load)

    logger.set_logger_dir(args.logdir, action=args.logact)
    log_config_info(configs)

    train_configs = get_config(DH3D, configs)
    launch_train_with_config(train_configs, SimpleTrainer())
