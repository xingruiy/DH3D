#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017 ComputerGraphics Tuebingen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch


import numpy as np
import tensorflow as tf

from __init__ import flex_pooling
from misc import FakePointCloud, VerboseTestCase

case = FakePointCloud(B=2, N=32, K=4, Din=2, Dout=6, Dp=3)


class FlexPoolTest(VerboseTestCase):
    def __init__(self, methodName="runTest"):
        super(FlexPoolTest, self).__init__(methodName)

    def _forward(self, use_gpu=False, dtype=np.float32):
        with tf.device('/gpu:0') if use_gpu else tf.device('/cpu:0'):
            case.init_ops(dtype=dtype)
            actual_op, _ = flex_pooling(case.features_op, case.neighborhood_op)
            return actual_op

    def test_forward_same_float32(self):
        cpu = self._forward(use_gpu=False, dtype=np.float32)
        gpu = self._forward(use_gpu=True, dtype=np.float32)
        self.assertAllClose(cpu, gpu)

    def test_forward_same_float64(self):
        cpu = self._forward(use_gpu=False, dtype=np.float64)
        gpu = self._forward(use_gpu=True, dtype=np.float64)
        self.assertAllClose(cpu, gpu)

    def _backward(self, use_gpu=False, dtype=np.float32):
        with tf.device('/gpu:0') if use_gpu else tf.device('/cpu:0'):
            case.init_ops(dtype=dtype)
            with tf.GradientTape() as tape:
                tape.watch(case.features_op)
                actual_op, winner_op = flex_pooling(
                    case.features_op,
                    case.neighborhood_op)

                grad = tape.gradient(
                    actual_op, [case.features_op])[0]

            return grad, winner_op

    def test_backward_same_float32(self):
        cpu, winner_cpu = self._backward(use_gpu=False, dtype=np.float32)
        gpu, winner_gpu = self._backward(use_gpu=True, dtype=np.float32)
        self.assertAllClose(winner_cpu, winner_gpu)
        self.assertAllClose(cpu, gpu)

    def test_backward_same_float64(self):
        cpu, winner_cpu = self._backward(use_gpu=False, dtype=np.float64)
        gpu, winner_gpu = self._backward(use_gpu=True, dtype=np.float64)
        self.assertAllClose(winner_cpu, winner_gpu)
        self.assertAllClose(cpu, gpu)

    def _simple_backward(self, use_gpu=False):
        # BN
        with tf.device('/gpu:0') if use_gpu else tf.device('/cpu:0'):
            x = np.array([[[1], [2], [5], [3]]]).transpose(0, 2, 1)
            n = np.array([[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1],
                           [3, 0, 1, 2, ]]]).transpose(0, 2, 1)

            x = tf.convert_to_tensor(x.astype(np.float32))
            n = tf.convert_to_tensor(n.astype(np.int32))

            with tf.GradientTape() as tape:
                tape.watch(x)
                actual_op, _ = flex_pooling(x, n)
                graph_features_grad = tape.gradient(actual_op, [x])[0]
                return graph_features_grad

    def test_backward_simple_cpu(self):
        cpu = self._simple_backward(use_gpu=False)
        self.assertEqual(tf.reduce_sum(cpu), 4)

    def test_backward_simple_gpu(self):
        gpu = self._simple_backward(use_gpu=True)
        self.assertEqual(tf.reduce_sum(gpu), 4)


if __name__ == '__main__':
    tf.test.main()
