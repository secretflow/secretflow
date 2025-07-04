# Copyright 2024 Ant Group Co., Ltd.
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

import numpy as np
import tensorflow as tf

import secretflow as sf
from secretflow.utils.communicate import ForwardData
from secretflow_fl.ml.nn.sl.agglayer.agg_layer import AggLayer
from secretflow_fl.ml.nn.sl.agglayer.agg_method import Average, Concat, Sum


class TestPlainAggLayer:
    def test_plain_average(self, sf_simulation_setup_devices):
        devices = sf_simulation_setup_devices
        # SETUP DEVICE
        alice, bob, carol = (devices.alice, devices.bob, devices.carol)
        # GIVEN
        average_agglayer = AggLayer(
            device_agg=carol,
            parties=[alice, bob],
            device_y=bob,
            agg_method=Average(axis=0),
        )
        a = alice(
            lambda: ForwardData(
                hidden=tf.constant(
                    ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]])
                )
            )
        )()
        b = bob(
            lambda: ForwardData(
                hidden=tf.constant(
                    ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]])
                )
            )
        )()

        c = bob(
            lambda: tf.constant(
                ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]])
            )
        )()

        # WHEN
        forward_obj = sf.reveal(average_agglayer.forward({alice: a, bob: b}))
        backward_obj = sf.reveal(average_agglayer.backward(c))
        np.testing.assert_almost_equal(
            forward_obj.hidden.numpy(),
            tf.constant(
                ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]])
            ).numpy(),
            decimal=5,
        )

        np.testing.assert_almost_equal(
            backward_obj[alice].numpy(),
            tf.constant(
                ([[0.5, 1.0], [1.5, 2.0]], [[5.0, 10.0], [15.0, 20.0]])
            ).numpy(),
            decimal=5,
        )

        np.testing.assert_almost_equal(
            backward_obj[bob].numpy(),
            tf.constant(
                ([[0.5, 1.0], [1.5, 2.0]], [[5.0, 10.0], [15.0, 20.0]])
            ).numpy(),
            decimal=5,
        )

    def test_plain_default(self, sf_simulation_setup_devices):
        devices = sf_simulation_setup_devices
        # SETUP DEVICE
        alice, bob = (devices.alice, devices.bob)
        # GIVEN
        default_agglayer = AggLayer(
            device_agg=bob,
            parties=[alice, bob],
            device_y=bob,
            agg_method=None,
        )
        default_agglayer.set_basenet_output_num({alice: 2, bob: 2})
        a = alice(
            lambda: ForwardData(
                hidden=tf.constant(
                    ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]])
                )
            )
        )()
        b = bob(
            lambda: ForwardData(
                hidden=tf.constant(
                    ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]])
                )
            )
        )()
        c = bob(
            lambda: tf.constant(
                (
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[10.0, 20.0], [30.0, 40.0]],
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[10.0, 20.0], [30.0, 40.0]],
                )
            )
        )()

        # WHEN
        forward_obj = sf.reveal(default_agglayer.forward({alice: a, bob: b}))
        backward_obj = sf.reveal(default_agglayer.backward(c))

        np.testing.assert_almost_equal(
            forward_obj[0].hidden.numpy(),
            tf.constant(
                ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]])
            ).numpy(),
            decimal=5,
        )
        np.testing.assert_almost_equal(
            forward_obj[1].hidden.numpy(),
            tf.constant(
                ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]])
            ).numpy(),
            decimal=5,
        )

        np.testing.assert_almost_equal(
            backward_obj[alice].numpy(),
            tf.constant(
                ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]])
            ).numpy(),
            decimal=5,
        )
        np.testing.assert_almost_equal(
            backward_obj[bob].numpy(),
            tf.constant(
                ([[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]])
            ).numpy(),
            decimal=5,
        )


class TestSPUAggLayer:
    def test_spu_average(self, sf_simulation_setup_devices):
        devices = sf_simulation_setup_devices
        # SETUP DEVICE
        alice, bob = (devices.alice, devices.bob)
        spu = devices.spu

        # GIVEN

        spu_agglayer = AggLayer(
            device_agg=spu,
            parties=[alice, bob],
            device_y=bob,
            agg_method=Average(axis=0),
        )
        a = alice(lambda: ForwardData(hidden=tf.constant([[1.0, 2.0], [3.0, 4.0]])))()
        b = bob(lambda: ForwardData(hidden=tf.constant([[1.0, 2.0], [3.0, 4.0]])))()
        c = bob(lambda: tf.constant([[1.0, 2.0], [3.0, 4.0]]))()
        # WHEN
        forward_obj = sf.reveal(spu_agglayer.forward({alice: a, bob: b}))
        backward_obj = sf.reveal(spu_agglayer.backward(c))
        # THEN
        np.testing.assert_almost_equal(
            forward_obj.hidden.numpy(),
            tf.constant([[1.0, 2.0], [3.0, 4.0]]).numpy(),
            decimal=5,
        )
        np.testing.assert_almost_equal(
            backward_obj[alice].numpy(),
            tf.constant([[0.5, 1.0], [1.5, 2.0]]).numpy(),
            decimal=5,
        )
        np.testing.assert_almost_equal(
            backward_obj[bob].numpy(),
            tf.constant([[0.5, 1.0], [1.5, 2.0]]).numpy(),
            decimal=5,
        )


class TestConcatAggLayer:
    def test_spu_concat(self, sf_simulation_setup_devices):
        devices = sf_simulation_setup_devices
        # SETUP DEVICE
        alice, bob = (devices.alice, devices.bob)
        spu = devices.spu

        spu_agglayer = AggLayer(
            device_agg=spu,
            parties=[alice, bob],
            device_y=bob,
            agg_method=Concat(axis=1),
        )
        a = alice(
            lambda: ForwardData(hidden=tf.constant([[1.0, 2.0, 5.0], [3.0, 4.0, 5.0]]))
        )()
        b = bob(lambda: ForwardData(hidden=tf.constant([[1.0, 2.0], [3.0, 4.0]])))()
        c = bob(
            lambda: tf.constant([[1.0, 2.0, 5.0, 1.0, 2.0], [3.0, 4.0, 5.0, 3.0, 4.0]])
        )()
        # WHEN
        forward_obj = sf.reveal(spu_agglayer.forward({alice: a, bob: b}))
        backward_obj = sf.reveal(spu_agglayer.backward(c))

        # THEN
        np.testing.assert_almost_equal(
            forward_obj.hidden.numpy(),
            tf.constant([[1.0, 2.0, 5.0, 1.0, 2.0], [3.0, 4.0, 5.0, 3.0, 4.0]]).numpy(),
            decimal=5,
        )
        np.testing.assert_almost_equal(
            backward_obj[alice].numpy(),
            tf.constant([[1.0, 2.0, 5.0], [3.0, 4.0, 5.0]]).numpy(),
            decimal=5,
        )
        np.testing.assert_almost_equal(
            backward_obj[bob].numpy(),
            tf.constant([[1.0, 2.0], [3.0, 4.0]]).numpy(),
            decimal=5,
        )


class TestSumAggLayer:
    def test_spu_sum(self, sf_simulation_setup_devices):
        devices = sf_simulation_setup_devices
        # SETUP DEVICE
        alice, bob = (devices.alice, devices.bob)
        spu = devices.spu

        spu_agglayer = AggLayer(
            device_agg=spu,
            parties=[alice, bob],
            device_y=bob,
            agg_method=Sum(axis=0),
        )
        a = alice(lambda: ForwardData(hidden=tf.constant([[1.0, 2.0], [3.0, 4.0]])))()
        b = bob(lambda: ForwardData(hidden=tf.constant([[1.0, 2.0], [3.0, 4.0]])))()
        c = bob(lambda: tf.constant([[2.0, 4.0], [6.0, 8.0]]))()
        # WHEN
        forward_obj = sf.reveal(spu_agglayer.forward({alice: a, bob: b}))
        backward_obj = sf.reveal(spu_agglayer.backward(c))

        # THEN
        np.testing.assert_almost_equal(
            forward_obj.hidden.numpy(),
            tf.constant([[2.0, 4.0], [6.0, 8.0]]).numpy(),
            decimal=5,
        )
        np.testing.assert_almost_equal(
            backward_obj[alice].numpy(),
            tf.constant([[2.0, 4.0], [6.0, 8.0]]).numpy(),
            decimal=5,
        )
        np.testing.assert_almost_equal(
            backward_obj[bob].numpy(),
            tf.constant([[2.0, 4.0], [6.0, 8.0]]).numpy(),
            decimal=5,
        )
