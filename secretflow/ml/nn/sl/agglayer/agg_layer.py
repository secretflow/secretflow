# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregation Layer for SLModel

"""
from typing import Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

import secretflow as sf
from secretflow.device import HEU, PYU, SPU, DeviceObject, PYUObject
from secretflow.ml.nn.sl.agglayer.agg_method import AggMethod
from secretflow.utils.communicate import ForwardData
from secretflow.utils.compressor import Compressor, MixedCompressor, SparseCompressor
from secretflow.utils.errors import InvalidArgumentError

COMPRESS_DEVICE_LIST = (PYU,)


class AggLayer(object):
    """
    The aggregation layer is situated between Basenet and Fusenet and is responsible for feature fusion, communication compression, and other intermediate layer logic.
    Attributes:
        device_agg: The party do aggregation,it can be a PYU,SPU,etc.
        parties: List of all parties.
        device_y: The party which has fusenet
        agg_method: Aggregation method must inherit from agg_method.AggMethod
        backend: tensorflow or torch
        compressor: Define strategy tensor compression algorithms to speed up transmission.

    """

    def __init__(
        self,
        device_agg: Union[PYU, SPU, HEU],
        parties: List[PYU],
        device_y: PYU,
        agg_method: AggMethod = None,
        backend: str = "tensorflow",
        compressor: Compressor = None,
    ):
        assert isinstance(
            device_agg, (PYU, SPU, HEU)
        ), f'Accepts device in [PYU,SPU,HEU]  but got {type(device_agg)}.'
        if not agg_method and device_agg != device_y:
            raise InvalidArgumentError(f"Default mode, device_agg must to be device_y")
        self.device_agg = device_agg
        self.parties = parties
        self.device_y = device_y
        self.server_data = None
        self.agg_method = agg_method
        self.backend = backend.lower()
        self.compressor = compressor
        self.basenet_output_num = None
        self.hiddens = None
        self.losses = None
        self.fuse_sparse_masks = None if self.compressor is None else []
        self.is_compressed = None if self.compressor is None else []

    def get_parties(self):
        return self.parties

    def set_basenet_output_num(self, basenet_output_num: int):
        self.basenet_output_num = basenet_output_num

    @staticmethod
    def convert_to_ndarray(
        *data: List, backend: str
    ) -> Union[List[jnp.ndarray], jnp.ndarray]:
        def _convert_to_ndarray(hidden):
            # processing data
            if not isinstance(hidden, jnp.ndarray):
                if backend == "torch":
                    hidden = jnp.array(hidden.detach().numpy())
                elif backend == "tensorflow":
                    hidden = jnp.array(hidden.numpy())
                if isinstance(hidden, np.ndarray):
                    hidden = jnp.array(hidden)
            return hidden

        if isinstance(data, Tuple) and len(data) == 1:
            # The case is after packing and unpacking using PYU, a tuple of length 1 will be obtained, if 'num_return' is not specified to PYU.
            data = data[0]
        if isinstance(data, (List, Tuple)):
            return [_convert_to_ndarray(d) for d in data]
        else:
            return _convert_to_ndarray(data)

    @staticmethod
    def convert_to_tensor(hidden: Union[List, Tuple], backend: str):
        if hidden is None:
            return None
        if backend == "tensorflow":
            import tensorflow as tf

            if isinstance(hidden, (List, Tuple)):
                hidden = [tf.convert_to_tensor(d) for d in hidden]
            else:
                hidden = tf.convert_to_tensor(hidden)
        elif backend == "torch":
            import torch

            if isinstance(hidden, (List, Tuple)):
                hidden = [torch.Tensor(d.tolist()) for d in hidden]
            else:
                hidden = torch.Tensor(hidden.tolist())
        else:
            raise InvalidArgumentError(
                f"Invalid backend, only support 'tensorflow' or 'torch', but got {backend}"
            )
        return hidden

    @staticmethod
    def get_hiddens(f_data):
        if isinstance(f_data, (Tuple, List)):
            if isinstance(f_data[0], ForwardData):
                return [d.hidden for d in f_data]
            else:
                return [d for d in f_data]
        else:
            if isinstance(f_data, ForwardData):
                return f_data.hidden
            else:
                return f_data

    @staticmethod
    def get_reg_loss(f_data: ForwardData):
        if isinstance(f_data, (Tuple, List)):
            if isinstance(f_data[0], ForwardData):
                return [d.losses for d in f_data]
            else:
                return None
        else:
            if isinstance(f_data, ForwardData):
                return f_data.losses
            else:
                return None

    @staticmethod
    def set_forward_data(
        hidden,
        losses,
    ):
        return (
            ForwardData(
                hidden=hidden,
                losses=losses,
            )
            if hidden is not None
            else None
        )

    @staticmethod
    def do_compress(
        data, compressor, backend, fuse_sparse_masks=None, iscompressed=None
    ):
        """compress data"""
        compute_data = data.hidden if isinstance(data, ForwardData) else data
        if compute_data is None:
            return None
        working_data = (
            compute_data if isinstance(compute_data, list) else [compute_data]
        )
        working_data = AggLayer.convert_to_ndarray(working_data, backend=backend)
        if iscompressed is None:
            # if not set which to compress, then all data need to be compressed.
            iscompressed = [True] * len(working_data)
        # when using sparse compressor, we need the sparse mask to compress to avoid compress again..
        # the mask is like [None, None, mask1, mask2] if 1,2 data is not compressed.
        if (
            isinstance(compressor, (SparseCompressor, MixedCompressor))
            and fuse_sparse_masks is not None
        ):
            assert len(fuse_sparse_masks) == len(
                data
            ), f'length of fuse_sparse_masks and gradient mismatch: {len(fuse_sparse_masks)} - {len(data)}'
            working_data = list(
                map(
                    lambda d, mask, compressed: compressor.compress(d, sparse_mask=mask)
                    if compressed
                    else d,
                    working_data,
                    fuse_sparse_masks,
                    iscompressed,
                )
            )
        else:
            working_data = list(
                map(
                    lambda d, compressed: compressor.compress(d) if compressed else d,
                    working_data,
                    iscompressed,
                )
            )
        working_data = (
            working_data if isinstance(compute_data, list) else working_data[0]
        )
        if isinstance(data, ForwardData):
            data.hidden = working_data
        else:
            data = working_data
        return data

    @staticmethod
    def do_decompress(
        data: Optional[Union[ForwardData, 'torch.Tensor', 'tf.Tensor']],
        compressor: Compressor,
        backend: str,
        fuse_sparse_masks,
        is_compressed,
    ):
        """
        Decompress the data by the provided compressor.
        Args:
            data: The compressed data.
            compressor: the compressor to decompress hidden.
            backend: backend.
            fuse_sparse_masks: A list of the sparse compressed masks if needed.
            is_compressed: A list of is compressed flags.
        """
        if data is None:
            return None, fuse_sparse_masks, is_compressed
        compute_data = data.hidden if isinstance(data, ForwardData) else data
        working_data = (
            compute_data if isinstance(compute_data, list) else [compute_data]
        )
        is_compress: list = compressor.iscompressed(working_data)
        # is_compress must be all True or all False, since they came from same data.
        assert all(is_compressed) or not any(is_compressed)
        fuse_sparse_mask = [None] * len(working_data)
        if all(is_compress):
            if isinstance(compressor, (SparseCompressor, MixedCompressor)):
                fuse_sparse_mask = [wd.get_sparse_mask() for wd in working_data]
            working_data = compressor.decompress(working_data)
        working_data = AggLayer.convert_to_tensor(working_data, backend)
        fuse_sparse_masks += fuse_sparse_mask
        is_compressed += is_compress
        working_data = (
            working_data if isinstance(compute_data, list) else working_data[0]
        )
        if isinstance(data, ForwardData):
            data.hidden = working_data
        else:
            data = working_data

        return data, fuse_sparse_masks, is_compressed

    @staticmethod
    def split_to_parties(
        data: Union[List, "torch.Tensor"], basenet_output_num, parties
    ) -> List[PYUObject]:
        assert (
            basenet_output_num is not None
        ), "Agglayer should know output num of each participates"
        if sum(basenet_output_num.values()) == 1:
            return data
        else:
            assert len(data) == sum(
                basenet_output_num.values()
            ), f"data length in backward = {len(data)} is not consistent with basenet need = {sum(basenet_output_num.values())},"

            result = []
            start_idx = 0
            for p in parties:
                data_slice = data[start_idx : start_idx + basenet_output_num[p]]
                result.append(data_slice)
                start_idx = start_idx + basenet_output_num[p]
            return result

    def collect(self, data: Dict[PYU, DeviceObject]) -> List[DeviceObject]:
        """Collect data from participates
        TODO: Support compress communication when using agg method.
        """
        assert data, 'Data to aggregate should not be None or empty!'

        # Record the values of fields in ForwardData except for hidden
        self.losses = []

        coverted_data = []
        for device, f_datum in data.items():
            if device not in self.parties:
                continue
            hidden = device(self.get_hiddens)(f_datum)
            loss = device(self.get_reg_loss)(f_datum)
            # transfer other fields to device_y
            self.losses.append(loss.to(self.device_y))

            # aggregate hiddens on device_agg, then push to device_y
            hidden = device(self.convert_to_ndarray)(hidden, backend=self.backend)
            # do compress before send to device agg
            if isinstance(self.device_agg, COMPRESS_DEVICE_LIST) and self.compressor:
                hidden = device(self.compressor.compress)(hidden)
            coverted_data.append(hidden)
        # do transfer
        server_data = [d.to(self.device_agg) for d in coverted_data]

        # do decompress after recieve data from each parties
        if isinstance(self.device_agg, COMPRESS_DEVICE_LIST) and self.compressor:
            server_data = [
                self.device_agg(
                    lambda compressor, d: compressor.decompress(d)
                    if compressor.iscompressed(d)
                    else d
                )(self.compressor, d)
                for d in server_data
            ]
            return server_data
        return server_data

    @staticmethod
    def parse_gradients(gradients):
        if isinstance(gradients, List) and len(gradients) == 1:
            return gradients[0]
        else:
            return gradients

    def scatter(self, data: Union[List, PYUObject]) -> Dict[PYU, DeviceObject]:
        """Send ForwardData to participates"""
        # do compress before send to participates
        # FIXME: The return of PYU proxy are inconsistent when "num_return==1" and ">1"
        if isinstance(data, PYUObject):
            data = self.device_agg(self.parse_gradients)(data)
            data = [data]
        if isinstance(self.device_agg, COMPRESS_DEVICE_LIST) and self.compressor:
            data = [self.device_agg(self.compressor.compress)(datum) for datum in data]
        # send
        result = {}

        for p, d in zip(self.parties, data):
            datum = d.to(p)
            # do decompress after recieve from device agg
            if isinstance(self.device_agg, COMPRESS_DEVICE_LIST) and self.compressor:
                datum = p(self.compressor.decompress)(datum)
            # convert to tensor
            datum = p(self.convert_to_tensor)(datum, self.backend)
            result[p] = datum
        return result

    def forward(
        self,
        data: Dict[PYU, DeviceObject],
    ) -> list:
        """Forward aggregate the embeddings calculated by all parties according to the agg_method

        Args:
            data: A dict contain PYU and ForwardData
        Returns:
            agg_data_tensor: return aggregated result in tensor type
        """
        assert data, 'Data to aggregate should not be None or empty!'
        if self.is_compressed is not None:
            # forward() will be called multi-times, the masks and is_compressed need to be reset each time.
            self.fuse_sparse_masks = []
            self.is_compressed = []
        if self.agg_method:
            server_data = self.collect(data)
            self.hiddens = server_data
            # agg hiddens
            agg_hiddens = self.device_agg(
                self.agg_method.forward,
            )(*server_data)

            # send to device y
            agg_hiddens = agg_hiddens.to(self.device_y)

            # TODO: Supports sparse calculation and add compression from device_agg to device_y. #juxing
            # if self.compressor:
            #     agg_hiddens, fuse_sparse_masks, is_compressed = self.device_y(
            #         self.decompress_hiddens,
            #         num_returns=3,
            #     )([agg_hiddens], self.compressor)
            #     self.fuse_sparse_masks = fuse_sparse_masks
            #     self.is_compressed = is_compressed

            # convert to tensor on device y
            agg_hidden_tensor = self.device_y(self.convert_to_tensor)(
                agg_hiddens, self.backend
            )

            # make new ForwardData and return
            agg_forward_data = self.device_y(self.set_forward_data)(
                agg_hidden_tensor, self.losses
            )

            return agg_forward_data
        else:
            compute_data = []
            for device in data:
                working_data = data[device]
                if self.compressor:
                    if device != self.device_y:
                        working_data = device(self.do_compress)(
                            working_data, self.compressor, self.backend
                        ).to(self.device_y)
                    (
                        working_data,
                        self.fuse_sparse_masks,
                        self.is_compressed,
                    ) = self.device_y(self.do_decompress, num_returns=3)(
                        working_data,
                        self.compressor,
                        self.backend,
                        self.fuse_sparse_masks,
                        self.is_compressed,
                    )
                else:
                    working_data = working_data.to(self.device_y)
                compute_data.append(working_data)
            return compute_data

    def backward(
        self,
        gradient: DeviceObject,
    ) -> Dict[PYU, DeviceObject]:
        """Backward split the gradients to all parties according to the agg_method

        Args:
            gradient: Gradient, tensor format calculated from fusenet
        Returns:
            scatter_gragient: Return gradients computed following the agg_method.backward and send to each parties
        """
        assert gradient, 'gradient to aggregate should not be None or empty!'
        if self.agg_method:
            # TODO: Supports sparse calculation and add compression from device_y to device_agg. #juxing
            # if self.compressor:
            #     gradient = self.device_y(self.compress_gradients)(
            #         gradient,
            #         self.fuse_sparse_masks,
            #         self.compressor,
            #         self.is_compressed,
            #     )
            gradient_len = len(gradient) if isinstance(gradient, (list, tuple)) else 1
            gradient_numpy = self.device_y(
                self.convert_to_ndarray, num_returns=gradient_len
            )(gradient, backend=self.backend)
            if gradient_len == 1:
                gradient_numpy = [gradient_numpy]

            gradient_numpy = [gn.to(self.device_agg) for gn in gradient_numpy]

            if isinstance(self.device_agg, SPU):
                # do agg layer backward
                p_gradient = self.device_agg(
                    self.agg_method.backward,
                    static_argnames='parties_num',
                    num_returns_policy=sf.device.SPUCompilerNumReturnsPolicy.FROM_USER,
                    user_specified_num_returns=len(self.parties),
                )(
                    *gradient_numpy,
                    inputs=self.hiddens,
                    parties_num=len(self.parties),
                )
            else:
                p_gradient = self.device_agg(
                    self.agg_method.backward,
                    num_returns=len(self.parties),
                )(
                    *gradient_numpy,
                    inputs=self.hiddens,
                    parties_num=len(self.parties),
                )
            scatter_g = self.scatter(p_gradient)
        else:
            # default branch, input gradients is from fusenet, belong to device_y
            assert (
                gradient.device == self.device_y
            ), "The device of gradients(PYUObject) must located on party device_y "
            if self.compressor:
                # Compress if needed (same device will be passed)
                gradient = self.device_y(self.do_compress)(
                    gradient,
                    self.compressor,
                    self.backend,
                    self.fuse_sparse_masks,
                    self.is_compressed,
                )
            # split gradients to parties by index
            # TODO: In GPU mode, specifying num_gpus is required for executing remote functions.
            #  However, even if you specify the GPU, it can still result in serious performance issues.
            p_gradient = self.device_y(
                self.split_to_parties,
                num_returns=len(self.parties),
            )(gradient, self.basenet_output_num, self.parties)
            # handle single feature mode
            if isinstance(p_gradient, PYUObject):
                p_gradient = self.device_y(self.parse_gradients)(p_gradient)
                p_gradient = [p_gradient]

            scatter_g = {}
            for device, gradient in zip(self.parties, p_gradient):
                if device != self.device_y:
                    gradient = gradient.to(device)
                if self.compressor and device != self.device_y:
                    gradient = device(self.compressor.decompress)(gradient)
                scatter_g[device] = gradient
        return scatter_g
