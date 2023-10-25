# Copyright 2022 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import logging

from secretflow.utils.compressor import CompressedData
from secretflow.utils.compressor.base import Compressor


class QuantizedCompressedData(CompressedData):
    def __init__(self, compressed_data, quant_bits: int, origin_type=None):
        super().__init__(compressed_data)
        self.quant_bits = quant_bits
        self.origin_type = origin_type


class QuantizedCompressor(Compressor):
    """Abstract base class for quantized compressor"""

    def __init__(self, quant_bits: int = 8):
        """Initialize

        Args:
            quant_bits: the compressed bits length.
        """
        super().__init__()
        self.quant_bits = quant_bits
        self.np_type = self._infer_np_type(quant_bits)

    def _infer_np_type(self, quant_bits):
        if self.quant_bits <= 0 or self.quant_bits > 64:
            logging.error(
                f"The quantized bits len must be between 0 and 64, got {quant_bits}"
            )
            raise RuntimeError(
                f"The quantized bits len must be between 0 and 64, got {quant_bits}"
            )
        recommend_bits = {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64}
        if quant_bits in recommend_bits:
            return recommend_bits[quant_bits]
        else:
            logging.warning(
                f"It is recommended to use 8/16/32/64 as the compression bits (got {self.quant_bits}),"
                f"That will loss accuracy but not save communication traffic."
            )
            for i in range(64):
                if (quant_bits + i) in recommend_bits:
                    return recommend_bits[(quant_bits + i)]
        raise RuntimeError(f"Unknown input quant_bits {quant_bits}")


class QuantizedZeroPoint(QuantizedCompressor):
    """Quantized compressor with strengthen the QuantizedLSTM with replacing 32-bit RQM to 8-bit zero point.
    The tests show that the QuantizedZeroPoint compressor has higner accuracy than QuantizedLSTM.
    Reference paper 2017 "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference".

    Link: https://arxiv.org/abs/1712.05877
    """

    class ZeroPointCompressData(QuantizedCompressedData):
        def __init__(
            self, compressed_data, quant_bits, origin_type, scale, nudged_zero_point
        ):
            super().__init__(compressed_data, quant_bits, origin_type)
            self.scale = scale
            self.nudged_zero_point = nudged_zero_point

    def __init__(self, quant_bits: int = 8):
        super().__init__(quant_bits)

    def _compress_one(self, data: np.ndarray, **kwargs) -> "ZeroPointCompressData":
        _max = np.max(data)
        _min = np.min(data)
        qmin = -(int)(1 << (self.quant_bits - 1))
        qmax = (int)((1 << (self.quant_bits - 1)) - 1)

        scale = (_max - _min) / (qmax - qmin)
        initial_zero_point = qmin - _min / scale
        nudged_zero_point = initial_zero_point
        if initial_zero_point < qmin:
            nudged_zero_point = qmin
        elif initial_zero_point > qmax:
            nudged_zero_point = qmax
        else:
            nudged_zero_point = int(initial_zero_point)
        transformed_val = data / scale + nudged_zero_point
        clamped_val = np.clip(transformed_val, qmin, qmax)
        quantized = np.round(clamped_val)
        return self.ZeroPointCompressData(
            quantized.astype(self.np_type),
            self.quant_bits,
            data.dtype,
            scale,
            nudged_zero_point,
        )

    def _decompress_one(self, data: "ZeroPointCompressData") -> np.ndarray:
        return (
            data.compressed_data.astype(data.origin_type)
            - float(data.nudged_zero_point)
        ) * data.scale


class QuantizedLSTM(QuantizedCompressor):
    """Quantized compressor with LSTM, a basic algorithm which replace float with int.

    Reference paper 2016 "On the efficient representation and execution of deep acoustic models".

    Link: https://arxiv.org/abs/1607.04683
    """

    class LSTMCompressData(QuantizedCompressedData):
        def __init__(self, compressed_data, quant_bits, origin_type, q, rqm):
            super().__init__(compressed_data, quant_bits, origin_type)
            self.q = q
            self.rqm = rqm

    def __init__(self, quant_bits: int = 8):
        super().__init__(quant_bits)

    def _compress_one(self, data: np.ndarray, **kwargs) -> "LSTMCompressData":
        _max = np.max(data)
        _min = np.min(data)
        q_scale = int((1 << self.quant_bits) - 1)
        q_shift = int(1 << (self.quant_bits - 1))

        q = q_scale / (_max - _min)
        rqm = int(round(q * _min) + q_shift)
        quantized = np.round(q * data) - rqm
        return self.LSTMCompressData(
            quantized.astype(self.np_type), self.quant_bits, data.dtype, q, rqm
        )

    def _decompress_one(self, data: "LSTMCompressData") -> np.ndarray:
        return (
            data.compressed_data.astype(data.origin_type) + float(data.rqm)
        ) / float(data.q)


class QuantizedFP(QuantizedCompressor):
    """Quantized compressor with low-bit floating points, fp16/32/64 will be change directly in numpy format, while fp8 will be stored as int8 object.

    Reference paper "FP8 FORMATS FOR DEEP LEARNING".

    Link: https://arxiv.org/pdf/2209.05433.pdf
    """

    class FPCompressData(QuantizedCompressedData):
        def __init__(self, compressed_data, quant_bits, origin_type, scale=None):
            super().__init__(compressed_data, quant_bits, origin_type)
            self.scale = scale

    def __init__(self, quant_bits: int = 8, format='E4M3'):
        super().__init__(quant_bits)
        if quant_bits not in [8, 16, 32, 64]:
            raise RuntimeError(
                f"The quantized bits for QuantizedFP must in 8/16/32/64, got {quant_bits}"
            )

        if quant_bits == 8 and format not in ['E4M3', 'E5M2']:
            raise RuntimeError(
                f"The format for fp8 quantized must in E4M3/E5M2, got {format}"
            )
        config = {
            'E4M3': {'max_value': 448, 'mant_len': 8, 'exp_offset': 6},
            'E5M2': {'max_value': 57344, 'mant_len': 4, 'exp_offset': 14},
        }
        self.config = config[format]

    def _compress_one(self, data: np.ndarray, **kwargs) -> "FPCompressData":
        if self.quant_bits > 8:
            # fp 16/32/64
            return self.FPCompressData(
                data.astype(getattr(np, f'float{self.quant_bits}')),
                self.quant_bits,
                data.dtype,
            )
        else:
            # fp8 with a scale factor, store as np.int8.
            q_sign = np.sign(data)

            out = np.abs(data)
            max_division = np.max(out) if np.max(out) > 0 else 1
            scale = self.config['max_value'] / max_division
            out = out * scale
            mant, exp = np.frexp(
                out
            )  # frexp mantissa range is (-1, 1), not (-2, -1] and [1, 2)
            q_exp = np.where(
                exp > -self.config['exp_offset'], exp + self.config['exp_offset'], 0
            )
            q_mant = np.round((2 * mant - 1) * self.config['mant_len'])

            quantized = q_sign * (q_exp * self.config['mant_len'] + q_mant)
            return self.FPCompressData(
                quantized.astype(self.np_type), self.quant_bits, data.dtype, scale
            )

    def _decompress_one(self, data: "FPCompressData") -> np.ndarray:
        if self.quant_bits != 8:
            return data.compressed_data.astype(data.origin_type)
        else:
            # decompose fp8(stored in int8) to default fp
            quantized = data.compressed_data
            sign = np.sign(quantized)
            abs_quantized = np.abs(quantized)
            exp = (abs_quantized // self.config['mant_len']) - self.config['exp_offset']
            mant = (
                (abs_quantized % self.config['mant_len']).astype(data.origin_type)
                / self.config['mant_len']
                + 1
            ) / 2

            ori_data = sign * np.ldexp(mant, exp) / data.scale
            return ori_data


class QuantizedKmeans(QuantizedCompressor):
    """Quantized compressor with Kmeans, a algorithm which replace float with relatived centroid's index.

    Reference paper 2016 "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding".

    Link: https://arxiv.org/abs/1510.00149
    """

    class KmeansCompressData(QuantizedCompressedData):
        def __init__(self, compressed_data, quant_bits, origin_type=None, q=None):
            super().__init__(compressed_data, quant_bits, origin_type)
            self.q = q

    def __init__(self, quant_bits: int = 8, n_clusters=None):
        super().__init__(quant_bits)
        from sklearn.cluster import KMeans

        if n_clusters is None:
            self.n_clusters = quant_bits
        else:
            self.n_clusters = n_clusters
        self.km = KMeans(self.n_clusters, n_init=1, max_iter=50)

    def _compress_one(self, data: np.ndarray, **kwargs) -> "KmeansCompressData":
        if data.flatten().shape[0] <= self.n_clusters:
            return self.KmeansCompressData(data, self.quant_bits)
        ori_shape = data.shape
        self.km.fit(np.expand_dims(data.flatten(), axis=1))

        quantized = self.km.labels_ - (1 << (self.quant_bits - 1))

        quantized = np.reshape(quantized, ori_shape)
        q = self.km.cluster_centers_

        return self.KmeansCompressData(
            quantized.astype(self.np_type), self.quant_bits, data.dtype, q
        )

    def _decompress_one(self, data: "KmeansCompressData") -> np.ndarray:
        if data.compressed_data.flatten().shape[0] <= self.n_clusters:
            return data.compressed_data
        label = data.compressed_data.astype(data.origin_type) + (
            1 << (self.quant_bits - 1)
        )
        dequantized = np.zeros_like(label)
        for i in range(data.q.shape[0]):
            dequantized[label == i] = data.q[i]

        return dequantized
