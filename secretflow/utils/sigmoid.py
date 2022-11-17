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

import jax.numpy as jnp
from enum import Enum, unique
from secretflow.utils.errors import InvalidArgumentError


def t1_sig(x, limit: bool = True):
    '''
    taylor series referenced from:
    https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/
    '''
    T0 = 1.0 / 2
    T1 = 1.0 / 4
    ret = T0 + x * T1
    if limit:
        return jnp.select([ret < 0, ret > 1], [0, 1], ret)
    else:
        return ret


def t3_sig(x, limit: bool = True):
    '''
    taylor series referenced from:
    https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/
    '''
    T3 = -1.0 / 48
    ret = t1_sig(x, False) + jnp.power(x, 3) * T3
    if limit:
        return jnp.select([x < -2, x > 2], [0, 1], ret)
    else:
        return ret


def t5_sig(x, limit: bool = True):
    '''
    taylor series referenced from:
    https://mortendahl.github.io/2017/04/17/private-deep-learning-with-mpc/
    '''
    T5 = 1.0 / 480
    ret = t3_sig(x, False) + jnp.power(x, 5) * T5
    if limit:
        return jnp.select([ret < 0, ret > 1], [0, 1], ret)
    else:
        return ret


def seg3_sig(x):
    '''
    f(x) = 0.5 + 0.125x if -4 <= x <= 4
           1            if       x > 4
           0            if  -4 > x
    '''
    return jnp.select([x < -4, x > 4], [0, 1], 0.5 + x * 0.125)


def df_sig(x):
    '''
    https://dergipark.org.tr/en/download/article-file/54559
    Dataflow implementation of sigmoid function:
    F(x) = 0.5 * ( x / ( 1 + |x| ) ) + 0.5
    df_sig has higher precision than sr_sig if x in [-2, 2]
    '''
    return 0.5 * (x / (1 + jnp.abs(x))) + 0.5


def sr_sig(x):
    '''
    https://en.wikipedia.org/wiki/Sigmoid_function#Examples
    Square Root approximation functions:
    F(x) = 0.5 * ( x / ( 1 + x^2 )^0.5 ) + 0.5
    sr_sig almost perfect fit to sigmoid if x out of range [-3,3]
    '''
    return 0.5 * (x / jnp.sqrt(1 + jnp.square(x))) + 0.5


def ls7_sig(x):
    '''Polynomial fitting'''
    return (
        5.00052959e-01
        + 2.35176260e-01 * x
        - 3.97212202e-05 * jnp.power(x, 2)
        - 1.23407424e-02 * jnp.power(x, 3)
        + 4.04588962e-06 * jnp.power(x, 4)
        + 3.94330487e-04 * jnp.power(x, 5)
        - 9.74060972e-08 * jnp.power(x, 6)
        - 4.74674505e-06 * jnp.power(x, 7)
    )


def mix_sig(x):
    '''
    mix ls7 & sr sig, use ls7 if |x| < 4 , else use sr.
    has higher precision in all input range.
    NOTICE: this method is very expensive, only use for hessian matrix.
    '''
    ls7 = ls7_sig(x)
    sr = sr_sig(x)
    return jnp.select([x < -4, x > 4], [sr, sr], ls7)


def real_sig(x):
    return 1 / (1 + jnp.exp(-x))


@unique
class SigType(Enum):
    REAL = 'real'
    T1 = 't1'
    T3 = 't3'
    T5 = 't5'
    DF = 'df'
    SR = 'sr'
    # DO NOT use this except in hessian case.
    MIX = 'mix'


def sigmoid(x, sig_type: SigType):
    if sig_type is SigType.REAL:
        return real_sig(x)
    elif sig_type is SigType.T1:
        return t1_sig(x)
    elif sig_type is SigType.T3:
        return t3_sig(x)
    elif sig_type is SigType.T5:
        return t5_sig(x)
    elif sig_type is SigType.DF:
        return df_sig(x)
    elif sig_type is SigType.SR:
        return sr_sig(x)
    elif sig_type is SigType.MIX:
        return mix_sig(x)
    else:
        raise InvalidArgumentError(f'Unsupported sigtype: {sig_type}')
