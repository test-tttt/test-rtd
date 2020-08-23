# Copyright (c) 2020 Paddle Quantum Authors. All Rights Reserved.
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

"""
辅助函数
"""

from functools import reduce

from numpy import diag, dot, identity
from numpy import kron as np_kron
from numpy import linalg, sqrt
from numpy import sum as np_sum
from numpy import transpose as np_transpose
from numpy import zeros as np_zeros

from paddle.complex import elementwise_add
from paddle.complex import kron as pp_kron
from paddle.complex import matmul
from paddle.complex import transpose as pp_transpose

from paddle.fluid.dygraph import to_variable
from paddle.fluid.framework import ComplexVariable
from paddle.fluid.layers import concat, cos, ones, reshape, sin
from paddle.fluid.layers import zeros as pp_zeros

__all__ = [
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "partial_trace",
    "compute_fid",
    "NKron",
]


def rotation_x(theta):
    r"""生成单量子比特X门

    Args:
        theta (Variable): X门的旋转角度

    Returns:
        ComplexVariable: 单量子比特X门的矩阵形式
    """

    cos_value = cos(theta/2)
    sin_value = sin(theta/2)
    zero_pd = pp_zeros([1], "float32")
    rx_re = concat([cos_value, zero_pd, zero_pd, cos_value], axis=0)
    rx_im = concat([zero_pd, -sin_value, -sin_value, zero_pd], axis=0)

    return ComplexVariable(reshape(rx_re, [2, 2]), reshape(rx_im, [2, 2]))


def rotation_y(theta):
    r"""生成单量子比特Y门

    Args:
        theta (Variable): Y门的旋转角度

    Returns:
        ComplexVariable: 单量子比特Y门的矩阵形式
    """

    cos_value = cos(theta/2)
    sin_value = sin(theta/2)
    ry_re = concat([cos_value, -sin_value, sin_value, cos_value], axis=0)
    ry_im = pp_zeros([2, 2], "float32")
    return ComplexVariable(reshape(ry_re, [2, 2]), ry_im)


def rotation_z(theta):
    r"""生成单量子比特Z门

    Args:
        theta (Variable): Z门的旋转角度

    Returns:
        ComplexVariable: 单量子比特Z门的矩阵形式
    """

    cos_value = cos(theta/2)
    sin_value = sin(theta/2)
    zero_pd = pp_zeros([1], "float32")
    rz_re = concat([cos_value, zero_pd, zero_pd, cos_value], axis=0)
    rz_im = concat([-sin_value, zero_pd, zero_pd, sin_value], axis=0)

    return ComplexVariable(reshape(rz_re, [2, 2]), reshape(rz_im, [2, 2]))


def partial_trace(rho_AB, dim1, dim2, A_or_B):
    r"""求AB复合系统下的偏迹

    Args:
        rho_AB (Variable): AB复合系统的密度矩阵
        dim1 (int): A系统的维度
        dim2 (int): B系统的维度
        A_orB (int): 1表示求系统A，2表示求系统B

    Returns：
        ComplexVariable: 求得的偏迹
    """

    # dim_total = dim1 * dim2
    if A_or_B == 2:
        dim1, dim2 = dim2, dim1

    idty_np = identity(dim2).astype("complex64")
    idty_B = to_variable(idty_np)

    zero_np = np_zeros([dim2, dim2], "complex64")
    res = to_variable(zero_np)

    for dim_j in range(dim1):
        row_top = pp_zeros([1, dim_j], dtype="float32")
        row_mid = ones([1, 1], dtype="float32")
        row_bot = pp_zeros([1, dim1 - dim_j - 1], dtype="float32")
        bra_j_re = concat([row_top, row_mid, row_bot], axis=1)
        bra_j_im = pp_zeros([1, dim1], dtype="float32")
        bra_j = ComplexVariable(bra_j_re, bra_j_im)

        if A_or_B == 1:
            row_tmp = pp_kron(bra_j, idty_B)
            res = elementwise_add(
                res,
                matmul(
                    matmul(row_tmp, rho_AB),
                    pp_transpose(
                        ComplexVariable(row_tmp.real, -row_tmp.imag),
                        perm=[1, 0]), ), )

        if A_or_B == 2:
            row_tmp = pp_kron(idty_B, bra_j)
            res += matmul(
                matmul(row_tmp, rho_AB),
                pp_transpose(
                    ComplexVariable(row_tmp.real, -row_tmp.imag), perm=[1, 0]),
            )
    return res


def compute_fid(rho, sigma):
    r"""计算两个密度算符的保真度

    Args:
        rho (numpy.ndarray): 密度算符 :math:`\rho`
        sigma (numpy.ndarray): 密度算符 :math:`\sigma`

    Returns:
        numpy.ndarray: 所求得的保真度
    """

    rho_eigenval, rho_eigenmatrix = linalg.eig(rho)
    sr_rho = dot(
        dot(rho_eigenmatrix, diag(sqrt(rho_eigenval.real + 1e-7))),
        np_transpose(rho_eigenmatrix).conjugate(), )
    rho_sigma_rho = dot(dot(sr_rho, sigma), sr_rho)

    return np_sum(sqrt(linalg.eigvals(rho_sigma_rho).real + 1e-7))


def NKron(AMatrix, BMatrix, *args):
    r"""对输入的矩阵依次求克罗内克积

    Args:
        AMatrix (numpy.ndarray): 输入的矩阵A
        BMatrix (numpy.ndarray): 输入的矩阵B
        *args (numpy.ndarray): 输入的其他矩阵

    Returns:
        numpy.ndarray: 所求得的克罗内克积
    """

    return reduce(
        lambda result, index: np_kron(result, index),
        args,
        np_kron(AMatrix, BMatrix), )
