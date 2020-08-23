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
circuit
"""

from numpy import binary_repr, eye, identity

from paddle.complex import kron as pp_kron
from paddle.complex import matmul

from paddle.fluid import dygraph

from paddle_quantum.utils import rotation_x, rotation_y, rotation_z

__all__ = [
    "dic_between2and10",
    "base_2_change",
    "cnot_construct",
    "identity_generator",
    "single_gate_construct",
    "UAnsatz",
]


def dic_between2and10(n):
    r"""建立2进制和10进制对应的词典

    Args:
        n (int): 要转换的范围 ``[0, 2**n]``

    Returns:
        tuple(dict, dict): 2进制到10进制的 ``dict`` 和10进制到2进制的 ``dict``

    for example: if n=3, the dictionary is
    dic2to10: {'000': 0, '011': 3, '010': 2, '111': 7, '100': 4, '101': 5, '110': 6, '001': 1}
    dic10to2: ['000', '001', '010', '011', '100', '101', '110', '111']
    """

    dic2to10 = {}
    dic10to2 = [None] * 2**n

    for i in range(2**n):
        binary_text = binary_repr(i, width=n)
        dic2to10[binary_text] = i
        dic10to2[i] = binary_text

    return dic2to10, dic10to2  # the returned dic will have 2 ** n value


def base_2_change(string_n, i_ctrl, j_target):
    r""" 对一个二进制串，根据控制位和目标位进行转换

    Args:
        string_n (str): 二进制串
        i_ctrl (int): 第 ``i`` 位是控制位
        j_target (int): 第 ``j`` 位是目标位

    Returns:
        str: 转换后的二进制串
    """

    string_n_list = list(string_n)
    if string_n_list[i_ctrl] == "1":
        string_n_list[j_target] = str((int(string_n_list[j_target]) + 1) % 2)
    return "".join(string_n_list)


def cnot_construct(n, ctrl):
    r"""构建CNOT门

    对于2量子比特的量子线路，当control为[1, 2]时，其矩阵形式为：

    .. math::

        \begin{align}
        CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
        &=\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}
        \end{align}

    Args:
        n (int): 量子比特数
        ctrl (list): ``ctrl[0]`` 表示控制位， ``ctrl[1]`` 表示目标位

    Returns:
        numpy.ndarray: CNOT门的矩阵表示
    """

    mat = eye(2**n)
    dummy_mat = eye(2**n)
    dic2to10, dic10to2 = dic_between2and10(n)
    """ for example: if n=3, the dictionary is
    dic2to10: {'000': 0, '011': 3, '010': 2, '111': 7, '100': 4, '101': 5, '110': 6, '001': 1}
    dic10to2: ['000', '001', '010', '011', '100', '101', '110', '111']
    """

    for row in range(2**n):
        """ for each decimal index 'row', transform it into binary,
            and use 'base_2_change()' function to compute the new binary index 'row'.
            Lastly, use 'dic2to10' to transform this new 'row' into decimal.

            For instance, n=3, ctrl=[1,3]. if row = 5,
            its process is 5 -> '101' -> '100' -> 4
        """
        new_string_base_2 = base_2_change(dic10to2[row], ctrl[0] - 1,
                                          ctrl[1] - 1)
        new_int_base_10 = dic2to10[new_string_base_2]
        mat[row] = dummy_mat[new_int_base_10]

    return mat.astype("complex64")


def identity_generator(n):
    r"""生成n*n的单位矩阵

    Args:
        n (int): 单位矩阵的维度

    Returns:
        Variable: 生成的单位矩阵
    """

    idty_np = identity(2**n, dtype="float32")
    idty = dygraph.to_variable(idty_np)

    return idty


def single_gate_construct(mat, n, which_qubit):
    r"""构建单量子比特门

    Args:
        mat (Variable): 输入的矩阵
        n (int): 量子比特数
        which_qubit (int): 输入矩阵所在的量子比特号

    Returns:
        Variable: 得到的单量子比特门矩阵
    """

    idty = identity_generator(n - 1)

    if which_qubit == 1:
        mat = pp_kron(mat, idty)

    elif which_qubit == n:
        mat = pp_kron(idty, mat)

    else:
        I_top = identity_generator(which_qubit - 1)
        I_bot = identity_generator(n - which_qubit)
        mat = pp_kron(pp_kron(I_top, mat), I_bot)

    return mat


class UAnsatz:
    r"""这个是用户用来搭建量子电路的接口

    Attributes:
        n (int): 初始量子比特个数
        state (Variable): 保存当前的态矢量或者当前量子线路的矩阵表示
    """

    def __init__(self, n, input_state=None):
        r"""初始化函数

        Args:
            n (int): 该量子线路的量子比特数
            input_state (Variable, optional): 输入的态矢量，默认为None，会构建一个单位矩阵，表示当前量子线路的矩阵表示
        """

        self.n = n
        self.state = input_state if input_state is not None else identity_generator(
            self.n)

    def rx(self, theta, which_qubit):
        r"""添加关于x轴的单量子比特旋转门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} \cos\frac{\theta}{2} & -i*\sin\frac{\theta}{2} \\ -i*\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

        Args:
            theta (Variable): 旋转角度
            which_qubit (int): 作用在的qubit的编号,其值应该在[0, n)范围内，n为该量子线路的量子比特数。

        Returns:
            Variable: 当前量子线路输出的态矢量或者当前量子线路的矩阵表示

        ..  code-block:: python

            theta = np.array([np.pi], np.float64)
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                num_qubits = 1
                cir = UAnsatz(num_qubits)
                which_qubit = 1
                cir.rx(theta[0], which_qubit)
        """

        transform = single_gate_construct(
            rotation_x(theta), self.n, which_qubit)
        self.state = matmul(self.state, transform)

    def ry(self, theta, which_qubit):
        r"""添加关于y轴的单量子比特旋转门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

        Args:
            theta (Variable): 旋转角度
            which_qubit (int): 作用在的qubit的编号,其值应该在[0, n)范围内，n为该量子线路的量子比特数。

        Returns:
            Variable: 当前量子线路输出的态矢量或者当前量子线路的矩阵表示

        ..  code-block:: python

            theta = np.array([np.pi], np.float64)
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                num_qubits = 1
                cir = UAnsatz(num_qubits)
                which_qubit = 1
                cir.ry(theta[0], which_qubit)
        """

        transform = single_gate_construct(
            rotation_y(theta), self.n, which_qubit)
        self.state = matmul(self.state, transform)

    def rz(self, theta, which_qubit):
        r"""添加关于y轴的单量子比特旋转门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} \exp{-i\frac{\theta}{2}} & 0 \\ 0 & \exp{i\frac{\theta}{2}} \end{bmatrix}

        Args:
            theta (Variable): 旋转角度
            which_qubit (int): 作用在的qubit的编号,其值应该在[0, n)范围内，n为该量子线路的量子比特数。

        Returns:
            Variable: 当前量子线路输出的态矢量或者当前量子线路的矩阵表示

        ..  code-block:: python

            theta = np.array([np.pi], np.float64)
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                num_qubits = 1
                cir = UAnsatz(num_qubits)
                which_qubit = 1
                cir.ry(theta[0], which_qubit)
        """

        transform = single_gate_construct(
            rotation_z(theta), self.n, which_qubit)
        self.state = matmul(self.state, transform)

    def cnot(self, control):
        r"""添加一个CONT门。

        对于2量子比特的量子线路，当control为[0, 1]时，其矩阵形式为：

        .. math::

            \begin{align}
            CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
            &=\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}
            \end{align}

        Args:
            control (list): 作用在的qubit的编号，control[0]为控制位，control[1]为目标位，其值都应该在[0, n)范围内，n为该量子线路的量子比特数。

        ..  code-block:: python

            num_qubits = 2
            with fluid.dygraph.guard():
                cir = UAnsatz(num_qubits)
                cir.cnot([0, 1])
        """

        cnot = dygraph.to_variable(cnot_construct(self.n, control))
        self.state = matmul(self.state, cnot)
