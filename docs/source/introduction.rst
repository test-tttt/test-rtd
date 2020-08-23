.. _header-n0:

Paddle Quantum （量桨）
=======================

Paddle
Quantum（量桨）是基于百度飞桨开发的量子机器学习工具集，支持量子神经网络的搭建与训练，提供易用的量子机器学习开发套件与量子优化、量子化学等前沿量子应用工具集，使得百度飞桨也因此成为国内首个目前也是唯一一个支持量子机器学习的深度学习框架。

.. figure:: https://release-data.cdn.bcebos.com/Paddle%20Quantum.png
   :alt:

量桨建立起了人工智能与量子计算的桥梁，不但可以快速实现量子神经网络的搭建与训练，还提供易用的量子机器学习开发套件与量子优化、量子化学等前沿量子应用工具集，并提供多项自研量子机器学习应用。通过百度飞桨深度学习平台赋能量子计算，量桨为领域内的科研人员以及开发者便捷地开发量子人工智能的应用提供了强有力的支撑，同时也为广大量子计算爱好者提供了一条可行的学习途径。

.. _header-n6:

特色
----

-  易用性：提供简洁的神经网络搭建与丰富的量子机器学习案例。

-  通用性与拓展性：支持常用量子电路模型，提供多项优化工具。

-  特色工具集：提供量子优化、量子化学等前沿量子应用工具集，自研多项量子机器学习应用。

.. _header-n15:

安装步骤
--------

.. _header-n16:

Install PaddlePaddle
~~~~~~~~~~~~~~~~~~~~

请参考
`PaddlePaddle <https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/index_cn.html>`__
安装配置页面。此项目需求 PaddlePaddle 1.8.0 或更高版本。

.. _header-n19:

下载 Paddle Quantum 并安装
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   git clone http://github.com/PaddlePaddle/quantum

.. code:: shell

   cd quantum
   pip install -e .

.. _header-n23:

或使用 requirements.txt 安装依赖包
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   python -m pip install --upgrade -r requirements.txt

.. _header-n25:

使用 openfermion 读取xyz 描述文件 （仅可在linux下安装使用）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VQE中调用 openfermion 读取分子xyz文件并计算，因此需要安装 openfermion 和
openfermionpyscf。

.. code:: shell

   pip install openfermion
   pip install openfermionpyscf

.. _header-n29:

运行
~~~~

.. code:: shell

   cd paddle_quantum/QAOA/example
   python main.py

.. _header-n32:

入门与开发
----------

.. _header-n33:

教程入门
~~~~~~~~

量子计算是由量子力学与计算理论交叉而成的全新计算模型，具有强大的信息处理优势和广阔的应用前景，被视作未来计算技术的心脏。量子计算的相关介绍与入门知识可以参考
[1-3]。

量子机器学习是一门结合量子计算与机器学习的交叉学科，一方面利用量子计算的信息处理优势促进人工智能的发展，另一方面也利用现有的人工智能的技术突破量子计算的研发瓶颈。关于量子机器学习的入门资料可以参考
[4-6]。Paddle
Quantum（量桨）建立起了人工智能与量子计算的桥梁，为量子机器学习领域的研发提供强有力的支撑，也提供了丰富的案例供开发者学习。

.. _header-n37:

案例入门
~~~~~~~~

特别的，我们提供了涵盖量子优化、量子化学、量子机器学习等多个领域的案例供大家学习。比如：

-  量子近似优化（QAOA），完成安装步骤后打开 tutorial\QAOA.ipynb
   即可进行研究学习。

.. code:: shell

   cd tutorial
   jupyter notebook  QAOA.ipynb

-  量子特征求解器（VQE），完成安装步骤后打开 tutorial\VQE.ipynb
   即可进行研究学习。

.. code::

   cd tutorial
   jupyter notebook  VQE.ipynb

.. _header-n48:

开发
~~~~

Paddle Quantum 使用 setuptools 的develop
模式进行安装，相关代码修改可以直接进入\ ``paddle_quantum``
文件夹进行修改。python 文件携带了自说明注释。

.. _header-n51:

交流与反馈
----------

-  我们非常欢迎您欢迎您通过\ `Github
   Issues <https://github.com/PaddlePaddle/Quantum/issues>`__\ 来提交问题、报告与建议。

-  QQ技术交流群: 1076223166

.. _header-n57:

使用Paddle Quantum的工作
------------------------

我们非常欢迎开发者使用Paddle
Quantum进行量子机器学习的研发，如果您的工作有使用Paddle
Quantum，也非常欢迎联系我们。目前使用 Paddle Quantum 的代表性工作关于
Gibbs 态制备如下：

[1] Youle Wang, Guangxi Li, and Xin Wang. 2020. Variational quantum
Gibbs state preparation with a truncated Taylor series. arXiv2005.08797.
[`pdf <https://arxiv.org/pdf/2005.08797.pdf>`__]

.. _header-n61:

Copyright and License
---------------------

Paddle Quantum 使用 `Apache-2.0 license <LICENSE>`__\ 许可证。

.. _header-n64:

References
----------

[1] `量子计算 -
百度百科 <https://baike.baidu.com/item/量子计算/11035661?fr=aladdin>`__

[2] Michael A Nielsen and Isaac L Chuang. 2010. Quantum computation and
quantum information. Cambridge university press.

[3] Phillip Kaye, Raymond Laflamme, and Michele Mosca. 2007. An
Introduction to Quantum Computing.

[4] Jacob Biamonte, Peter Wittek, Nicola Pancotti, Patrick Rebentrost,
Nathan Wiebe, and Seth Lloyd. 2017. Quantum machine learning. Nature
549, 7671, 195–202. [`pdf <https://arxiv.org/pdf/1611.09347>`__]

[5] Maria Schuld, Ilya Sinayskiy, and Francesco Petruccione. 2015. An
introduction to quantum machine learning. Contemp. Phys. 56, 2, 172–185.
[`pdf <https://arxiv.org/pdf/1409.3097>`__]

[6] Marcello Benedetti, Erika Lloyd, Stefan Sack, and Mattia Fiorentini.
2019. Parameterized quantum circuits as machine learning models. Quantum
Sci. Technol. 4, 4, 043001. [`pdf <https://arxiv.org/pdf/1906.07682>`__]
