:notoc:

隐语
=============

`隐语 <https://github.com/secretflow/secretflow.git>`_ 是一个安全、开放、可扩展，用于隐私保护数据分析和机器学习的统一框架。隐语提供了

- 设备抽象，将多方安全计算(MPC)、同态加密(HE)和可信执行环境(TEE)等隐私计算技术抽象为密文设备，将明文计算抽象为明文设备
- 基于抽象设备的计算图，支持将数据分析和机器学习工作流表示为计算图。
- 基于计算图的机器学习/数据分析能力，支持数据水平/垂直/混合切分等多种场景。

==========
框架
==========
.. image:: _static/secretflow_arch.svg


.. panels::

    ---

    入门
    ^^^^^^^^^^^^^^^

    如果你还不认识 **隐语** ， 欢迎访问入门文档.

    +++
    
    .. link-button:: getting_started
            :type: ref
            :text: 马上开始
            :classes: btn-primary stretched-link

    ---

    教程
    ^^^^^^^^^^^^^^^

    想要了解更多 **隐语** 的用法? 欢迎访问我们的教程.

    +++
    
    .. link-button:: tutorial
            :type: ref
            :text: 马上开始
            :classes: btn-primary stretched-link

    ---

    开发者
    ^^^^^^^^^^^^^^^

    想要成为 **隐语** 的开发者? 欢迎访问我们的开发者文档.

    +++
    
    .. link-button:: development
            :type: ref
            :text: 马上开始
            :classes: btn-primary stretched-link

.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   getting_started/index
   tutorial/index
   development/index
